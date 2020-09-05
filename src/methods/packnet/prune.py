"""Handles all the pruning-related stuff."""
import torch.nn as nn
import torch


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn, current_dataset_idx):
        self.model = model
        self.prune_perc = prune_perc
        self.train_bias = train_bias
        self.train_bn = train_bn

        self.current_masks = None
        self.previous_masks = previous_masks
        self.current_dataset_idx = torch.tensor(current_dataset_idx, requires_grad=False, dtype=torch.uint8)

        valid_key = list(previous_masks.keys())[0]
        if self.previous_masks[valid_key][0].is_cuda:
            self.current_dataset_idx = self.current_dataset_idx.cuda()
        print('Init pruner: current_dset_idx={}'.format(self.current_dataset_idx))

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(self.current_dataset_idx.cuda())]
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_perc * tensor.numel())
        print("abs_tensor", abs_tensor)
        print("cutoff_rank", cutoff_rank)
        print("abs_tensor.view(-1)", abs_tensor.view(-1).cpu())
        print("abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)", abs_tensor.view(-1).cpu().kthvalue(cutoff_rank))
        print("abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]", abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0])

        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0].item()

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * \
                      previous_mask.eq(self.current_dataset_idx.cuda())

        # mask = 1 - remove_mask
        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}

        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask(
                    module.weight.data, self.previous_masks[module_idx], module_idx)
                self.current_masks[module_idx] = mask.cuda()
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0

    def make_grads_zero(self, cuda=False):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    if cuda:
                        module.weight.grad.data[layer_mask.ne(
                            self.current_dataset_idx.cuda())] = 0
                    else:
                        module.weight.grad.data[layer_mask.ne(
                            self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self, dataset_idx, debug=False):  # if debug:
        """To be done to retrieve weights just for a particular dataset."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # keep in mask: 0 < mask idx <= {}
                if debug:
                    print("MODULE {}".format(str(module_idx)))
                weight = module.weight.data
                mask = self.previous_masks[module_idx]
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx).cuda()] = 0.0
                if debug:
                    [print("KEEPING {}/{} WEIGHTS FOR TASK IDX {}"
                           .format(str(sum(mask.eq(idx).view(1, -1).int()[0]).item()), str(len(mask.view(1, -1)[0])),
                                   str(idx)))
                     for idx in range(1, dataset_idx + 1)]

    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.copy_(biases[module_idx])

    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    biases[module_idx] = module.bias.data.clone()
        return biases

    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                if torch.any(mask.eq(self.current_dataset_idx)):  # Can also be a checkpoint
                    print("[WARNING] Already assigned weights for this task in mask?")

                # Tensor, assigning this task idx to the ones where 0 (no task assigned yet)
                mask[mask.eq(0)] = self.current_dataset_idx

        self.current_masks = self.previous_masks

    def mask_summary(self, skiptask=4):
        assert self.previous_masks
        print("Assuming final model after task pruning and learning all tasks")

        modules = []
        cap_assigned = [[] for task in range(self.current_dataset_idx.item())]
        conv_cnt = 0
        fc_cnt = 0
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print("==> idx {}".format(module_idx))
                if isinstance(module, nn.Conv2d):
                    modules.append("Conv{}".format(conv_cnt))
                    conv_cnt += 1
                else:
                    modules.append("FC{}".format(fc_cnt))
                    fc_cnt += 1
                mask = self.previous_masks[module_idx]

                for idx in list(range(1, self.current_dataset_idx + 1, skiptask)) + [self.current_dataset_idx]:
                    assigned_this_task = sum(mask.le(idx).view(1, -1).int()[0]).item()  # All capacity used at this task
                    # assigned_no_task = sum(mask.eq(0).view(1, -1).int()[0]).item() # because of rounding each time round(self.prune_perc * tensor.numel())
                    assigned_no_task = 0
                    assigned_this_task -= assigned_no_task
                    total = len(mask.view(1, -1)[0])
                    try:
                        print("KEEPING {}/{} WEIGHTS FOR TASK IDX {} ({} assigned to no task)".format(
                            str(assigned_this_task), str(total),
                            str(idx), str(assigned_no_task)))
                    except:
                        print("[ERROR] Assigned this task={}".format(assigned_this_task))
                        assigned_this_task = 0
                    cap_assigned[idx - 1].append(assigned_this_task / total * 100)

        return modules, cap_assigned

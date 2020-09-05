# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog
from torch.autograd import Variable
from utilities import plot as Plots
import copy

import methods.rehearsal.model.common as common

# Auxiliary functions useful for GEM's inner optimization.
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector (current task)
        input:  memories, (t * p)-vector (previous task gradients based on memories)
        output: x, p-vector (projecting current task gradient if necessary)

        margin: higher = more memory strength
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    """
    Shared head, with all amounts of classes.
    Heads are masked out
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        # Set raw network
        self.net = torch.load(args.prev_model_path)

        # Replace output layer
        last_layer_index = str(len(self.net.classifier._modules) - 1)
        num_ftrs = self.net.classifier._modules[last_layer_index].in_features
        original_head = copy.deepcopy(self.net.classifier._modules[last_layer_index])
        self.net.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, n_outputs)
        self.n_outputs = n_outputs
        print("NEW FC CLASSIFIER HEAD with {} units".format(n_outputs))

        # Copy old head weights into new head
        original_head_output_size = original_head.out_features
        self.net.classifier._modules[last_layer_index].weight.data[:original_head_output_size].copy_(
            original_head.weight.data)  # shape [200,128] vs [20,128]
        self.net.classifier._modules[last_layer_index].bias.data[:original_head_output_size].copy_(
            original_head.bias.data)  # shape [200] vs [20]
        del original_head
        print("COPIED FIRST MODEL CLASSIFIER INTO EXTENDED HEAD: orig size = ", original_head_output_size)

        if args.cuda:
            self.net.cuda()

        self.ce = nn.CrossEntropyLoss()
        self.n_memories = args.n_memories  # global: memories per task
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = None  # declare in forward for 1st sample
        self.memory_labels = torch.LongTensor(n_tasks, self.n_memories)

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims),
                                  n_tasks)  # Matrix with for each task a grad vector (len=all params)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.n_tasks = n_tasks
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.cum_nc_per_task = [sum(args.nc_per_task[:idx + 1]) for idx, nc in enumerate(args.nc_per_task)]
        assert self.cum_nc_per_task[-1] == sum(args.nc_per_task)
        print("cum_nc_per_task={}".format(self.cum_nc_per_task))
        self.init_setup(args)

    def init_setup(self, args):
        """
        Setup changes that also have to be loaded when loading model.
        :param args:
        :return:
        """
        self.dropout_masks = {}
        self.opt = optim.SGD(self.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
        self.margin = args.memory_strength
        print("SET optimizer and margin (mem_strength)")

    def compute_offsets(self, task_idx, cum_nc_per_task):
        """
            Compute offsets (for cifar) to determine which
            outputs to select for a given task.

            Output idxs: [offset1, offset2[
        """
        return common.compute_offsets(task_idx, cum_nc_per_task)

    def reset_dropout_config(self):
        self.dropout_masks = {}

    def forward(self, x, t, args=None, train_mode=False, p_retain_unit=0.5):
        if self.gpu:
            x = x.cuda()  # Makes sense as batch samples are also only sent to CUDA device at runtime

        # Feature repr
        feat = self.net.features(x)
        feat = feat.view(feat.size(0), -1)

        # Classifier with manual dropout
        output = feat
        for idx, module in enumerate(self.net.classifier.children()):
            if isinstance(module, nn.Dropout):
                if module.training:  # Nothing to do in eval (scaling done during training)
                    if idx not in self.dropout_masks:  # save and hold mask until refresh
                        # Generate unit mask
                        mask = Variable(
                            torch.bernoulli(output.data.new(output.data.size())[0].fill_(p_retain_unit))
                        ) / p_retain_unit  # Scaling included

                        self.dropout_masks[idx] = mask
                    else:
                        mask = self.dropout_masks[idx]
                    try:
                        output = output * mask.expand(output.shape[0], *mask.shape)
                    except:
                        raise Exception()
            else:
                output = module(output)

        # make sure we predict classes within the current task (1 head)
        offset1, offset2 = self.compute_offsets(t, self.cum_nc_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y, paths, args=None):
        """

        :param x: batch of tr samples
        :param t: task_idx (int starting from 0)
        :param y: batch of tr labels
        :return:
        """
        self.net.train()
        self.reset_dropout_config()  # New config per batch
        batch_stats = {'projected_grads': [0]}

        # update memory
        if t != self.old_task:
            self.init_new_task(t, x)
        self.fill_buffer(t, paths, y)

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = self.compute_offsets(past_task, self.cum_nc_per_task)

                # Loading exemplar set
                transform = args.task_imgfolders['train'].transform
                input_imgfolder = self.memory_data.convert_imagefolder(transform, self.memory_labels, entry=past_task)
                input_dataloader = self.memory_data.get_dataloader(input_imgfolder, batch_size=args.batch_size)

                for data in input_dataloader:
                    inputs, targets = data
                    inputs = inputs.squeeze()
                    inputs = Variable(inputs.cuda()) if self.gpu else Variable(inputs)

                    outputs = self.forward(inputs, past_task)[:, offset1: offset2]
                    if self.gpu:  # Free up space on GPU, but don't remove original CPU mem data of model
                        del inputs

                    if self.gpu:
                        outputs = outputs.cuda()
                        targets = targets.cuda()
                    ptloss = self.ce(outputs, targets)

                    if self.gpu:  # Free up space on GPU
                        del outputs
                        del targets

                    ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = self.compute_offsets(t, self.cum_nc_per_task)
        outputs = self.forward(x, t)[:, offset1: offset2]

        _, preds = torch.max(outputs.data, 1)
        correct_classified = torch.sum(preds == y.data)
        loss = self.ce(outputs, y)  # No offset to y! (task-specific labels, task-specific output)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])  # current is not yet observed
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),  # matrix multiplication:
                            self.grads.index_select(1, indx))
            constraint_violations = (dotp < 0).sum()
            if constraint_violations != 0:
                batch_stats['projected_grads'] = [constraint_violations.item()]
                project2cone2(self.grads[:, t].unsqueeze(1),  # Gradient current task
                              self.grads.index_select(1, indx), self.margin)  # All gradient vectors of previous tasks
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)
        self.opt.step()

        return loss, correct_classified, batch_stats

    def observe_FT(self, x, t, y, paths=None, args=0):
        """

        :param x:
        :param t: current task idx
        :param y: labels (idxes FC layer assumed task-specific head)
        :param n_exemplars_to_append:
        :return:
        """
        self.zero_grad()

        # Actual current task batch
        offset1, offset2 = self.compute_offsets(t, self.cum_nc_per_task)
        output = self.forward(x, t)[:, offset1: offset2]
        _, preds = torch.max(output.data, 1)
        correct_classified = torch.sum(preds == y.data)  # In dedicated head (offsetted)
        loss = self.ce(output, y)

        loss.backward()
        self.opt.step()

        return loss, correct_classified

    def init_new_task(self, t, batch):
        print("From old task {} to new task {}, observed tasks = {}".format(self.old_task, t, self.observed_tasks))
        self.observed_tasks.append(t)
        self.old_task = t

        if self.memory_data is None:
            input_dim = batch.shape[1:]
            self.memory_data = common.RehearsalMemory(self.n_tasks, self.n_memories, input_dim)
            print("Mem data shape = ", self.memory_data.shape)

    def fill_buffer(self, t, paths, y):
        """
        Fill exemplar buffer.
        :return:
        """
        buffer_cycle = False
        # MEMORY UPDATE, actual batch x,y remains untouched
        # Update ring buffer storing examples from current task
        # Buffer will always keep the last batches (random exemplar keeping)
        bsz = y.data.size(0)  # batch size
        endcnt = min(self.mem_cnt + bsz,
                     self.n_memories)  # storing batch in mem: but can't exceed limit of n_mems per task
        effbsz = endcnt - self.mem_cnt  # effective batch size (what still fits in mem)
        self.memory_data[t, self.mem_cnt: endcnt] = paths[: effbsz]
        if bsz == 1:
            self.memory_labels[t, self.mem_cnt] = y.data[0].cpu()
        else:
            self.memory_labels[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz].cpu())
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:  # circular buffer
            self.mem_cnt = 0
            buffer_cycle = True

        return buffer_cycle

    def manage_memory(self, t, args):
        """
        Fill buffer with exemplars for first task model.
        :param t:
        :param args:
        :return:
        """
        # Iterate over data.
        for data in args.dset_loaders['train']:
            # get the inputs
            inputs, labels, paths = data  # Labels are output layer indices!
            inputs = inputs.squeeze()

            # wrap them in Variable
            if self.gpu:
                inputs, labels = Variable(inputs.cuda()), \
                                 Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            if t != self.old_task:
                self.init_new_task(t, inputs)
            buffer_full = self.fill_buffer(t, paths, labels)
            if buffer_full:
                print("BUFFER FILLED WITH EXEMPLARS")
                return
        print("[WARNING] BUFFER WAS NOT FILLED WITH EXEMPLARS...")

    def check_exemplars(self, t, max_count=10):
        print("CHECKING EXEMPLARS")
        offset1, offset2 = self.compute_offsets(t, self.cum_nc_per_task)
        count = 0
        for key in range(offset1, offset2):
            n_exemplars = len(self.memory_data[t])
            print('#exemplars = {}, for task {}'.format(n_exemplars, key))
            for ex in range(n_exemplars):

                Plots.imshow_tensor(self.memory_data[t, ex], title="exemplar {} of task {}".format(ex, key))
                count += 1
                if max_count == count:
                    return

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import copy

import numpy as np
import random
from torch.autograd import Variable
from collections import OrderedDict
from utilities import plot as Plots

from data.imgfolder import ImageFolder_Subset_ClassIncremental, ImageFolder_Subset_PathRetriever
import methods.rehearsal.model.common as common
import tqdm


class Net(torch.nn.Module):
    """
    Based on GEM re-implementation of iCARL. However, we adapt memory usage to the original paper:
    1st task gets whole buffer capacity,...
    """

    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.n_tasks = n_tasks
        self.n_memories_per_task = int(args.n_memories)  # Memory PER TASK
        self.n_total_memories = args.n_memories * n_tasks  # TOTAL memory

        # Set raw network
        raw_net = torch.load(args.prev_model_path)
        self.n_feat = raw_net.classifier._modules[str(0)].in_features
        self.n_outputs = n_outputs

        # Define feature extractor and output layer
        self.net_feat = raw_net.features
        self.net_classifier = raw_net.classifier

        # Idxs/sizes
        self.init_head(n_outputs)

        # Replace output layer
        print("NEW FC CLASSIFIER HEAD with {} units".format(n_outputs))

        if args.cuda:
            self.net_feat = self.net_feat.cuda()
            self.net_classifier = self.net_classifier.cuda()

        # setup losses
        self.ce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')  # for distillation
        self.lsm = torch.nn.LogSoftmax(dim=1)  # Instead of using Temp=2, use LogSoftmax instead (also more sensitive)
        self.sm = torch.nn.Softmax(dim=1)

        # memory
        self.mem_class_x = None
        self.mem_class_y = OrderedDict()  # Distillation outputs for the exemplars (not actual labels)

        self.gpu = args.cuda
        self.n_outputs = n_outputs

        # Counters
        self.exemplar_count = 0
        self.observed_tasks = []
        self.observed_tasks_imgfolders = []
        self.old_task = -1

        # Common with loaded model setup
        self.init_setup(args)

    def init_head(self, new_n_outputs):
        is_cuda = next(self.net_classifier.parameters()).is_cuda
        last_layer_index = str(len(self.net_classifier._modules) - 1)
        last_layer_in_feats = self.net_classifier._modules[last_layer_index].in_features

        original_head = copy.deepcopy(self.net_classifier._modules[last_layer_index])
        self.net_classifier._modules[last_layer_index] = nn.Linear(last_layer_in_feats,
                                                                   new_n_outputs)  # New outputlayer

        # Copy old head weights into new head
        original_head_output_size = original_head.out_features
        self.net_classifier._modules[last_layer_index].weight.data[:original_head_output_size].copy_(
            original_head.weight.data)  # shape [200,128] vs [20,128]
        self.net_classifier._modules[last_layer_index].bias.data[:original_head_output_size].copy_(
            original_head.bias.data)  # shape [200] vs [20]
        if is_cuda:
            self.net_classifier = self.net_classifier.cuda()
        del original_head
        print("COPIED MODEL CLASSIFIER INTO EXTENDED HEAD: orig size = {}, new size={}".format(
            original_head_output_size, new_n_outputs))

    def init_setup(self, args):
        """ Setup changes that also have to be loaded when loading model."""
        self.opt = optim.SGD(self.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
        self.reg = args.memory_strength  # LWF reg
        print("SET optimizer and reg (mem_strength)")

        if args.n_outputs != self.n_outputs:
            self.init_head(args.n_outputs)
            self.n_outputs = args.n_outputs

        if args.n_tasks != self.n_tasks:
            print("Going from {} tasks to {} tasks, but memory retains initial capacity: {}".format(
                self.n_tasks, args.n_tasks, self.n_total_memories
            ))
            self.n_tasks = args.n_tasks

        self.nc_per_task = args.nc_per_task
        self.cum_nc_per_task = [sum(args.nc_per_task[:idx + 1]) for idx, nc in enumerate(args.nc_per_task)]
        assert self.cum_nc_per_task[-1] == sum(args.nc_per_task)
        print("cum_nc_per_task={}".format(self.cum_nc_per_task))

    def compute_offsets(self, task_idx, cum_nc_per_task=None):
        cum_nc_per_task = self.cum_nc_per_task if cum_nc_per_task is None else cum_nc_per_task
        return common.compute_offsets(task_idx, cum_nc_per_task)

    def forward(self, x, t, args, train_mode=False):
        """
        Forward for eval: using Nearest Neighbour classification of exemplar sets.
        Uses only feature extractor! Not output layer.

        Called when:
            model = Net()
            model(x,t)
        """
        if train_mode:
            return self.forward_training(x, t)

        # nearest neighbor
        nd = self.n_feat
        ns = x.size(0)

        if (self.cum_nc_per_task[t] - self.nc_per_task[t]) not in self.mem_class_x.exemplars.keys():
            # no exemplar in memory yet, output uniform distr. over classes in
            # task t above, we check presence of first class for this task, we
            # should check them all
            out = torch.Tensor(ns, self.n_outputs).fill_(-10e10)
            out[:, int(self.cum_nc_per_task[t] - self.nc_per_task[t]): int(self.cum_nc_per_task[t])].fill_(
                1.0 / self.nc_per_task[t])
            if self.gpu:
                out = out.cuda()
            return out
        means = None

        # Check if it is postpruned model with exemplar set included for latest task
        offset1, offset2 = self.compute_offsets(t)
        for cc in range(offset1, offset2):
            transform = args.task_imgfolders['train'].transform
            class_imgfolder = self.mem_class_x.convert_imagefolder(transform, targetlist=None, entry=cc)
            class_dataloader = self.mem_class_x.get_dataloader(class_imgfolder, batch_size=args.batch_size)
            class_mean_feature = self.get_mean_feat(class_dataloader, include_labels=False, include_paths=False)
            if means is None:
                means = torch.ones(self.nc_per_task[t], *class_mean_feature.shape) * float('inf')
            means[cc - offset1] = class_mean_feature

        if self.gpu:
            means = means.cuda()
        classpred = torch.LongTensor(ns)
        preds = self.get_feature(x).data.clone()
        for ss in range(ns):
            dist = (means - preds[ss].expand(self.nc_per_task[t], nd)).norm(2, 1)
            _, ii = dist.min(0)
            ii = ii.squeeze()
            classpred[ss] = ii.item() + offset1

        # Output 1-hot vectors (nearest neighbour)
        out = torch.zeros(ns, self.n_outputs)

        if self.gpu:
            out = out.cuda()
        for ss in range(ns):  # For all samples, put the one with closest neighbour to 1
            out[ss, classpred[ss]] = 1
        return out  # return 1-of-C code, ns x nc

    def forward_training(self, x, t):
        """
        Forward used during training, gives actual output of the output classification layer.
        """
        output = self.get_feature(x)
        output = self.net_classifier(output)

        # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def get_feature(self, x):
        """Returns feature of the input img batch x, reformatted in single row tensor."""
        output = self.net_feat(x)
        output = output.view(output.size(0), -1)
        return output

    def observe_FT(self, x, t, y, paths, args):
        """FT only training on current data, not exemplars."""
        self.zero_grad()
        offset1, offset2 = self.compute_offsets(t)
        output = self.forward_training(x, t)[:, offset1: offset2]
        loss = self.ce(output, y)

        _, preds = torch.max(output.data, 1)
        correct_classified = torch.sum(preds == y.data)

        # bprop and update
        loss.backward()
        self.opt.step()

        return loss, correct_classified

    def train(self, train=True):  # self.train(False)
        self.net_feat.train(train)
        self.net_classifier.train(train)

    def observe(self, x, t, y, paths, args):
        """
        Training, shared head for all classes in current task (not class-incremental).
        Memory update is task based, i.e. new per-class memories updated on ending of task.

        :param x: batch of tr samples
        :param t: task_idx (int starting from 0)
        :param y: batch of tr labels
        """
        self.train()
        batch_stats = None

        # update memories on ending of task
        if t != self.old_task:
            self.init_new_task(t, args)
        loss, correct_classified = self.update_representation(x, t, y, args)

        return loss, correct_classified, batch_stats

    def init_new_task(self, t, args):
        print("From old task {} to new task {}, observed tasks = {}".format(self.old_task, t, self.observed_tasks))
        self.observed_tasks.append(t)
        # Instead of memx, memy which store ALL the task data in mem (not just buffer data...)
        self.observed_tasks_imgfolders.append(args.task_imgfolders)
        self.old_task = t

        if self.mem_class_x is None:
            batch, _, _ = args.dset_loaders['train']
            input_dim = batch.shape[1:]
            self.mem_class_x = common.RehearsalMemory(self.nc_per_task[t], self.n_memories_per_task, input_dim)

    def check_exemplars(self, t, max_count=10):
        print("CHECKING EXEMPLARS")
        offset1, offset2 = self.compute_offsets(t)
        count = 0
        for key in range(offset1, offset2):
            n_exemplars = self.mem_class_x[key].shape[0]
            print('#exemplars = {}, for class {}'.format(n_exemplars, key))
            for ex in range(n_exemplars):
                Plots.imshow_tensor(self.mem_class_x[ex], title="exemplar {} of class {}".format(ex, key))
                count += 1
                if max_count == count:
                    return

    def get_mean_feat(self, dataloader, include_labels=True, include_paths=True):
        """
        Calculate mean for all samples in the dataloader.
        :param dataloader:
        :return:
        """
        ##########################################
        # CALCULATE MEAN FEAT
        batch_of_means = None
        exemplar_shape = None
        for data in dataloader:
            if include_paths:
                if include_labels:
                    inputs, labels, paths = data  # Labels are output layer indices!
                else:
                    inputs, paths = data
            else:
                if include_labels:
                    inputs, labels = data  # Labels are output layer indices!
                else:
                    inputs = data

            inputs = Variable(inputs)

            if self.gpu:
                inputs = inputs.cuda()

            if exemplar_shape is None:
                exemplar_shape = inputs[0].size()

            # Mean over batch-dim
            batch_mean = self.get_feature(inputs).data.clone().mean(0).unsqueeze(0)
            if batch_of_means is None:
                batch_of_means = batch_mean
            else:
                batch_of_means = torch.cat((batch_of_means, batch_mean), dim=0)
            del inputs
        # Mean of means = total mean
        mean_feature = batch_of_means.mean(0).squeeze()
        return mean_feature

    def manage_memory(self, t, args):
        """
        Manage the exemplar sets.

        In original GEM implementation: assume only 1 epoch
        Here multiple epochs (5 -> See supplemental)

        Constructs per class of the task:
        1) Mean feature
        2) Priority List, by:
            *) Cost of all class-samples (with already picked exemplars) to this mean
            *) Get corresponding indices
            *) Actual storing via dataloader
        """

        if t != self.old_task:
            self.init_new_task(t, args)

        # Reduce exemplar set by updating value of num. exemplars per class
        self.exemplar_count = int(self.n_total_memories / self.cum_nc_per_task[t])  # K/m paper
        print("EACH OF THE {} CLASSES GETS {} EXEMPLARS".format(self.cum_nc_per_task[t], self.exemplar_count))
        assert self.exemplar_count > 0, "Each class should get at least 1 exemplar"
        current_num_classes = self.nc_per_task[t]  # Prev task t

        # Truncate other memories (Keep first in prioritized lists)
        print("STEP1: TRUNCATING PREV MEMORIES PER CLASS")
        for key, mem in self.mem_class_x.exemplars.items():
            self.mem_class_x[key] = self.mem_class_x[key][:self.exemplar_count]

        for key, mem in self.mem_class_y.items():
            self.mem_class_y[key] = self.mem_class_y[key][:self.exemplar_count]

        ####################################
        # Construct exemplar set for last task
        ####################################
        print("STEP 2: Construct exemplar set for last task")
        offset1, offset2 = self.compute_offsets(t)
        for taskhead_class_idx in range(current_num_classes):
            print("-" * 10, "HEAD LABEL ", taskhead_class_idx, "-" * 10)
            sharedhead_class_idx = taskhead_class_idx + offset1

            current_task_imgfolder = self.observed_tasks_imgfolders[t]['train']
            current_task_imgfolder_class = ImageFolder_Subset_ClassIncremental(current_task_imgfolder,
                                                                               target_idx=taskhead_class_idx)
            current_task_imgfolder_class = ImageFolder_Subset_PathRetriever(current_task_imgfolder_class)
            print("current_task_imgfolder_class loaded")

            # No race conditions: num_workers=0
            class_dataloader = torch.utils.data.DataLoader(current_task_imgfolder_class,
                                                           batch_size=args.batch_size, shuffle=False, num_workers=0)
            print("dataloader created, batchsize={}".format(class_dataloader.batch_size))

            max_exemplar_count = self.exemplar_count if self.exemplar_count < len(
                current_task_imgfolder_class) else len(current_task_imgfolder_class)

            if max_exemplar_count <= 0:
                print("SKIPPING: no samples ({}) of class {}".format(max_exemplar_count, taskhead_class_idx))
                continue

            ##########################################
            # CALCULATE MEAN FEAT

            # Mean of means = total mean
            mean_dataloader = torch.utils.data.DataLoader(current_task_imgfolder_class,
                                                          batch_size=args.batch_size, shuffle=False, num_workers=8)
            mean_feature = self.get_mean_feat(mean_dataloader)
            print("Mean feats calculated")

            ##########################################
            # MAKE PRIORITY LIST (Make indx_ranking)
            exemplars = torch.zeros(max_exemplar_count, *self.mem_class_x.input_dim)
            if self.gpu:
                exemplars = exemplars.cuda()
            print("NEW EXEMPLAR SET SHAPE: ", exemplars.shape)
            n_class_samples = len(current_task_imgfolder_class)  # Size for this class

            # Store priority list of exemplars (from cdata)
            taken = torch.zeros(n_class_samples)  # used to keep track of which examples we have already used
            indx_ranking = np.zeros(max_exemplar_count, dtype='int32')
            exemplar_paths = [None] * indx_ranking.size
            for ex_idx in tqdm.tqdm(range(max_exemplar_count), "Exemplars added"):

                # SUM OF exemplars already selected for this exemplar set
                prev = torch.zeros(1, self.n_feat)
                if self.gpu:
                    prev = prev.cuda()
                if ex_idx > 0:
                    selected_exemplars = exemplars[:ex_idx]
                    # Sanity check for empty exemplars
                    for ex_it in range(ex_idx):
                        assert len(torch.nonzero(selected_exemplars[ex_it])) > 0
                    feat_selected_exemplars = self.get_feature(selected_exemplars)
                    prev = feat_selected_exemplars.data.clone().sum(0)  # SUM

                # CALC COST (Gather all costs w.r.t. exemplar set so far)
                cost = torch.Tensor(n_class_samples).fill_(10e10)
                for batch_cnt, data in enumerate(class_dataloader):
                    inputs, labels, paths = data  # Labels are output layer indices!
                    inputs = Variable(inputs)
                    if self.gpu:
                        inputs = inputs.cuda()
                    model_output = self.get_feature(inputs).data.clone()
                    idx_offset = batch_cnt * class_dataloader.batch_size
                    assert idx_offset < n_class_samples

                    n_samples = model_output.shape[0]
                    # Closest distance to mean feat,
                    # from the ones already selected (prev), together with the new one (batch entries in model_output)
                    batch_cost = (mean_feature.unsqueeze(0).expand(n_samples, self.n_feat)  # Repeat mean feat
                                  - (model_output + prev.expand(n_samples, self.n_feat))  # Sum all prev w current
                                  / (ex_idx + 1)  # Make sum an avg
                                  ).norm(2, 1)  # The 2 norm of |dist mean_feat - dist avg with new|

                    cost[idx_offset: idx_offset + class_dataloader.batch_size] = batch_cost
                assert not torch.any(cost == 10e10)
                assert cost.shape[0] == n_class_samples, "NEED COST FOR EACH OF THE SAMPLES"

                # GET INDEX SORTING (SMALLEST TO HIGHEST COST)
                _, indx = cost.sort(0)  # Sort entries, use indx to select winner in cdata

                winner = 0  # Iterate indices of the calculated costs
                while winner < indx.size(0) and taken[indx[winner]] == 1:
                    winner += 1

                # Get winner example
                found_winner = False
                for batch_cnt, data in enumerate(class_dataloader):
                    if indx[winner] < (batch_cnt + 1) * class_dataloader.batch_size:
                        inputs, labels, paths = data
                        inputs = Variable(inputs)
                        if self.gpu:
                            inputs = inputs.cuda()

                        idx = indx[winner] - batch_cnt * class_dataloader.batch_size
                        winner_img = inputs[idx]
                        winner_path = paths[idx]
                        found_winner = True
                        break
                assert found_winner, "Winner not found in data"

                # Found a new winner
                if winner < indx.size(0):
                    taken[indx[winner]] = 1
                    indx_ranking[ex_idx] = int(indx[winner])

                    # Find exemplar
                    exemplars[ex_idx] = winner_img
                    exemplar_paths[ex_idx] = winner_path
                else:  # All the exemplars are ranked
                    # Truncate amount of exemplars: All are taken, but still not filled
                    print("[WARN] # SAMPLES (={}) < # EXEMPLARS PER CLASS (={})".format(n_class_samples, indx.size(0)))
                    indx_ranking = indx_ranking[:indx.size(0)]
                    exemplars = exemplars[:indx.size(0)].clone()
                    exemplar_paths = exemplar_paths[:indx.size(0)]
                    self.exemplar_count = indx.size(0)
                    break  # It's a sorted list so once an item doesn't suffice, don't iterate others anymore

                del prev, cost

            # update memory with exemplars
            self.mem_class_x[sharedhead_class_idx] = exemplar_paths

            # recompute outputs for distillation purposes
            self.train(False)
            self.mem_class_y[sharedhead_class_idx] = self.forward_training(exemplars, t).data.clone()
            self.train(True)
            print()

    def update_representation(self, x, t, y, args, T=2):
        """
        Train using classifier and supervision of labels.
        :param x: batch input
        :param t: task idx
        :param y: labels
        :param args:
        :return:
        """
        ####################################
        # ALGOR 3: Update Representation
        ####################################
        # Current batch (current task data): [s,t]
        self.zero_grad()
        offset1, offset2 = self.compute_offsets(t)
        output = self.forward_training(x, t)[:, offset1: offset2]
        del x
        loss = self.ce(output, y)

        _, preds = torch.max(output.data, 1)
        correct_classified = torch.sum(preds == y.data)

        # DISTILLATION: Exemplar part of batch [1,s-1]
        total_ex_loss = 0
        total_ex_loss_count = 0
        if self.exemplar_count > 0:
            # Append fixed amount for classes and rnd for leftovers (CONSTRAINED TO EXEMPLAR_COUNT)
            n_classes = len(self.mem_class_x)
            n_fixed_exemplars_per_class = int(np.floor(args.n_exemplars_to_append_per_batch / n_classes))
            if n_fixed_exemplars_per_class > self.exemplar_count:  # Truncate to max n_exemplars per task
                n_fixed_exemplars_per_class = self.exemplar_count
                n_exemplars_per_class = np.repeat(n_fixed_exemplars_per_class, n_classes)
            else:  # Assign random ones as well
                n_exemplars_per_class = np.repeat(n_fixed_exemplars_per_class, n_classes)
                n_random_exemplars = args.n_exemplars_to_append_per_batch % n_classes

                rnd_cnt = 0
                while rnd_cnt < n_random_exemplars:
                    idx = random.randint(0, n_classes - 1)
                    if n_exemplars_per_class[idx] < self.exemplar_count:
                        n_exemplars_per_class[idx] += 1
                        rnd_cnt += 1

            # Iterate over all previously seen classes (over all tasks)
            for task in range(t):
                offset1, offset2 = self.compute_offsets(task)  # For all previous tasks: do knowledge distillation

                task_exemplarlist = []
                task_targetlist = []
                for local_class_idx in range(0, self.nc_per_task[task]):  # Iterate over seen classes in task
                    class_idx = local_class_idx + offset1

                    if n_exemplars_per_class[class_idx] > 0:
                        # first generate a minibatch with one example per class from previous tasks

                        # Pick rnd samples from prev_task exemplar set
                        exemplar_idxes = np.random.permutation(len(self.mem_class_x[class_idx]))[
                                         :n_exemplars_per_class[class_idx]].tolist()

                        # Gather samples
                        inp_dist_paths = []
                        target_dist = torch.zeros(n_exemplars_per_class[class_idx], self.nc_per_task[task])
                        for ex_cnt, ex_idx in enumerate(exemplar_idxes):
                            inp_dist_paths.append(self.mem_class_x[class_idx][ex_idx])  # Input exemplars
                            target_dist[ex_cnt] = self.mem_class_y[class_idx][ex_idx].clone()[
                                                  offset1: offset2]  # Outputs before update
                            assert sum(torch.eq(target_dist[ex_cnt], -10e10)) < 1

                        # Loading exemplar set
                        exemplarlist = inp_dist_paths
                        targetlist = self.mem_class_x.format_targetlist(target_dist, class_idx, ex_idx=None,
                                                                        is_target_distr=True)

                        task_exemplarlist.extend(exemplarlist)
                        task_targetlist.extend(targetlist)

                if len(task_exemplarlist) <= 0:
                    continue
                transform = args.task_imgfolders['train'].transform
                input_imgfolder = self.mem_class_x.get_imagefolder(task_exemplarlist, task_targetlist, transform)
                input_dataloader = self.mem_class_x.get_dataloader(input_imgfolder, batch_size=args.total_batch_size)

                # Iterate exemplars and calculate knowledge distillation
                for data in input_dataloader:
                    inputs, targets = data
                    inputs = Variable(inputs.cuda()) if self.gpu else Variable(inputs)
                    targets = Variable(targets.cuda()) if self.gpu else Variable(targets)

                    # Calculate loss
                    ex_out = self.forward_training(inputs, task)[:, offset1: offset2]
                    if self.gpu:  # Free up space on GPU, but don't remove original CPU mem data of model
                        del inputs

                    if self.gpu:
                        ex_out = ex_out.cuda()
                        targets = targets.cuda()

                    distr_log_ex_out = self.lsm(ex_out / T)
                    distr_target = self.sm(targets / T)
                    ex_loss = self.kl(distr_log_ex_out, distr_target) * (T ** 2)

                    # Numerical deviations when calculated on same distr
                    if ex_loss < 0:
                        if ex_loss < -1e06:
                            print("WARNING high negative loss value: ", ex_loss)
                        ex_loss = 0
                    total_ex_loss += ex_loss
                    total_ex_loss_count += 1
                total_ex_loss = total_ex_loss / total_ex_loss_count
                total_ex_loss = self.reg * total_ex_loss
                loss += total_ex_loss

        # bprop and update
        loss.backward()
        self.opt.step()

        return loss, correct_classified

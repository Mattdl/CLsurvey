# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import methods.rehearsal.model.common as common
from torch.autograd import Variable


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

        self.full_mem_mode = args.full_mem_mode
        print("Full_mem_mode = ", self.full_mem_mode)

        # Set raw network
        self.net = torch.load(args.prev_model_path)

        # Replace output layer
        last_layer_index = str(len(self.net.classifier._modules) - 1)
        num_ftrs = self.net.classifier._modules[last_layer_index].in_features
        self.net.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, n_outputs)
        self.n_outputs = n_outputs
        print("NEW FC CLASSIFIER HEAD with {} units".format(n_outputs))

        if args.cuda:
            self.net.cuda()

        self.ce = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
        if self.full_mem_mode:
            self.n_total_memories = args.n_memories * n_tasks  # TOTAL memory
            self.n_memories = self.n_total_memories  # Starts off with full memory capacity
            print("INIT: full capacity = {}".format(self.n_total_memories))
        else:
            self.n_memories = args.n_memories  # global: memories per task
        print("mem per task = {}".format(self.n_memories))
        self.gpu = args.cuda
        self.dropout_masks = {}

        # allocate episodic memory
        self.memory_data = None  # declare in forward for 1st sample, based on input size
        self.memory_labels = torch.LongTensor(n_tasks, self.n_memories)

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
        print("SET optimizer with current lr")

    def compute_offsets(self, task_idx, cum_nc_per_task):
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

    def observe_FT(self, x, t, y, paths, args):
        """
        Populates exemplar set of current task,
        And appends samples to batch of prev exemplar sets (rehearsal, with guaranteed samples from each exemplar set)

        Criterion ce avges per batch: so loss per sample
        This means that if summing the loss for exemplars, we should avg over all these batch means as well.
        Because the amount of batch means increases every new task.

        :param x: training batch of current task
        :param t: current task idx
        :param y: labels (idxes FC layer assumed task-specific head)
        :param n_exemplars_to_append: exemplars to append over all the prev exemplar sets.
        :return:
        """
        n_exemplars_to_append = args.n_exemplars_to_append_per_batch
        self.zero_grad()
        self.reset_dropout_config()

        #########################################
        # update memory
        if t != self.old_task:
            print("From old task {} to new task {}".format(self.old_task, t))
            self.observed_tasks.append(t)
            self.old_task = t
            self.mem_cnt = 0  # Reset counter!!

            if self.memory_data is None:
                input_dim = x.shape[1:]
                self.memory_data = common.RehearsalMemory(self.n_tasks, self.n_memories, input_dim)
                print("Mem data shape = ", self.memory_data.shape)

            # Redistribute memories
            if self.full_mem_mode:
                self.n_memories = int(self.n_total_memories / (len(self.observed_tasks)))  # Divide memory between tasks
                print("EACH TASK HAS NOW {} exemplars".format(self.n_memories))

                print("TRUNCATING PREV MEMORIES PER CLASS")
                for key, mem in self.memory_data.exemplars.items():
                    self.memory_data[key] = self.memory_data[key][:self.n_memories]

                self.memory_labels = self.memory_labels[:, :self.n_memories]

        #########################################
        # MEMORY UPDATE: Exemplar set new task
        # Update ring buffer storing examples from current task
        # Buffer will always keep the last batches (random exemplar keeping)
        bsz = y.data.size(0)  # batch size
        endcnt = min(self.mem_cnt + bsz,
                     self.n_memories)  # storing batch in mem: but can't exceed limit of n_mems per task
        effbsz = endcnt - self.mem_cnt  # effective batch size (what still fits in mem)
        self.memory_data[t, self.mem_cnt: endcnt] = paths[: effbsz]
        # Adapt labels as well
        if bsz == 1:
            self.memory_labels[t, self.mem_cnt] = y.data[0].cpu()
        else:
            self.memory_labels[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz].cpu())
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:  # circular buffer
            self.mem_cnt = 0
        assert self.mem_cnt <= self.n_memories
        assert effbsz >= 0

        #########################################
        # FORWARD EXEMPLARS PREV TASKS
        total_ex_loss = 0
        total_ex_loss_count = 0
        avg_sample_ex_loss = 0
        if t > 0 and n_exemplars_to_append > 0:
            n_fixed_exemplars_per_prev_task = int(np.floor(n_exemplars_to_append / t))
            n_random_exemplars = n_exemplars_to_append % t
            n_exemplars_per_prev_task = [n_fixed_exemplars_per_prev_task for task in range(0, t)]

            for rnd_ex_cnt in range(n_random_exemplars):
                n_exemplars_per_prev_task[random.randint(0, t - 1)] += 1

            # extend batch, adapt label
            for tt in range(t):
                past_task = self.observed_tasks[tt]
                if n_exemplars_per_prev_task[past_task] > 0:
                    offset1, offset2 = self.compute_offsets(past_task, self.cum_nc_per_task)

                    # Pick rnd samples from prev_task exemplar set
                    exemplar_idxes = []
                    while len(exemplar_idxes) < n_exemplars_per_prev_task[past_task]:
                        idx = random.randint(0, self.n_memories - 1)
                        if idx not in exemplar_idxes:
                            exemplar_idxes.append(idx)

                    # Loading exemplar set
                    transform = args.task_imgfolders['train'].transform
                    input_imgfolder = self.memory_data.convert_imagefolder(transform, self.memory_labels,
                                                                           entry=past_task, ex_idx=exemplar_idxes)
                    input_dataloader = self.memory_data.get_dataloader(input_imgfolder,
                                                                       batch_size=args.batch_size)
                    for data in input_dataloader:
                        inputs, targets = data
                        inputs = Variable(inputs.cuda()) if self.gpu else Variable(inputs)

                        output = self.forward(inputs, past_task)[:, offset1: offset2]
                        if self.gpu:
                            del inputs
                        if self.gpu:
                            output = output.cuda()
                            targets = targets.cuda()

                        total_ex_loss += self.ce(output, targets)
                        total_ex_loss_count += 1

                        if self.gpu:
                            del output, targets
            avg_sample_ex_loss = total_ex_loss / total_ex_loss_count

        #########################################
        # FORWARD EXEMPLARS CURRENT TASK
        offset1, offset2 = self.compute_offsets(t, self.cum_nc_per_task)
        output = self.forward(x, t)[:, offset1: offset2]
        _, preds = torch.max(output.data, 1)
        correct_classified = torch.sum(preds == y.data)  # In dedicated head (offsetted)
        new_task_loss = self.ce(output, y)

        #########################################
        # UPDATE
        loss = new_task_loss + avg_sample_ex_loss

        loss.backward()
        self.opt.step()

        return loss, correct_classified

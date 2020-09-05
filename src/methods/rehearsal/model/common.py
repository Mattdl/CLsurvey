# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from data.imgfolder import ImagePathlist


class RehearsalMemory(object):

    def __init__(self, n_entries, n_memories, input_dim):
        """
        Entries could be tasks/classes -> amount of key entries in the dictionary.
        :param n_entries: 
        :param n_memories: memories per entry
        :param input_dim: 
        """
        self.n_entries = n_entries  # Classes for iCarl, Tasks for GEM
        self.n_memories = n_memories
        self.input_dim = input_dim
        self.exemplars = {entry: [None] * n_memories for entry in range(n_entries)}
        self.shape = [self.n_entries, self.n_memories, *self.input_dim]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            entry, ex_idx = item

            if isinstance(ex_idx, list):
                return [self.exemplars[entry][x] for x in ex_idx]

            return self.exemplars[entry][ex_idx]
        elif isinstance(item, int):
            entry = item
            return self.exemplars[entry]
        else:
            raise IndexError("getitem with index :{} NOT VALID".format(item))

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            entry, ex_idx = key
            self.exemplars[entry][ex_idx] = value
        elif isinstance(key, int):
            self.exemplars[key] = value

        self.n_entries = len(self.exemplars)

    def __len__(self):
        self.n_entries = len(self.exemplars)
        assert self.n_entries == len(self.exemplars)
        return self.n_entries

    def convert_imagefolder(self, transform, targetlist=None, entry=None, ex_idx=None, exemplarlist=None,
                            is_target_distr=False):
        """
        Make subset of imagepaths where to sample from in dataloader.
        """

        if exemplarlist is None:
            exemplarlist = self.get_exemplarlist(entry, ex_idx)

        if targetlist is not None:
            targetlist = self.format_targetlist(targetlist, entry, ex_idx, is_target_distr)

        return self.get_imagefolder(exemplarlist, targetlist, transform=transform)

    def get_imagefolder(self, exemplarlist, targetlist, transform):
        return ImagePathlist(exemplarlist, targetlist, transform=transform)

    def get_exemplarlist(self, entry, ex_idx):
        if entry is None:
            entry = slice(0, self.n_entries)  # Get all
        if ex_idx is None:
            ex_idx = slice(0, self.n_memories)  # Get all
        exemplarlist = self.__getitem__((entry, ex_idx))
        return exemplarlist

    def format_targetlist(self, targetlist, entry, ex_idx=None, is_target_distr=False):
        if not is_target_distr:
            if ex_idx is not None:
                targetlist = targetlist[entry][ex_idx]
            else:
                targetlist = targetlist[entry]
        else:
            targetlist = targetlist.clone()

        return targetlist

    def get_dataloader(self, imgfolder, batch_size=None):
        if batch_size is None:
            batch_size = self.n_memories
        return torch.utils.data.DataLoader(imgfolder, batch_size=batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)

    def get_exemplar_lengths(self):
        return {task: len(paths) for task, paths in self.exemplars.items()}

    def __str__(self):
        return "SHAPE={}, EXEMPLARS={}".format(self.shape, self.get_exemplar_lengths())


def compute_offsets(task_idx, cum_nc_per_task):
    """
        Compute offsets (for cifar) to determine which
        outputs to select for a given task.

        Output idxs: [offset1, offset2[
    """
    if task_idx == 0:
        offset1 = 0
    else:
        offset1 = int(cum_nc_per_task[task_idx - 1])
    offset2 = int(cum_nc_per_task[task_idx])
    return offset1, offset2


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

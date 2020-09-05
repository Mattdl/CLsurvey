import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import copy
import os
import pdb
import shutil
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import utilities.utils as utils


class Elastic_SGD(optim.SGD):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(Elastic_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(Elastic_SGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # print('************************DOING A STEP************************')
        # loss=super(Elastic_SGD, self).step(closure)
        loss = None
        if closure is not None:
            loss = closure()
        index = 0
        reg_lambda = reg_params.get('lambda')
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                # print('************************ONE PARAM************************')

                if p.grad is None:
                    continue

                d_p = p.grad.data
                unreg_dp = p.grad.data.clone()
                # HERE MY CODE GOES
                reg_param = reg_params.get(p)

                omega = reg_param.get('omega')
                zero = torch.FloatTensor(p.data.size()).zero_()
                init_val = reg_param.get('init_val')
                w = reg_param.get('w')
                curr_wegiht_val = p.data.clone()
                # move the variables to cuda
                init_val = init_val.cuda()
                w = w.cuda()
                omega = omega.cuda()
                # get the difference
                weight_dif = curr_wegiht_val.add(-1, init_val)

                regulizer = torch.mul(weight_dif, 2 * reg_lambda * omega)
                # JUST NOW PUT BACK
                d_p.add_(regulizer)

                del weight_dif
                del omega

                del regulizer
                # HERE MY CODE ENDS

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    # pdb.set_trace()
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                #
                p.data.add_(-group['lr'], d_p)
                w_diff = p.data.add(-1, curr_wegiht_val)

                del curr_wegiht_val

                change = w_diff.mul(unreg_dp)
                del unreg_dp
                change = torch.mul(change, -1)
                del w_diff
                if 0:
                    if change.equal(zero.cuda()):
                        print('change zero')
                        pdb.set_trace()
                    if w.equal(zero.cuda()):
                        print('w zero')

                    if w.equal(zero.cuda()):
                        print('w after zero')
                    x = p.data.add(-init_val)
                    if x.equal(zero.cuda()):
                        print('path diff is zero')
                del zero
                del init_val
                w.add_(change)
                reg_param['w'] = w
                # after deadline

                reg_params[p] = reg_param
                index += 1
        return loss


def set_lr(optimizer, lr, count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count >= 10:
        continue_training = False
        print("training terminated")
    if count == 5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def traminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    utils.print_timing(time_elapsed, "TRAINING ONLY")
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model(model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir='./',
                resume='', saving_freq=5):
    print('dictoinary length' + str(len(dset_loaders)))
    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not imporved
    best_model = model
    best_acc = 0.0
    mem_snapshotted = False

    if os.path.isfile(resume):

        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print("val_beat_counts", val_beat_counts)
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    warning_NAN_counter = 0
    print("START EPOCH = ", str(start_epoch))
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since, best_acc)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # print('step')
                    optimizer.step(model.reg_params)

                # statistics
                if not math.isnan(loss.data.item()):
                    warning_NAN_counter += 1

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]

            if warning_NAN_counter > 0:
                print("SKIPPED NAN RUNNING LOSS FOR BATCH: ", warning_NAN_counter, " TIMES")
            print("EPOCH LOSS=", epoch_loss, ", RUNNING LOSS=", running_loss, ",DIVISION=", dset_sizes[phase])
            if epoch_loss > 1e4 or math.isnan(epoch_loss):
                print("TERMINATING: Epoch loss [", epoch_loss, "]  is NaN or > 1e4")
                return model, best_acc
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del loss
                    del preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1

        if epoch % saving_freq == 0:
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'arch': 'alexnet',
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc  # initialize importance dictionary


def initialize_reg_params(model):
    reg_params = {}
    for name, param in model.named_parameters():  # after deadline check
        w = torch.FloatTensor(param.size()).zero_()
        omega = torch.FloatTensor(param.size()).zero_()
        init_val = param.data.clone()
        reg_param = {}
        reg_param['omega'] = omega
        reg_param['w'] = w
        reg_param['init_val'] = init_val
        reg_param['name'] = name
        reg_params[param] = reg_param
    return reg_params


def update_reg_params(model, slak=1e-3):
    reg_params = model.reg_params
    index = 0
    for param in list(model.parameters()):
        print('index' + str(index))
        if param in reg_params.keys():
            print('updating index' + str(index))
            reg_param = reg_params.get(param)
            w = reg_param.get('w').cuda()
            zero = torch.FloatTensor(param.data.size()).zero_()

            if w.equal(zero.cuda()):
                print('W IS WRONG WARNING')
            omega = reg_param.get('omega')
            omega = omega.cuda()
            if not omega.equal(zero.cuda()):
                print('omega is not equal zero')
            else:
                print('omega is equal zero')

            omega = omega.cuda()
            init_val = reg_param.get('init_val')
            init_val = init_val.cuda()
            path_diff = param.data.add(-1, init_val)
            if path_diff.equal(zero.cuda()):
                print('PATH DIFF WRONG WARNING')
            dominator = path_diff.pow(2)
            dominator.add_(slak)
            this_omega = w.div(dominator)

            ####
            if 0:
                the_size = 1
                for x in this_omega.size():
                    the_size = the_size * x
                om = this_omega.view(the_size)
                randindex = torch.randperm(the_size)
                om = om[randindex.cuda()]
                this_omega = om.view(this_omega.size())

            this_omega = torch.max(this_omega, zero.cuda())
            print("**********max*************")
            print(this_omega.max())
            print("**********min*************")
            print(this_omega.min())
            omega.add_(this_omega)

            reg_param['omega'] = omega
            w = zero.cuda()
            reg_param['w'] = w
            reg_param['init_val'] = param.data.clone()
            reg_params[param] = reg_param
        else:
            print('initializing index' + str(index))
            w = torch.FloatTensor(param.size()).zero_()
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            reg_param['w'] = w
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
        index = index + 1
    return reg_params


def update_reg_params(model, slak=1e-3):
    reg_params = model.reg_params
    index = 0
    for param in list(model.parameters()):
        print('index' + str(index))
        if param in reg_params.keys():
            print('updating index' + str(index))
            reg_param = reg_params.get(param)
            w = reg_param.get('w').cuda()
            zero = torch.FloatTensor(param.data.size()).zero_()

            if w.equal(zero.cuda()):
                print('W IS WRONG WARNING')
            omega = reg_param.get('omega')
            omega = omega.cuda()
            if not omega.equal(zero.cuda()):
                print('omega is not equal zero')
            else:
                print('omega is equal zero')

            omega = omega.cuda()
            init_val = reg_param.get('init_val')
            init_val = init_val.cuda()
            path_diff = param.data.add(-1, init_val)
            if path_diff.equal(zero.cuda()):
                print('PATH DIFF WRONG WARNING')
            dominator = path_diff.pow(2)
            dominator.add_(slak)
            this_omega = w.div(dominator)

            ####
            if 0:
                the_size = 1
                for x in this_omega.size():
                    the_size = the_size * x
                om = this_omega.view(the_size)
                randindex = torch.randperm(the_size)
                om = om[randindex.cuda()]
                this_omega = om.view(this_omega.size())

            this_omega = torch.max(this_omega, zero.cuda())
            print("**********max*************")
            print(this_omega.max())
            print("**********min*************")
            print(this_omega.min())
            omega.add_(this_omega)

            reg_param['omega'] = omega
            w = zero.cuda()
            reg_param['w'] = w
            reg_param['init_val'] = param.data.clone()
            reg_params[param] = reg_param
        else:
            print('initializing index' + str(index))
            w = torch.FloatTensor(param.size()).zero_()
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            reg_param['w'] = w
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
        index = index + 1
    return reg_params


def update_reg_params_ref(model, slak=1e-3):
    reg_params = model.reg_params
    new_reg_params = {}
    index = 0
    for param in list(model.parameters()):
        print('index' + str(index))

        reg_param = reg_params.get(param)
        w = reg_param.get('w')
        zero = torch.FloatTensor(param.data.size()).zero_()
        if w.equal(zero.cuda()):
            print('wrong')
        omega = reg_param.get('omega')
        omega = omega.cuda()
        if not omega.equal(zero.cuda()):
            print('omega wrong')

        omega = omega.cuda()
        init_val = reg_param.get('init_val')
        init_val = init_val.cuda()
        path_diff = param.data.add(-init_val)
        if path_diff.equal(zero.cuda()):
            print('path_diff wrong')
        dominator = path_diff.pow_(2)
        dominator.add_(slak)
        this_omega = w.div(dominator)
        if this_omega.equal(zero.cuda()):
            print('this_omega wrong')
        omega.add_(this_omega)
        reg_param['omega'] = omega
        reg_param['w'] = w
        reg_param['init_val'] = param.data
        reg_params[param] = reg_param
        model.reg_params = reg_params
        new_reg_params[index] = reg_param
        index = index + 1
    return new_reg_params


def reassign_reg_params(model, new_reg_params):
    reg_params = model.reg_params
    reg_params = {}
    index = 0
    for param in list(model.parameters()):
        reg_param = new_reg_params[index]
        reg_params[param] = reg_param
        index = index + 1
    model.reg_params = reg_params
    return reg_params


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

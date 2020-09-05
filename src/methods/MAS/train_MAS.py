

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader

import utilities.utils as utils

class Weight_Regularized_SGD(optim.SGD):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, orth_reg=False, L1_decay=False):

        super(Weight_Regularized_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.orth_reg = orth_reg
        self.L1_decay = L1_decay

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):
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
                if p.grad is None:
                    continue
                d_p = p.grad.data

                ######################## CUSTOM UPDATE MAS: START ########################
                if p in reg_params:
                    reg_param = reg_params.get(p)
                    omega = reg_param.get('omega')
                    init_val = reg_param.get('init_val')
                    curr_wegiht_val = p.data
                    # move the variables to cuda
                    init_val = init_val.cuda()
                    omega = omega.cuda()

                    # get the difference
                    weight_dif = curr_wegiht_val.add(-1, init_val)

                    regulizer = weight_dif.mul(2 * reg_lambda * omega)

                    del weight_dif
                    del curr_wegiht_val
                    del init_val
                    d_p.add_(regulizer)
                    del regulizer
                    ######################## CUSTOM UPDATE MAS: END ########################

                if weight_decay != 0:
                    if self.L1_decay:
                        d_p.add_(weight_decay, p.data.sign())
                    else:
                        d_p.add_(weight_decay, p.data)
                # optionally you can use orthreg

                if self.orth_reg:
                    d_p.add_(orth_org_hook(p, {'beta': weight_decay}))
                if momentum != 0:
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

                p.data.add_(-group['lr'], d_p)
                index += 1
        return loss




def orth_org_hook(param, opt={}):
    if (len(param.size()) == 4):  # conv2d
        opt['epsilon'] = 1e-10
        opt['orth_lambda'] = 10
        if not 'beta' in opt.keys():
            opt['beta'] = 0.001

        ##################

        filters = param.data.clone().view(param.size(0), -1)
        norms = filters.norm(2, 1).squeeze()
        norms = norms.view(-1, 1).expand(filters.size())
        filters.div_(norms + opt['epsilon'])
        grad = torch.mm(filters, filters.transpose(1, 0))
        grad = torch.exp(grad * opt['orth_lambda'])
        grad = (grad * opt['orth_lambda']).div(grad + math.exp(opt['orth_lambda']))
        indeces = torch.LongTensor(range(grad.size(0))).cuda()
        grad[indeces, indeces] = 0
        grad = torch.mm(grad, filters)
        coef = opt['beta']

        grad = grad * coef
        grad = grad.view(param.size())
        return grad
    else:
        return torch.zeros(param.size()).cuda()  # ELASTIC SGD


class Objective_After_SGD(optim.SGD):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(Objective_After_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(Objective_After_SGD, self).__setstate__(state)

    def step(self, reg_params, batch_index, batch_size, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        index = 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:

                # print('************************ONE PARAM************************')

                if p.grad is None:
                    continue
                # param with zero learning rate will not be here
                if p in reg_params:
                    d_p = p.grad.data
                    unreg_dp = p.grad.data.clone()
                    # HERE MY CODE GOES
                    reg_param = reg_params.get(p)

                    zero = torch.FloatTensor(p.data.size()).zero_()
                    omega = reg_param.get('omega')
                    omega = omega.cuda()

                    # sum up the magnitude of the gradient
                    prev_size = batch_index * batch_size
                    curr_size = (batch_index + 1) * batch_size
                    omega = omega.mul(prev_size)

                    omega = omega.add(unreg_dp.abs_())
                    omega = omega.div(curr_size)
                    if omega.equal(zero.cuda()):
                        print('omega after zero')

                    reg_param['omega'] = omega
                    # pdb.set_trace()
                    reg_params[p] = reg_param
                index += 1
        return loss

def set_lr(optimizer, lr, count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count > 10:
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


# importance_dictionary: contains all the information needed for computing the w and omega

def train_model(model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir='./',
                resume='', saving_freq=5):
    print('dictoinary length' + str(len(dset_loaders)))
    # reg_params=model.reg_params
    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not imporved
    best_model = model
    best_acc = 0.0
    mem_snapshotted = False

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))
    # pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                # FOR MNIST DATASET
                inputs = inputs.squeeze()

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # print('step')
                    optimizer.step(model.reg_params)

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if epoch_loss > 1e4 or math.isnan(epoch_loss):
                print("NAN loss!")
                return model, best_acc
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del loss
                    del preds
                    best_acc = epoch_acc
                    # best_model = copy.deepcopy(model)
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1
        # epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        if epoch % saving_freq == 0:
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'alexnet',
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'model': model,
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc


def train_model_sparce(model, criterion, optimizer, lr_scheduler, lr, dset_loaders, dset_sizes, use_gpu, num_epochs,
                       exp_dir='./', resume='', lam=0):
    print('dictoinary length' + str(len(dset_loaders)))
    # reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))
    # pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':

                optimizer = lr_scheduler(optimizer, epoch, lr)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, norm = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = loss + lam * norm
                    loss.backward()
                    # print('step')
                    optimizer.step(model.reg_params)

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del labels
                del inputs
                del loss
                del preds
                best_acc = epoch_acc
                # best_model = copy.deepcopy(model)
                torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))

        # epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
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
    return model


# importance_dictionary: contains all the information needed for computing the w and omega

def compute_importance(model, optimizer, lr_scheduler, dset_loaders, use_gpu):
    print('dictoinary length' + str(len(dset_loaders)))
    # reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0

    # pdb.set_trace()

    epoch = 1
    optimizer = lr_scheduler(optimizer, epoch, 1)
    model.eval()  # Set model to training mode so we get the gradient

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    index = 0
    for dset_loader in dset_loaders:
        # pdb.set_trace()
        for data in dset_loader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), requires_grad=False), \
                                 Variable(labels.cuda(), requires_grad=False)
            else:
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            # later we could add L2Norm to the output
            Target_zeros = torch.zeros(outputs.size())
            Target_zeros = Target_zeros.cuda()
            Target_zeros = Variable(Target_zeros, requires_grad=False)

            loss = torch.nn.L1Loss(size_average=False)

            targets = loss(outputs, Target_zeros)

            targets.backward()

            print('batch number ', index)
            optimizer.step(model.reg_params, index, labels.size(0))
            index += 1

    return model


# importance_dictionary: contains all the information needed for computing the w and omega


def compute_importance_l2(model, optimizer, lr_scheduler, dset_loaders, use_gpu):
    print('dictoinary length' + str(len(dset_loaders)))
    # reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0

    epoch = 1
    optimizer = lr_scheduler(optimizer, epoch, 1)
    model.eval()  # Set model to training mode so we get the gradient

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    index = 0
    for dset_loader in dset_loaders:
        for data in dset_loader:
            # get the inputs
            inputs, labels = data
            if inputs.size(1) == 1 and len(inputs.size()) == 3:
                # for mnist, there is no channel
                # and  to avoid problems with the sparsity regulizers we remove that additional dimension generated by pytorch transformation
                inputs = inputs.view(inputs.size(0), inputs.size(2))
                # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), requires_grad=False), \
                                 Variable(labels.cuda(), requires_grad=False)
            else:
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            # loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            # later we could add L2Norm to the output

            # compute the L2 norm of output
            Target_zeros = torch.zeros(outputs.size())
            Target_zeros = Target_zeros.cuda()
            Target_zeros = Variable(Target_zeros, requires_grad=False)

            loss = torch.nn.MSELoss(size_average=False)

            targets = loss(outputs, Target_zeros)

            targets.backward()
            # print('step')

            optimizer.step(model.reg_params, index, labels.size(0))
            print('batch number ', index)
            index += 1

    return model


def compute_importance_l2_sparce(model, optimizer, lr_scheduler, dset_loaders, use_gpu):
    print('dictoinary length' + str(len(dset_loaders)))
    # reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0

    epoch = 1
    optimizer = lr_scheduler(optimizer, epoch, 1)
    model.eval()  # Set model to training mode so we get the gradient

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    index = 0
    for dset_loader in dset_loaders:
        for data in dset_loader:
            # get the inputs
            inputs, labels = data
            if inputs.size(1) == 1 and len(inputs.size()) == 3:
                # for mnist, there is no channel
                # and  to avoid problems with the sparsity regulizers we remove that additional dimension generated by pytorch transformation
                inputs = inputs.view(inputs.size(0), inputs.size(2))
            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), requires_grad=False), \
                                 Variable(labels.cuda(), requires_grad=False)
            else:
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs, x = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            # backward + optimize only if in training phase
            # later we could add L2Norm to the output

            # compute the L2 norm of output
            Target_zeros = torch.zeros(outputs.size())
            Target_zeros = Target_zeros.cuda()
            Target_zeros = Variable(Target_zeros, requires_grad=False)

            loss = torch.nn.MSELoss(size_average=False)

            targets = loss(outputs, Target_zeros)

            targets.backward()
            # print('step')
            optimizer.step(model.reg_params, index, labels.size(0))
            print('batch number ', index)
            index += 1

    return model



def compute_importance_gradient_vector(model, optimizer, lr_scheduler, dset_loaders, use_gpu):
    print('dictoinary length' + str(len(dset_loaders)))
    # reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0

    epoch = 1
    optimizer = lr_scheduler(optimizer, epoch, 1)
    model.eval()  # Set model to training mode so we get the gradient

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    index = 0
    for dset_loader in dset_loaders:
        for data in dset_loader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), requires_grad=False), \
                                 Variable(labels.cuda(), requires_grad=False)
            else:
                inputs, labels = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            # backward + optimize only if in training phase
            # later we could add L2Norm to the output

            # compute the L2 norm of output

            for output_i in range(0, outputs.size(1)):
                Target_zeros = torch.zeros(outputs.size())
                Target_zeros = Target_zeros.cuda()
                Target_zeros[:, output_i] = 1
                Target_zeros = Variable(Target_zeros, requires_grad=False)
                targets = torch.sum(outputs * Target_zeros)
                if output_i == (outputs.size(1) - 1):
                    targets.backward()
                else:
                    targets.backward(retain_graph=True)

                optimizer.step(model.reg_params, True, index, labels.size(0))
                optimizer.zero_grad()

            # print('step')
            optimizer.step(model.reg_params, False, index, labels.size(0))
            print('batch number ', index)
            index += 1

    return model


def initialize_reg_params(model, freeze_layers=[]):
    reg_params = {}
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            print('initializing param', name)
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            # initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
    return reg_params


# set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def initialize_store_reg_params(model, freeze_layers=[]):
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('storing previous omega', name)
                prev_omega = reg_param.get('omega')
                new_omega = torch.FloatTensor(param.size()).zero_()
                init_val = param.data.clone()
                reg_param['prev_omega'] = prev_omega
                reg_param['omega'] = new_omega

                # initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param] = reg_param

        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]
    return reg_params


# set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def initialize_store_aug_reg_params(model, freeze_layers=[]):
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('storing previous omega', name)
                prev_omega = reg_param.get('omega')
                new_omega = torch.FloatTensor(param.size()).zero_()
                init_val = param.data.clone()

                if 'neuron_omega_val' in reg_param.keys():
                    neuron_omega_val = reg_param['neuron_omega_val']
                    prev_omega = prev_omega - neuron_omega_val.expand_as(prev_omega)
                reg_param['prev_omega'] = prev_omega

                reg_param['omega'] = new_omega

                # initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param] = reg_param

        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]
    return reg_params


# set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def accumelate_reg_params(model, freeze_layers=[]):
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('restoring previous omega', name)
                prev_omega = reg_param.get('prev_omega')
                prev_omega = prev_omega.cuda()

                new_omega = (reg_param.get('omega')).cuda()
                acc_omega = torch.add(prev_omega, new_omega)

                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega

                reg_params[param] = reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]
    return reg_params


def subtract_reg_params(model, freeze_layers=[]):
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('subtracting new omega', name)
                prev_omega = reg_param.get('prev_omega')
                prev_omega = prev_omega.cuda()

                new_omega = (reg_param.get('omega')).cuda()
                acc_omega = prev_omega.add(-1, new_omega)
                zer = torch.FloatTensor(param.size()).zero_()
                acc_omega = torch.max(acc_omega, zer.cuda())
                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega

                reg_params[param] = reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]
    return reg_params


# set omega to zero but after storing its value in a temp omega in which later we can accumolate them both by averaging
def accumelate_avg_reg_params(model, freeze_layers=[], number_of_tasks=2):
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('storing previous omega', name)
                prev_omega = (reg_param.get('prev_omega')).cuda()
                prev_omega = prev_omega * (number_of_tasks - 1)  # running average
                new_omega = (reg_param.get('omega')).cuda()
                acc_omega = (torch.add(prev_omega, new_omega)).div(number_of_tasks)
                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega
                reg_params[param] = reg_param
                del acc_omega
                del new_omega
                del prev_omega

    return reg_params


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

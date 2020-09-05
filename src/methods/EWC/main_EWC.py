import time
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

import methods.EWC.train_EWC as EWC_SGD
import utilities.utils as utils
import data.imgfolder as ImageFolderTrainVal


def fine_tune_EWC_acuumelation(dataset_path, previous_task_model_path, exp_dir, data_dir, reg_sets, reg_lambda=1,
                               num_epochs=100, lr=0.0008, batch_size=200, weight_decay=0, head_shared=False,
                               saving_freq=5):
    """
    dataset_path:               current task dataset.
    previous_task_model_path:   model of the first task
    exp_dir:                    directory where the trained model will be exported.
    data_dir:                   data_dir is the directory where images of the previous task is. If none, all dataset will be loaded in case of mnist.
    reg_sets:                   List of paths of previous data splits to be used when computing fisher matrix
    reg_lambda:                 hyper parameter of EWC penalty
    num_epochs:                 number of training epochs.
    head_shared:                are tasks sharing the last layers?
    """

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()

    print("Loading prev model from path: ", previous_task_model_path)
    start_preprocess_time = time.time()
    model_ft = torch.load(previous_task_model_path)

    # update the omega for the previous task, accumelate it over previous omegas
    model_ft = accumulate_EWC_weights(data_dir, reg_sets, model_ft, batch_size=batch_size)
    # set the lambda for the EWC regularizer
    model_ft.reg_params['lambda'] = reg_lambda
    preprocessing_time = time.time() - start_preprocess_time
    utils.save_preprocessing_time(exp_dir, preprocessing_time)

    # get the number of features in this network and add a new task head
    if not head_shared:
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
        model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
        print("NEW FC CLASSIFIER HEAD with {} units".format(len(dset_classes)))

    criterion = nn.CrossEntropyLoss()
    # update the objective based params

    if use_gpu:
        model_ft = model_ft.cuda()

    # call the EWC optimizer
    optimizer_ft = EWC_SGD.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)

    if not os.path.exists(exp_dir):
        print("Going to exp_dir=", exp_dir)
        os.makedirs(exp_dir)

    # if there is a checkpoint to be resumed, in case where the training has stopped before on a given task
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    # train the model
    # this training functin passes the reg params to the optimizer to be used for penalizing changes on important params
    model_ft, acc = EWC_SGD.train_model(model_ft, criterion, optimizer_ft, lr, dset_loaders, dset_sizes, use_gpu,
                                        num_epochs, exp_dir, resume, saving_freq=saving_freq)

    return model_ft, acc


def accumulate_EWC_weights(data_dir, reg_sets, model_ft, batch_size):
    """
    accumulate fisher information matrix with the previously computed one,
    if this is the first task then initialize omega to zero and compute current fisher.
    """

    # ========================
    # define a transformation without augmentation, on the given dataset path.
    data_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dset_loaders = []
    for data_path in reg_sets:

        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
            dset = torch.load(data_path)
            dset = dset['train']

        dset_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8, pin_memory=True)
        dset_loaders.append(dset_loader)
    # =============================================================================

    use_gpu = torch.cuda.is_available()
    # in case of the first task, initialize reg_params to zero
    if not hasattr(model_ft, 'reg_params'):
        reg_params = initialize_reg_params(model_ft)
        model_ft.reg_params = reg_params
    # store previous omega values (Accumulated Fisher)
    reg_params = store_prev_reg_params(model_ft)
    model_ft.reg_params = reg_params
    # compute fisher
    data_len = len(dset)
    model_ft = diag_fisher(model_ft, dset_loader, data_len)
    # accumulate the current fisher with the previosly computed one
    reg_params = accumelate_reg_params(model_ft)
    model_ft.reg_params = reg_params
    # print current omega stat.
    sanitycheck(model_ft)
    return model_ft


def sanitycheck(model):
    for name, param in model.named_parameters():
        print(name)
        if param in model.reg_params:
            reg_param = model.reg_params.get(param)
            omega = reg_param.get('omega')

            print('omega max is', omega.max().item())
            print('omega min is', omega.min().item())
            print('omega mean is', omega.mean().item())  # here the ewc code goes


def diag_fisher(model, dset_loader, data_len):
    reg_params = model.reg_params
    model.eval()

    for data in dset_loader:
        model.zero_grad()
        x, label = data
        x, label = Variable(x).cuda(), Variable(label, requires_grad=False).cuda()

        output = model(x)
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), label, size_average=False)
        loss.backward()

        for n, p in model.named_parameters():
            if p in reg_params:
                reg_param = reg_params.get(p)
                omega = reg_param['omega'].cuda()
                omega += p.grad.data ** 2 / data_len  # Each datasample only contributes 1/datalength to the total
                reg_param['omega'] = omega
    return model


def initialize_reg_params(model, freeze_layers=None):
    freeze_layers = [] if freeze_layers is None else freeze_layers
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
def store_prev_reg_params(model, freeze_layers=None):
    freeze_layers = [] if freeze_layers is None else freeze_layers
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
def accumelate_reg_params(model, freeze_layers=None):
    freeze_layers = [] if freeze_layers is None else freeze_layers
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

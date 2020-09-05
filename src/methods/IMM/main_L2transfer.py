import os
import os.path

import torch
import torch.nn as nn
from torchvision import models

from methods.IMM import train_L2transfer


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def update_reg_params(model, freeze_layers=None):
    """
    Constructs dictionary reg_param = {}, with:
        reg_param['omega'] = omega  # Importance weights (here all set to 1, as they are not used in mean-IMM)
        reg_param['init_val'] = init_val # The original weight values of the prev task model
    """
    print("UPDATING REG PARAMS")
    reg_params = model.reg_params
    freeze_layers = [] if freeze_layers is None else freeze_layers

    for index, (name, param) in enumerate(model.named_parameters()):
        print('Named params: index' + str(index))

        # If the param is a reg_param within the model
        if param in reg_params.keys():  # Restoring SI omega/init_val hyperparams
            if name not in freeze_layers:
                print('updating index' + str(index))

                omega = torch.ones(param.size())
                init_val = param.data.clone()

                reg_param = {}
                reg_param['omega'] = omega  # Importance weights?
                reg_param['init_val'] = init_val  # The original weights of prev network

                # Update model
                reg_params[param] = reg_param
            else:
                reg_param = reg_params.get(param)
                print('removing unused frozen omega', name)
                del reg_param['omega']
                del reg_params[param]
        else:
            print('initializing index' + str(index))
            omega = torch.ones(param.size())
            init_val = param.data.clone()

            reg_param = {}
            reg_param['omega'] = omega
            reg_param['init_val'] = init_val

            reg_params[param] = reg_param  # Update model

    return reg_params


def fine_tune_l2transfer(dataset_path, model_path, exp_dir, batch_size=100, num_epochs=100, lr=0.0004, reg_lambda=100,
                         init_freeze=0, weight_decay=0, saving_freq=5):
    """
    IMM pipeline, only using L2-transfer technique and weight transfer.

    reg_params is dictionary which looks like:
    - param tensors
    - param weights/backup: tensor(){omega=[one-vectors], init_val=[weights prev task net]}
    - lambda = the regularization hyperparameter used

    :param reg_lambda:  reg hyperparam for the L2-transfer
    """
    print('lr is ' + str(lr))

    ########################################
    # DATASETS
    ########################################
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    ########################################
    # LOAD INIT MODEL
    ########################################
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        print("=> RESUMING FROM CHECKPOINTED MODEL: ", resume)
    else:
        if not os.path.isfile(model_path):
            model_ft = models.alexnet(pretrained=True)
            print("=> STARTING PRETRAINED ALEXNET")

        else:
            model_ft = torch.load(model_path)
            print("=> STARTING FROM OTHER MODEL: ", model_path)

    # Replace last layer classifier, for the amount of classes in the current dataset
    if not init_freeze:
        # Alexnet vs VGG
        last_layer_index = len(model_ft.classifier) - 1
        num_ftrs = model_ft.classifier[last_layer_index].in_features
        model_ft.classifier._modules[str(last_layer_index)] = nn.Linear(num_ftrs, len(dset_classes))

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # If not resuming from an preempted IMM model, cleanup model
    if not os.path.isfile(resume):

        # Prev task model, last 2 hyperparam
        parameters = list(model_ft.parameters())
        parameter1 = parameters[-1]
        parameter2 = parameters[-2]

        # Try to remove them from the reg_params
        try:
            model_ft.reg_params.pop(parameter1, None)
            model_ft.reg_params.pop(parameter2, None)
        except:
            print('nothing to remove')

        # The regularization params are the parameters of the prev model (trying to)
        reg_params = update_reg_params(model_ft)
        print('update')
        reg_params['lambda'] = reg_lambda  # The regularization hyperparam
        model_ft.reg_params = reg_params

    # Only transfer here to CUDA, preventing non-cuda network adaptations
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft = model_ft.cuda()

    ########################################
    # TRAIN
    ########################################
    # Define Optimizer for IMM: extra loss term in step
    optimizer_ft = train_L2transfer.Weight_Regularized_SGD(model_ft.parameters(), lr,
                                                           momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()  # Loss
    model_ft, acc = train_L2transfer.train_model(model_ft, criterion, optimizer_ft, lr, dset_loaders, dset_sizes,
                                                 use_gpu, num_epochs, exp_dir, resume,
                                                 saving_freq=saving_freq)
    return model_ft, acc


def models_mean_moment_matching(models, task):
    for name, param in models[task].named_parameters():
        print(name)
        mean_param = torch.zeros(param.data.size()).cuda()
        for i in range(0, len(models)):
            if models[i].state_dict()[name].size() != mean_param.size():
                return models[task]
            mean_param = mean_param + models[i].state_dict()[name]
        mean_param = mean_param / len(models)
        param.data = mean_param.clone()

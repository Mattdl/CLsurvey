import warnings
import time
import os

import torch
import torch.nn as nn
from torchvision import models

from methods.SI import train_SI

import utilities.utils as utils

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def fine_tune_elastic(dataset_path, model_path, exp_dir, batch_size=200, num_epochs=100, lr=0.0004, reg_lambda=100,
                      init_freeze=0, weight_decay=0, saving_freq=5):
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume = os.path.join(exp_dir, 'epoch.pth.tar')
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
    else:
        if not os.path.isfile(model_path):
            print('***********************Starting from ALEXNET*****************************')

            warnings.warn('The model path is empty we are starting from alexnet')
            model_ft = models.alexnet(pretrained=True)

        else:
            model_ft = torch.load(model_path)
        if not init_freeze:
            last_layer_index = str(len(model_ft.classifier._modules) - 1)
            num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    if use_gpu:
        print("Using GPU")
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    start_preprocess_time = time.time()
    if not os.path.isfile(resume):

        if not hasattr(model_ft, 'reg_params'):
            reg_params = train_SI.initialize_reg_params(model_ft)

            print('initialize')
        else:
            parameters = list(model_ft.parameters())
            parameter1 = parameters[-1]
            parameter2 = parameters[-2]
            model_ft.reg_params.pop(parameter1, None)
            model_ft.reg_params.pop(parameter2, None)
            # pdb.set_trace()
            reg_params = train_SI.update_reg_params(model_ft)
            print('update')
        reg_params['lambda'] = reg_lambda
        model_ft.reg_params = reg_params
    preprocessing_time = time.time() - start_preprocess_time
    utils.save_preprocessing_time(exp_dir, preprocessing_time)

    optimizer_ft = train_SI.Elastic_SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
    # if  hasattr(model_ft, 'reg_params'):
    #    reg_params=Elastic_Training.reassign_reg_params(model_ft,new_reg_params)

    model_ft, best_acc = train_SI.train_model(model_ft, criterion, optimizer_ft, lr, dset_loaders, dset_sizes,
                                              use_gpu, num_epochs, exp_dir, resume,
                                              saving_freq=saving_freq)

    return model_ft, best_acc

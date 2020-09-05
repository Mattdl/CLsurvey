import time
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from methods.MAS import train_MAS
import utilities.utils as utils
import data.imgfolder as ImageFolderTrainVal


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch > 0 and epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


def replace_heads(previous_model_path, current_model_path):
    current_model_ft = torch.load(current_model_path)

    previous_model_ft = torch.load(previous_model_path)
    current_model_ft.classifier._modules['6'] = previous_model_ft.classifier._modules['6']
    return current_model_ft


def fine_tune_objective_based_acuumelation(dataset_path, previous_task_model_path, init_model_path, exp_dir, data_dir,
                                           reg_sets, reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200,
                                           weight_decay=0, b1=True, L1_decay=False, head_shared=False,
                                           saving_freq=5):
    """
    In case of accumelating omega for the different tasks in the sequence, baisically to mimic the setup of other method where 
    the reguilizer is computed on the training set. Note that this doesn't consider our adaptation

    """
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()

    start_preprocess_time = time.time()
    model_ft = torch.load(previous_task_model_path)
    if isinstance(model_ft, dict):
        model_ft = model_ft['model']
    if b1:
        # compute the importance with batch size of 1, to mimic the online setting
        update_batch_size = 1
    else:
        update_batch_size = batch_size
    # update the omega for the previous task, accumelate it over previous omegas
    model_ft = accumulate_objective_based_weights(data_dir, reg_sets, model_ft, update_batch_size, norm,
                                                  test_set="train")
    # set the lambda for the MAS regularizer
    model_ft.reg_params['lambda'] = reg_lambda
    preprocessing_time = time.time() - start_preprocess_time
    utils.save_preprocessing_time(exp_dir, preprocessing_time)

    # get the number of features in this network and add a new task head

    ##############
    if not head_shared:
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        if not init_model_path is None:
            init_model = torch.load(init_model_path)
            model_ft.classifier._modules[last_layer_index] = init_model.classifier._modules[last_layer_index]

        else:
            num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))

            # ************************************************
    criterion = nn.CrossEntropyLoss()
    # update the objective based params

    if use_gpu:
        model_ft = model_ft.cuda()

    # call the MAS optimizer
    optimizer_ft = train_MAS.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,
                                                    weight_decay=weight_decay, L1_decay=L1_decay)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if init_model_path is not None:
        del init_model
    # if there is a checkpoint to be resumed, in case where the training has stopped before on a given task
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    # train the model
    # this training functin passes the reg params to the optimizer to be used for penalizing changes on important params
    model_ft, acc = train_MAS.train_model(model_ft, criterion, optimizer_ft, lr, dset_loaders,
                                          dset_sizes, use_gpu, num_epochs, exp_dir, resume,
                                          saving_freq=saving_freq)

    return model_ft, acc


def accumulate_objective_based_weights(data_dir, reg_sets, model_ft, batch_size, norm='L2', test_set="train"):
    data_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dset_loaders = []
    for data_path in reg_sets:

        # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset = torch.load(data_path)
            dset = dset[test_set]

        dset_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8, pin_memory=True)
        dset_loaders.append(dset_loader)
    # =============================================================================

    use_gpu = torch.cuda.is_available()
    # hack
    if not hasattr(model_ft, 'reg_params'):
        reg_params = train_MAS.initialize_reg_params(model_ft)
        model_ft.reg_params = reg_params

    reg_params = train_MAS.initialize_store_reg_params(model_ft)
    model_ft.reg_params = reg_params

    optimizer_ft = train_MAS.Objective_After_SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

    if norm == 'L2':
        print('********************objective with L2 norm***************')
        model_ft = train_MAS.compute_importance_l2(model_ft, optimizer_ft, exp_lr_scheduler,
                                                   dset_loaders, use_gpu)
    else:
        model_ft = train_MAS.compute_importance(model_ft, optimizer_ft, exp_lr_scheduler, dset_loaders,
                                                use_gpu)

    reg_params = train_MAS.accumelate_reg_params(model_ft)
    model_ft.reg_params = reg_params
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
            print('omega mean is', omega.mean().item())


# if omega was already computed based on another trial
def move_omega(model1, model2):
    for name1, param1 in model1.named_parameters():
        print(name1)
        if param1 in model1.reg_params:
            for name2, param2 in model2.named_parameters():
                if name1 == name2 and param1.data.size() == param2.data.size():
                    reg_param1 = model1.reg_params.get(param1)
                    reg_param2 = model2.reg_params.get(param2)
                    omega1 = reg_param1.get('omega')
                    reg_param2['omega'] = omega1.clone()

    return model2

import time

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from methods.EBLL.AlexNet_EBLL import *
from data.imgfolder import *
import utilities.utils as utils


def add_task_autoencoder_for_training(current_model):
    new_model = torch.nn.Module()
    new_model.add_module('features', current_model.features)
    new_model.add_module('autoecnoder', AutoEncoder(256 * 6 * 6), 100)
    new_model.add_module('classifier', current_model.classifier)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def Fdistillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """

    loss = F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * scale
    return loss


def Rdistillation_loss(y, teacher_scores, T, scale):
    p_y = F.softmax(y)
    p_y = p_y.pow(1 / T)
    sumpy = p_y.sum(1)
    sumpy = sumpy.view(sumpy.size(0), 1)
    p_y = p_y.div(sumpy.repeat(1, scale))
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))
    loss = -p_teacher_scores * torch.log(p_y)
    loss = loss.sum(1)

    loss = loss.sum(0) / loss.size(0)
    return loss


def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """

    maxy, xx = y.max(1)
    maxy = maxy.view(y.size(0), 1)
    norm_y = y - maxy.repeat(1, scale)
    ysafe = norm_y / T
    exsafe = torch.exp(ysafe)
    sumex = exsafe.sum(1)
    ######Tscores
    maxT, xx = teacher_scores.max(1)
    maxT = maxT.view(maxT.size(0), 1)
    teacher_scores = teacher_scores - maxT.repeat(1, scale)
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))
    loss = torch.sum(torch.log(sumex) - torch.sum(p_teacher_scores * ysafe, 1))

    loss = loss / teacher_scores.size(0)
    return loss


def train_autoencoder(model, optimizer, task_criterion, encoder_criterion, lr_scheduler, lr, dset_loaders, dset_sizes,
                      use_gpu, num_epochs, exp_dir='./', resume='', alpha=1e-6):
    best_acc = 0
    count = 0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        count = checkpoint['count']
        best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    if use_gpu:
        model = model.cuda()
    print(str(start_epoch))

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_encoder_loss = 0.0
            running_task_loss = 0.0
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
                outputs, encoder_input, encoder_output = model(inputs)
                encoder_input = Variable(encoder_input)
                _, preds = torch.max(outputs.data, 1)
                task_loss = task_criterion(outputs, labels)

                encoder_loss = encoder_criterion(encoder_output, encoder_input)
                # Compute distillation loss.
                total_loss = alpha * encoder_loss + task_loss

                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                # statistics
                running_task_loss += task_loss.data.item()
                running_encoder_loss += encoder_loss.data.item()
                running_loss += total_loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            encoder_loss = running_encoder_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('{} Encoder LOSS: {:.4f} Acc: {:.4f}'.format(
                phase, encoder_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    count = 0
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                else:
                    if count == 5:
                        print('Best val Acc: {:4f}'.format(best_acc))
                        return model, best_acc
                    count += 1

        epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'encoder_loss': encoder_loss,
            'count': count,
            'best_acc': best_acc,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch_file_name)
        print()

    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc


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


def termination_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    utils.print_timing(time_elapsed, "TRAINING ONLY")
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model_ebll(model, original_model, criterion, code_criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu,
                     num_epochs, exp_dir='./', resume='', temperature=2, reg_alpha=1e-6, saving_freq=5,
                     reg_lambda=1):
    print('dictoinary length' + str(len(dset_loaders)))
    # set orginal model to eval mode
    original_model.eval()

    since = time.time()
    preprocessing_time = 0
    val_beat_counts = 0  # number of time val accuracy not imporved
    mem_snapshotted = False
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])

        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))
    for name, param in model.named_parameters():
        for namex, paramx in original_model.named_parameters():
            if namex == name:
                if param.data.equal(paramx):
                    print('param ', name, ' didnt change ')

    if use_gpu:
        model = model.cuda()
        original_model = original_model.cuda()

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    termination_protocol(since, best_acc)
                    utils.save_preprocessing_time(exp_dir, preprocessing_time)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_code_loss = 0.0
            # Iterate over data.
            for data in dset_loaders[phase]:
                start_preprocess_time = time.time()
                # get the inputs
                inputs, labels = data
                if phase == 'train':
                    original_inputs = inputs.clone()

                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        original_inputs = original_inputs.cuda()
                        original_inputs = Variable(original_inputs, requires_grad=False)
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    if phase == 'train':
                        original_inputs = Variable(original_inputs, requires_grad=False)
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                original_model.zero_grad()
                # forward
                # tasks_outputs and target_logits are lists of outputs for each task in the previous model and current model
                orginal_logits, orginal_codes = original_model(original_inputs)
                # Move to same GPU as current model.
                target_logits = [Variable(item.data, requires_grad=False)
                                 for item in orginal_logits]

                target_codes = [Variable(item.data, requires_grad=False)
                                for item in orginal_codes]
                del orginal_logits
                scale = [item.size(-1) for item in target_logits]
                tasks_outputs, tassk_codes = model(inputs)
                _, preds = torch.max(tasks_outputs[-1].data, 1)
                task_loss = criterion(tasks_outputs[-1], labels)

                # Compute distillation loss.
                dist_loss = 0.0
                code_loss = 0.0
                # Apply distillation loss to all old tasks.

                if phase == 'train':
                    for idx in range(len(target_logits)):
                        dist_loss += distillation_loss(tasks_outputs[idx], target_logits[idx], temperature, scale[idx])
                    # backward + optimize only if in training phase
                    for idx in range(len(target_codes)):
                        code_loss += code_criterion(tassk_codes[idx], target_codes[idx])

                total_loss = reg_lambda * dist_loss + task_loss + reg_alpha * code_loss
                preprocessing_time += time.time() - start_preprocess_time
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                # statistics
                running_loss += task_loss.data.item()
                if phase == 'train':
                    running_code_loss += code_loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            epoch_code_loss = running_code_loss / dset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                print('TASK CODE LOSS: {:.4f}'.format(epoch_code_loss))
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del tasks_outputs
                    del labels
                    del inputs
                    del task_loss
                    del preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1

        if epoch % saving_freq == 0 or (phase == 'val' and epoch_acc > best_acc):
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'arch': 'alexnet',
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)
        print()

    termination_protocol(since, best_acc)
    utils.save_preprocessing_time(exp_dir, preprocessing_time)
    return model, best_acc


def fine_tune_Adam_Autoencoder(dataset_path, previous_task_model_path, exp_dir='', batch_size=200, num_epochs=100,
                               lr=0.01, pretrained=True, alpha=1e-6, auto_dim=100, last_layer_name=6):
    """
    Train an Autencoder based on Alexnet for previous task. The previous task model is needed,
    as the Autoencoder needs to be inserted between feature extractor and classifier. Because the output
    Loss of the model is also used in order to train the Autoencoder (classifier loss error backprop to autoencoder).

    :param previous_task_model_path:    Model where Autoencoder is inserted in between feat extr and classifier.
    :param last_layer_name: Needs to be hardcoded as param, starting idx for heads in classifier (heads are stacked)
    """
    print("*" * 50, " AUTOENCODER TRAINING", "*" * 50)
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        print("LOADED AUTOENCODER CHECKPOINT AT: ", resume)
    else:
        model_ft = torch.load(previous_task_model_path)

        # No dropout layers please
        first_FC_layer = utils.get_first_FC_layer(model_ft.classifier)
        num_ftrs = first_FC_layer.in_features
        print("Starting from model with feat input dim of classifier: ", num_ftrs)

        if hasattr(model_ft, 'reg_params'):
            model_ft.reg_params = None

        model_ft = AlexNet_ENCODER(model_ft, dim=auto_dim, last_layer_name=last_layer_name, num_ftrs=num_ftrs)

    if use_gpu:
        model_ft = model_ft.cuda()

    task_criterion = nn.CrossEntropyLoss()
    encoder_criterion = nn.MSELoss()
    optimizer_ft = optim.Adadelta(model_ft.autoencoder.parameters(), lr)

    model_ft, best_acc = train_autoencoder(model_ft, optimizer_ft, task_criterion, encoder_criterion, exp_lr_scheduler,
                                           lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir, resume,
                                           alpha=alpha)
    return model_ft, best_acc


def fine_tune_SGD_EBLL(dataset_path, previous_task_model_path, autoencoder_model_path, init_model_path='', exp_dir='',
                       batch_size=200, num_epochs=100, lr=0.0004, init_freeze=1, weight_decay=0,
                       reg_alpha=1e-6, saving_freq=5, reg_lambda=1):
    """ Train the neural network, given trained auto encoders of previous tasks. """
    print("*" * 50, " FULL MODEL TRAINING", "*" * 50)
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    print("Dataloaders initialized")

    use_gpu = torch.cuda.is_available()

    autoencoder_model = torch.load(autoencoder_model_path)
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    print("Starting from previous model: " + previous_task_model_path)
    model_ft = torch.load(previous_task_model_path)

    if not (type(model_ft) is AlexNet_EBLL):
        print("Initial model is no ModelWrapperEBLL, creating wrapper object")
        last_layer_index = (len(model_ft.classifier._modules) - 1)
        model_ft = AlexNet_EBLL(model_ft, autoencoder_model.autoencoder, last_layer_name=last_layer_index)
    else:
        # add the new autoencoder
        print("Adding autoencoder to the backbone net")
        model_ft.autoencoders.add_module(str(len(model_ft.autoencoders._modules.items())),
                                         autoencoder_model.autoencoder.encode)

    original_model = copy.deepcopy(model_ft)
    num_ftrs = model_ft.classifier[model_ft.last_layer_name].in_features

    # Attach new head for new task
    if not init_freeze:
        print("No freeze mode")
        model_ft.classifier.add_module(str(len(model_ft.classifier._modules)), nn.Linear(num_ftrs, len(dset_classes)))
    else:
        print("Freeze mode")
        init_model = torch.load(init_model_path)
        model_ft.classifier.add_module(str(len(model_ft.classifier._modules)),
                                       init_model.classifier[model_ft.last_layer_name])
        del init_model
        # do something else
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if use_gpu:
        model_ft = model_ft.cuda()
        original_model = original_model.cuda()

    model_ft.reg_params = {}
    model_ft.reg_params['lambda'] = reg_lambda
    model_ft.reg_params['reg_alpha'] = reg_alpha

    print("Config training")
    criterion = nn.CrossEntropyLoss()
    encoder_criterion = nn.MSELoss()
    params = list(model_ft.features.parameters()) + list(model_ft.classifier.parameters())
    optimizer_ft = optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    print("DONE Config training")

    model_ft = train_model_ebll(model_ft, original_model, criterion, encoder_criterion, optimizer_ft, lr, dset_loaders,
                                dset_sizes, use_gpu, num_epochs, exp_dir, resume, reg_alpha=reg_alpha,
                                saving_freq=saving_freq, reg_lambda=reg_lambda)

    return model_ft


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

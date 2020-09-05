import time
import os
import copy

import torch
from torch.autograd import Variable

import methods.rehearsal.main_rehearsal
import utilities.utils as utils

def set_lr(optimizer, lr, count, decay_threshold=5, early_stop_threshold=10):
    """
    Early stop or decay learning rate by a factor of 0.1 based on count.
    Dynamic decaying speed (count based).
    :param lr:              Current learning rate
    :param count:           Amount of times of not increasing accuracy.
    """
    continue_training = True

    # Early Stopping
    if count > early_stop_threshold:
        continue_training = False
        print("training terminated")

    # Decay
    if count == decay_threshold:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def termination_protocol(since, best_acc, best_model, exp_dir):
    """
    Final stats printing: time and best validation accuracy.
    :param since:
    :param best_acc:
    :return:
    """
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    utils.print_timing(time_elapsed, "TRAINING ONLY")


    torch.save(best_model, os.path.join(exp_dir, 'best_model.pth.tar'))
    print("-> SAVED best model")


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_model(model, args, dset_sizes, resume='', save_models_mode=False, saving_freq=10):
    """
    :param model: model object to start from
    :param criterion: loss function
    :param dset_loaders:    train and val dataset loaders
    :param dset_sizes:      train and val sizes
    :param exp_dir:         where to output trained model
    :param resume:          path to model to resume from, empty string otherwise
    :param stack_head_cutoff: LWF, EBLL wrapper models have stacked heads
    """

    optimizer = model.opt
    exp_dir = args.save_path
    lr = args.lr
    use_gpu = args.cuda
    num_epochs = args.n_epochs

    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not improved
    best_acc = 0.0
    best_model = None

    # Resuming from model if specified
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
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

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    termination_protocol(since, best_acc, best_model, exp_dir)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_batch_stats = None
            batch_stats = None

            # Iterate over data.
            for data in args.dset_loaders[phase]:
                # get the inputs
                inputs, labels, paths = data  # Labels are output layer indices!
                # inputs = inputs.squeeze()

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # print(inputs[0].element_size() * inputs[0].nelement())

                if phase == 'train':
                    if args.finetune:
                        loss, correct_classified = model.observe_FT(inputs, args.task_idx, labels, paths, args)
                    else:
                        loss, correct_classified, batch_stats = model.observe(inputs, args.task_idx, labels, paths,
                                                                              args)

                if torch.isnan(loss):
                    print("Canceling because Nan LOSS")
                    return model, best_acc

                if phase == 'val':
                    loss, correct_classified = methods.rehearsal.main_rehearsal.eval_batch(model, inputs,
                                                                                           labels, args)

                # running statistics
                running_loss += loss.data.item()
                running_corrects += correct_classified

                if batch_stats is not None:
                    if total_batch_stats is None:
                        total_batch_stats = batch_stats
                    else:
                        for key, value in batch_stats.items():
                            total_batch_stats[key].extend(value)

            # epoch statistics
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if total_batch_stats is not None:
                for key, value in total_batch_stats.items():
                    print("{} = {}".format(key, value))

            # new best model: deep copy the model or save to .pth
            if phase == 'val':
                if epoch_acc > best_acc:
                    del loss
                    best_acc = epoch_acc
                    if save_models_mode:
                        torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                    best_model = copy.deepcopy(model)
                    print("-> New best model")
                else:
                    val_beat_counts += 1

        # Epoch checkpoint
        if save_models_mode and epoch % saving_freq == 0:
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

    termination_protocol(since, best_acc, best_model, exp_dir)
    return model, best_acc

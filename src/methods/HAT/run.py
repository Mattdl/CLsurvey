""" This script is wrapped by our main framework script."""

import sys, argparse, time
import torch

import methods.HAT.HAT_utils as HATutils


def main(overwrite_args):
    """
    :param nc_per_task: array with the amount of classes for each task
    """
    tstart = time.time()

    # Arguments
    parser = argparse.ArgumentParser(description='xxx')
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='', type=str, required=False,
                        choices=['mnist2', 'pmnist', 'cifar', 'mixture'], help='(default=%(default)s)')
    parser.add_argument('--approach', default='', type=str, required=False,
                        choices=['pathnet', 'hat'], help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=200, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--save_freq', default=20, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=1e10, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')

    #####################################
    # ARGS PARSING
    #####################################
    args = parser.parse_known_args()[0]
    for key_arg, val_arg in overwrite_args.items():  # Overwrite with specified method args
        setattr(args, key_arg, val_arg)
    args.task_idx = args.task_count - 1
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    ########################################################################################################################
    if not torch.cuda.is_available():
        print('[CUDA unavailable]')
        sys.exit()

    # Args -- Approach
    if args.approach == 'pathnet':
        if args.finetune_mode:
            from methods.HAT.approaches import pathnet_finetune as approach
        else:
            from methods.HAT.approaches import pathnet as approach
    elif args.approach == 'hat':
        if args.finetune_mode:
            from methods.HAT.approaches import hat_finetune as approach
        else:
            from methods.HAT.approaches import hat as approach
    else:
        raise NotImplementedError("Method {} not implemented!".format(args.approach))

    # Args -- Network
    if "alex" in args.model_name:
        if args.approach == 'hat':
            from methods.HAT.networks import alexnet_hat as network
            print("ALEX-HAT net")
        elif args.approach == 'pathnet':
            from methods.HAT.networks import alexnet_pathnet as network
        else:
            raise NotImplementedError("Only HAT and PathNet implemented!, not: ", args.approach)
    elif "VGG" in args.model_name:
        if args.approach == 'hat':
            from methods.HAT.networks import vgg_hat as network
            print("VGG-HAT net")
        elif args.approach == 'pathnet':
            from methods.HAT.networks import vgg_pathnet as network
            print("VGG-PATHNET net")
    else:
        raise NotImplementedError("Tiny Imagenet still to be implemented!, not: ", args.model_name)

    print("RUNNING WITH ARGS: ", overwrite_args)
    ########################################################################################################################
    # DATASET
    #####################################
    # load data: We consider 1 task
    print("Loading dset: {}".format(args.dataset_path))
    dsets = torch.load(args.dataset_path)
    args.task_imgfolders = dsets
    args.dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        for x in ['train', 'val']}
    taskcla = [(t, nc) for t, nc in enumerate(args.nc_per_task)]
    inputsize = (3,) + args.dataset.input_size

    #####################################
    # LOADING MODEL
    #####################################
    print('Loading model...')
    if args.is_scratch_model:  # make model with embeddings
        assert args.task_idx == 0  # Has to start from scratch
        raw_model = torch.load(args.prev_model_path)
        net = network.Net(raw_model, inputsize, taskcla, args).cuda()  # Wrap model and integrate embeddings
    else:
        print("Loading prev model from path: {}".format(args.prev_model_path))  # load prev saved model
        net = torch.load(args.prev_model_path)
    HATutils.print_model_report(net)

    #####################################
    # Approach and Optimizer
    #####################################
    appr = approach.Appr(net, args.output, sbatch=args.batch_size, nepochs=args.nepochs, lr=args.lr, args=args,
                         lr_factor=2, lr_patience=30)
    print(appr.criterion)

    #####################################
    # Train
    #####################################
    best_val_model, best_val_acc = appr.train(args.task_idx, args.dset_loaders)
    print('-' * 100)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    return best_val_model, best_val_acc

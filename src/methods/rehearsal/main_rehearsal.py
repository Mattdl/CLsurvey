# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import argparse
import os
import numpy as np

import torch
import utilities.utils as utils
from data.imgfolder import ImageFolder_Subset_PathRetriever

import methods.rehearsal.train_rehearsal


def eval_batch(model, x_batch, y_batch, args):
    """
    Originally shared head, here we use offsets to simulate multi-head setting.
    Calculate # correctly classified of a batch
    :param model:
    :param x_batch:
    :param y_batch:
    :return:
    """
    model.eval()
    y_batch = y_batch.cpu()
    offset1, offset2 = model.compute_offsets(args.task_idx, model.cum_nc_per_task)  # No shared head
    output = model(x_batch, args.task_idx, args=args, train_mode=True).data.cpu()[:, offset1: offset2]
    _, pred = torch.max(output, 1, keepdim=False)
    correct_classified = torch.sum(pred == y_batch.data)
    target = y_batch  # reduced output (single head), so can keep labels
    loss = model.ce(output, target)
    return loss, correct_classified


def print_batch_stats(mem_mode, mem_per_task, n_tasks=10, batch_size=200, print_batch_constitution=False,
                      print_mem_constitution=False):
    assert mem_mode == 'partial_mem' or mem_mode == 'full_mem'
    dset_sizes = [8000] * (n_tasks + 1)  # TinyImgnet
    print("TINY IMAGENET SIZES: ", dset_sizes)

    prefix = ''
    postfix = ''
    for task_idx in range(0, n_tasks + 1):
        if mem_mode == 'partial_mem':  # Only trains on exemplar sets, no validation
            n_mem_samples = mem_per_task * task_idx
        elif mem_mode == 'full_mem':
            n_mem_samples = mem_per_task * n_tasks
        n_total_samples = float(dset_sizes[task_idx]) + n_mem_samples
        ratio = float(n_mem_samples) / n_total_samples
        n_exemplars_to_append_per_batch = int(np.ceil(batch_size * ratio))  # Ceil: at least 1 per task
        new_batch_size = batch_size - n_exemplars_to_append_per_batch

        if task_idx == n_tasks:
            prefix = '('
            postfix = ')'
        if print_batch_constitution:
            print("{}BATCH: {} new samples, {} exemplars{}"
                  .format(prefix, new_batch_size, n_exemplars_to_append_per_batch, postfix))
        if print_mem_constitution:
            print("{}mem length = {}, tr_dset_size={}, batch_ratio={}{}"
                  .format(prefix, float(n_mem_samples), float(dset_sizes[task_idx]), '%.3f' % ratio, postfix))


def main(overwrite_args, nc_per_task):
    """
    For quick implementation: args are overwritten by overwrite_args (if specified).
    Additional params are passed in the main function.

    Do this task and return acc (task contains multiple classes = class-incremental in this setup)
    :param overwrite_args:
    :param nc_per_task: array with the amount of classes for each task
    :return:
    """
    parser = argparse.ArgumentParser(description='Continuum learning')

    parser.add_argument('--task_name', type=str,
                        help='name of the task')
    parser.add_argument('--task_count', type=int,
                        help='count of the task, STARTING FROM 1')
    parser.add_argument('--prev_model_path', type=str,
                        help='path to prev model where to start from')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models during and at the end of training')
    parser.add_argument('--n_outputs', type=int, default=200,
                        help='total number of outputs for ALL tasks')
    parser.add_argument('--method', choices=['gem', 'baseline_rehearsal_full_mem',
                                             'icarl', 'baseline_rehearsal_partial_mem'], type=str, default='gem',
                        help='method to use for train')
    parser.add_argument('--postprocess', action="store_true",
                        help='Do datamanagement (e.g. update buffers) after task is learned')
    # implemented in separate step to only perform once on THE best model in pipeline
    parser.add_argument('--debug', action="store_true", help='Debug mode')
    # model parameters
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--n_inputs', type=int, default=-1,
                        help='number of hidden layers')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--is_scratch_model', action='store_true',
                        help='is this the first task you train for?')

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', action="store_true",
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=70,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', action="store_true", help='Use GPU?')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')

    # data parameters
    parser.add_argument('dataset_path', type=str,
                        help='path to imgfolders of datasets')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='number of hidden layers')

    #####################################
    # ARGS PARSING
    #####################################
    args = parser.parse_known_args()[0]

    # Overwrite with specified method args
    args.nc_per_task = nc_per_task  # Array with nr outputs per class
    for key_arg, val_arg in overwrite_args.items():
        setattr(args, key_arg, val_arg)

    # Index starting from 0 (for array)
    args.task_idx = args.task_count - 1
    args.n_exemplars_to_append_per_batch = 0

    if 'baseline_rehearsal' in args.method:
        args.finetune = True
        if args.method == 'baseline_rehearsal_full_mem':
            args.full_mem_mode = True
        elif args.method == 'baseline_rehearsal_partial_mem':
            args.full_mem_mode = False
        else:
            raise Exception("UNKNOWN BASELINE METHOD:", args.method)

    # Input checks
    assert args.n_outputs == sum(args.nc_per_task)
    assert args.n_tasks == len(nc_per_task)

    # Baselines from scratch, others from SI model
    if args.task_count == 1 and 'baseline' not in args.method:
        assert 'SI' in args.prev_model_path, "FIRST TASK NOT STARTING FROM SCRATCH, BUT FROM SI: " \
                                             "ONLY STORING WRAPPER WITH EXEMPLARS, path = {}" \
            .format(args.prev_model_path)
        assert args.postprocess, "FIRST TASK WE DO ONLY POSTPROCESSING"

    assert os.path.isfile(args.prev_model_path), "Must specify existing prev_model_path, got: " + args.prev_model_path

    print("RUNNING WITH ARGS: ", overwrite_args)
    #####################################
    # DATASET
    #####################################
    # load data: We consider 1 task, not class-incremental
    dsets = torch.load(args.dataset_path)
    args.task_imgfolders = dsets
    args.dset_loaders = {
        x: torch.utils.data.DataLoader(ImageFolder_Subset_PathRetriever(dsets[x]), batch_size=args.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True)
        for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

    # Assign random part of batch to exemplars
    if 'baseline' in args.method or args.method == 'icarl':
        if args.method == 'baseline_rehearsal_partial_mem':  # Only trains on exemplar sets, no validation
            # Based on lengths of tr datasets: random sampling, but guaranteed at each batch from all exemplar sets
            n_mem_samples = args.n_memories * args.task_idx
        elif args.method == 'baseline_rehearsal_full_mem' or args.method == 'icarl':
            n_mem_samples = args.n_memories * args.n_tasks

        n_total_samples = float(dset_sizes['train']) + n_mem_samples
        ratio = float(n_mem_samples) / n_total_samples
        if not args.debug:
            args.n_exemplars_to_append_per_batch = int(np.ceil(args.batch_size * ratio))  # Ceil: at least 1 per task
        else:
            args.n_exemplars_to_append_per_batch = int(args.batch_size / 2 + 17)
        args.total_batch_size = args.batch_size
        args.batch_size = args.batch_size - args.n_exemplars_to_append_per_batch

        print("BATCH CONSISTS OF: {} new samples, {} exemplars"
              .format(args.batch_size, args.n_exemplars_to_append_per_batch))
        print("mem length = {}, tr_dset_size={}, ratio={}"
              .format(float(n_mem_samples), float(dset_sizes['train']), ratio))

    #####################################
    # LOADING MODEL
    #####################################
    if args.is_scratch_model:  # make model with self.net
        assert args.task_idx == 0  # Has to start from 0
        Model = importlib.import_module('model.' + args.method)
        model = Model.Net(args.n_inputs, args.n_outputs, args.n_tasks, args)
    else:
        # load prev saved model
        print("Loading prev model from path: ", args.prev_model_path)
        model = torch.load(args.prev_model_path)
    print("MODEL LOADED")
    model.init_setup(args)

    # Checks
    assert model.n_tasks == args.n_tasks, "model tasks={}, args tasks={}".format(model.n_tasks, args.n_tasks)
    assert model.n_outputs == args.n_outputs
    print("MODEL SETUP FOR CURRENT TASK")

    if args.postprocess:
        #####################################
        # POSTPROCESS
        #####################################
        """
        iCARL each task, 
        GEM only first task model SI (during training no exemplars where gathered)
        """
        print("POSTPROCESSING")
        model.manage_memory(args.task_idx, args)
        # if args.debug:
        #     model.check_exemplars(args.task_idx)

        utils.create_dir(os.path.dirname(args.save_path), "Postprocessed model dir")
        torch.save(model, args.save_path)
        print("SAVED POSTPROCESSED MODEL TO: {}".format(args.save_path))
        return None, None
    else:
        #####################################
        # TRAIN
        #####################################
        # if there is a checkpoint to be resumed, in case where the training has stopped before on a given task
        resume = os.path.join(args.save_path, 'epoch.pth.tar')
        model, best_val_acc = methods.rehearsal.train_rehearsal.train_model(model, args, dset_sizes,
                                                                            resume=resume)
        print("FINISHED TRAINING")

        return model, best_val_acc

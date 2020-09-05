import shutil
import os
import time
import torch

import utilities.utils as utils


def lr_grid_single_task(args, manager, save_models_mode='keep_none'):
    """
    Finetunes from starting model, acquire best lr and acc. LR gridsearch, with #finetune_iterations per LR.
    Makes symbolic link to the overall best iteration, corresponding with the obtained best lr.
    """

    # Init
    manager.store_policy = StoragePolicy(save_models_mode)
    args.task_name = manager.dataset.get_taskname(args.task_counter)
    manager.ft_parent_exp_dir = os.path.join(manager.parent_exp_dir,
                                             'task_' + str(args.task_counter), 'FT_LR_GRIDSEARCH')
    utils.create_dir(manager.ft_parent_exp_dir)
    print("FINETUNE LR GRIDSEARCH: Task ", args.task_name)

    # Logfile
    logfile_parent_dir = os.path.join(manager.ft_parent_exp_dir, 'log')
    utils.create_dir(logfile_parent_dir)
    logfile = os.path.join(logfile_parent_dir, utils.get_now() + '_finetune_grid.log')
    utils.append_to_file(logfile, "FINETUNE GRIDSEARCH LOG: Processed LRs")

    # Load Checkpoint
    processed_lrs = {}
    grid_checkpoint_file = os.path.join(manager.ft_parent_exp_dir, 'grid_checkpoint.pth')
    if os.path.exists(grid_checkpoint_file):
        checkpoint = torch.load(grid_checkpoint_file)
        processed_lrs = checkpoint['processed_lrs']

        print("STARTING FROM CHECKPOINT: ", checkpoint)
        utils.append_to_file(logfile, "STARTING FROM CHECKPOINT")

    ########################################################
    # PRESTEPS
    args.presteps_elapsed_time = 0
    if hasattr(manager.method, 'grid_prestep'):
        manager.method.grid_prestep(args, manager)

    ########################################################
    # LR GRIDSEARCH
    best_acc = 0
    best_lr = None
    manager.best_exp_grid_node_dirname = None
    best_iteration_batch_dirs = []
    for lr in args.lrs:
        print("\n", "<" * 20, "LR ", lr, ">" * 20)
        accum_acc = 0
        best_iteration_dir = None
        best_iteration_acc = 0
        iteration_batch_dirs = []
        if lr not in processed_lrs:
            processed_lrs[lr] = {'acc': []}

        for finetune_iteration in range(0, args.finetune_iterations):
            print("\n", "-" * 20, "FT ITERATION ", finetune_iteration, "-" * 20)
            start_time = time.time()

            # Paths
            exp_grid_node_dirname = "lr=" + str(utils.float_to_scientific_str(lr))
            if args.finetune_iterations > 1:
                exp_grid_node_dirname += "_it" + str(finetune_iteration)
            manager.gridsearch_exp_dir = os.path.join(manager.ft_parent_exp_dir, exp_grid_node_dirname)
            iteration_batch_dirs.append(manager.gridsearch_exp_dir)

            if finetune_iteration < len(processed_lrs[lr]['acc']):
                acc = processed_lrs[lr]['acc'][finetune_iteration]
                utils.set_random(finetune_iteration)
                print("RESTORING FROM CHECKPOINT: ITERATION = ", finetune_iteration, "ACC = ", acc)
            else:
                # Set new seed for reproducability
                utils.set_random(finetune_iteration)

                # Only actually saved when in save_model mode
                utils.create_dir(manager.gridsearch_exp_dir)

                # TRAIN
                model, acc = manager.method.grid_train(args, manager, lr)

                # Append results
                processed_lrs[lr]['acc'].append(acc)
                msg = "LR = {}, FT Iteration {}/{}, Acc = {}".format(lr, finetune_iteration + 1,
                                                                     args.finetune_iterations, acc)
                print(msg)
                utils.append_to_file(logfile, msg)

            # New best
            if acc > best_iteration_acc:
                if args.finetune_iterations > 1:
                    msg = "=> NEW BEST FT ITERATION {}/{}: (Attempt '{}': Acc '{}' > best attempt Acc '{}')" \
                        .format(finetune_iteration + 1,
                                args.finetune_iterations,
                                finetune_iteration,
                                acc,
                                best_iteration_acc)
                    print(msg)
                    utils.append_to_file(logfile, msg)

                best_iteration_acc = acc
                best_iteration_dir = manager.gridsearch_exp_dir

            accum_acc = accum_acc + acc

            # update logfile/checkpoint
            torch.save({'processed_lrs': processed_lrs}, grid_checkpoint_file)

            # Save iteration hyperparams
            if hasattr(manager.method, "grid_chkpt") and manager.method.grid_chkpt:
                it_elapsed_time = time.time() - start_time
                hyperparams = {'val_acc': acc, 'lr': lr, 'iteration_elapsed_time': it_elapsed_time,
                               'args': vars(args), 'manager': vars(manager)}
                utils.print_timing(it_elapsed_time, 'TRAIN')
                manager.save_hyperparams(manager.gridsearch_exp_dir, hyperparams)
        avg_acc = accum_acc / args.finetune_iterations
        print("Done FT iterations\n")
        print("LR AVG ACC = ", avg_acc, ", BEST OF LRs ACC = ", best_acc)

        # New it-avg best
        if avg_acc > best_acc:
            best_lr = lr
            best_acc = avg_acc
            manager.best_exp_grid_node_dirname = best_iteration_dir  # Keep ref to best in all attempts
            print("UPDATE best lr = {}".format(best_lr))
            print("UPDATE best lr acc= {}".format(best_acc))

            utils.append_to_file(logfile, "UPDATE best lr = {}".format(best_lr))
            utils.append_to_file(logfile, "UPDATE best lr acc= {}\n".format(best_acc))

            # Clean all from previous best
            if manager.store_policy.only_keep_best:
                for out_dir in best_iteration_batch_dirs:
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir, ignore_errors=True)
                        print("[CLEANUP] removing {}".format(out_dir))
            best_iteration_batch_dirs = iteration_batch_dirs
        else:
            if manager.store_policy.only_keep_best:
                for out_dir in iteration_batch_dirs:
                    if os.path.exists(out_dir):
                        shutil.rmtree(out_dir, ignore_errors=True)
                        print("[CLEANUP] removing {}".format(out_dir))
        if manager.store_policy.keep_none:
            for out_dir in iteration_batch_dirs:
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir, ignore_errors=True)
                    print("[CLEANUP] removing {}".format(out_dir))
    print("FINETUNE DONE: best_lr={}, best_acc={}".format(best_lr, best_acc))

    ########################################################
    # POSTPROCESS
    if hasattr(manager.method, 'grid_poststep'):
        manager.method.grid_poststep(args, manager)

    return best_lr, best_acc


class StoragePolicy(object):
    def __init__(self, save_models_mode):
        if save_models_mode not in ['all', 'keep_none', 'only_keep_best']:
            raise Exception("Invalid value for save_models_mode")
        print("save_models_mode={}".format(save_models_mode))

        if save_models_mode == "all":
            self.keep_none = False  # Remove all models
            self.only_keep_best = False  # Only keep the best model saved
        elif save_models_mode == "only_keep_best":
            self.keep_none = False
            self.only_keep_best = True
        elif save_models_mode == 'keep_none':  # Store in each grid iteration the checkpoints, but afterwards remove all
            self.keep_none = True
            self.only_keep_best = False

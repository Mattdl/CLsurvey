import traceback
import os
import time
import torch
import copy
import sys
import operator

import utilities.utils as utils
import methods.method as methods
import framework.lr_grid_train as finetune_single_task


class HyperparameterFramework(object):
    """
    Consists of 2 phases.
    PHASE 1: Finetune LR gridsearch, save model with best acc and pass learning rate.
    PHASE 2: Use finetune LR, and use best acc for thresholding mechanism.
             Threshold T is x% of finetune acc (x is hyperparam), when validation acc below T, the regularization lambda
             is decayed, and a new attempt is tried. Until Max # attempts, then last model is taken (avoid inf loop).
    """

    def __init__(self, method):
        self.hyperparams = method.hyperparams  # Init with method hyperparameters (e.g. propagated prev task)
        self.hyperparams_backup = copy.deepcopy(self.hyperparams)
        self.hyperparam_idx = 0
        self.attempts = 0

    def _restore_state(self, state):
        print("Restoring to state = {}".format(state))
        try:
            for hkey in self.hyperparams.keys():  # Keep reference to original dict object
                self.hyperparams[hkey] = state['hyperparams'][hkey]
                self.hyperparams_backup[hkey] = state['hyperparams_backup'][hkey]
            self.hyperparam_idx = state['hyperparam_idx']
            self.attempts = state['attempts']
        except Exception as e:
            print(e)
            raise AttributeError("Attributes of inner state have changed: src:{} != target:{}".format(
                list(self.hyperparams.keys()), list(state.keys())))
        self._print_status("Restored Framework")

    def _get_state(self):
        return {"hyperparams": self.hyperparams,
                "hyperparams_backup": self.hyperparams_backup,
                "hyperparam_idx": self.hyperparam_idx,
                "attempts": self.attempts, }

    def _print_status(self, title="Framework Status"):
        print("-" * 40, "\n", title)
        for hkey in self.hyperparams.keys():
            print('hyperparam {}={} (backup={})'.format(
                hkey, self.hyperparams[hkey], self.hyperparams_backup[hkey]))
        print("hyperparam_idx={}".format(self.hyperparam_idx))
        print("attempts={}".format(self.attempts))
        print("-" * 40)

    def _save_chkpt(self, args, manager, threshold, task_lr_acc):
        hyperparams = {
            'acc_threshold': threshold, 'val_acc': task_lr_acc,
            'args': vars(args), 'manager': vars(manager), 'state': self._get_state()
        }
        print("Saving hyperparams: {}".format(hyperparams))
        manager.save_hyperparams(manager.heuristic_exp_dir, hyperparams)

    @staticmethod
    def maximalPlasticitySearch(args, manager):
        """ Phase 1. Coarse finetuning gridsearch."""
        start_time = time.time()
        finetune_lr, finetune_acc = finetune_single_task.lr_grid_single_task(args, manager,
                                                                             save_models_mode=args.save_models_mode)
        args.phase1_elapsed_time = time.time() - start_time
        utils.print_timing(args.phase1_elapsed_time, "PHASE 1 FT GRID")
        return finetune_lr, finetune_acc

    def stabilityDecay(self, args, manager, finetune_lr, finetune_acc):
        """ Phase 2. """
        args.lr = finetune_lr  # Set current lr based on previous phase
        manager.heuristic_exp_dir = os.path.join(
            manager.parent_exp_dir, 'task_' + str(args.task_counter), 'TASK_TRAINING')
        if hasattr(manager.method, 'train_init'):  # Setting some paths etc.
            manager.method.train_init(args, manager)

        chkpt_loaded = self.load_chkpt(manager)  # Always Load checkpoints
        if not chkpt_loaded:  # Init state
            self.attempts = 0
            self.hyperparams_backup = copy.deepcopy(self.hyperparams)
        if self.check_succes(manager):  # Skip this phase
            manager.best_model_path = os.path.join(manager.heuristic_exp_dir, 'best_model.pth.tar')  # Set paths
            return

        # PRESTEPS FOR METHODS
        args.presteps_elapsed_time = 0
        if hasattr(manager.method, 'prestep'):
            manager.method.prestep(args, manager)

        # CONTINUE
        max_attempts = args.max_attempts_per_task  # * len(manager.method.hyperparams)
        converged = False
        while not converged and self.attempts < max_attempts:
            print(" => ATTEMPT {}/{}: Hyperparams {}".format(self.attempts, max_attempts - 1, self.hyperparams))
            start_time = time.time()
            try:
                manager.method.hyperparams = self.hyperparams
                model, task_lr_acc = manager.method.train(args, manager, self.hyperparams)
            except:
                traceback.print_exc()
                sys.exit(1)

            # Accuracy on val set should be at least finetune_acc_threshold% of finetuning accuracy
            threshold = finetune_acc * args.inv_drop_margin  # A_ft * (1 - p) defined in paper

            ########################################
            # CONVERGE POLICY
            ########################################
            if task_lr_acc >= threshold:
                print('CONVERGED, (acc = ', task_lr_acc, ") >= (threshold = ", threshold, ")")
                converged = True
                args.convergence_iteration_elapsed_time = time.time() - start_time
                utils.print_timing(args.convergence_iteration_elapsed_time, "PHASE 2 CONVERGED FINAL IT")

            ########################################
            # DECAY POLICY
            ########################################
            else:
                print('DECAY HYPERPARAMS, (acc = ', task_lr_acc, ") < (threshold = ", threshold, ")")
                self.hyperparamDecay(args, manager)
                self.attempts += 1

                # Cleanup unless last attempt
                if self.attempts < max_attempts:
                    print('CLEANUP of previous model')
                    utils.rm_dir(manager.heuristic_exp_dir)
                else:
                    print("RETAINING LAST ATTEMPT MODEL")
                    converged = True

            # CHECKPOINT
            self._save_chkpt(args, manager, threshold, task_lr_acc)
            self._print_status()  # Framework Status

        # POST PREP
        manager.best_model_path = os.path.join(manager.heuristic_exp_dir, 'best_model.pth.tar')
        manager.create_success_token(manager.heuristic_exp_dir)

    def check_succes(self, manager):
        """ Check for success token """
        if os.path.exists(manager.get_success_token_path(manager.heuristic_exp_dir)):
            print("Already Successfull run. Skipping phase 2.")
            return True
        return False

    def load_chkpt(self, manager):
        """ Load checkpoint hyperparams from convergence. """
        utils.create_dir(manager.heuristic_exp_dir)
        hyperparams_path = os.path.join(manager.heuristic_exp_dir, utils.get_hyperparams_output_filename())
        try:
            print("Initiating framework chkpt:{}".format(hyperparams_path))
            chkpt = torch.load(hyperparams_path)
        except:
            print("CHECKPOINT LOAD FAILED: No state to restore, starting from scratch.")
            return False

        self._restore_state(chkpt['state'])
        print("SUCCESSFUL loading framework chkpt:{}".format(hyperparams_path))
        return True

    def hyperparamDecay(self, args, manager):
        """
           Decay strategies for two situations.
            1) Decay Single hyperparam.
            2) Decay of multiple hyperparams:
                Iterate all hyperparams and decay each of them individually (restore others),
                if none works, all of them are decayed together.

                Example decayfactor=0.5:
                    Attempt INIT: lambda = 5, alpha = 2,
                    Attempt 1: lambda = 2.5, alpha = 2      # Only lambda decays (restore values=(5,2))
                    Attempt 2: lambda = 5, alpha = 1        # Lambda is restored, alpha decays (restore values=(5,2))
                    Attempt 3: lambda = 2.5, alpha = 1      # Both decay from restored values
                                                            (update restore values=(5,2)--> (2.5,1))
                    Attempt 4: lambda = 1.25, alpha = 1
                    Attempt 5: lambda = 2.5, alpha = 0.5
                    etc.
        """
        op = manager.method.decay_operator if hasattr(manager.method, 'decay_operator') else operator.mul

        # Single hyperparam decay
        if len(self.hyperparams) == 1:
            hkey, hval = list(self.hyperparams.items())[0]
            before = hval
            self.hyperparams[hkey] = op(self.hyperparams[hkey], args.decaying_factor)
            print("Decayed {} -> {}".format(before, self.hyperparams[hkey]))

        # Multiple hyperparams decay
        else:
            # Decay running and backup (new ref values)
            if self.hyperparam_idx == len(self.hyperparams):
                self.hyperparam_idx = 0  # Reset
                for hkey, hval in self.hyperparams_backup.items():
                    self.hyperparams[hkey] = op(hval, args.decaying_factor)  # Update running hyperparams
                before = copy.deepcopy(self.hyperparams_backup)
                self.hyperparams_backup = copy.deepcopy(self.hyperparams)  # Update backup
                print("DECAYING ALL HYPERPARAMS: {} -> {}".format(before, self.hyperparams))

            # Decay 1, restore others
            else:
                before = copy.deepcopy(self.hyperparams)
                hlist = list(self.hyperparams.items())
                hkey = hlist[self.hyperparam_idx][0]
                self.hyperparams[hkey] = op(self.hyperparams_backup[hkey], args.decaying_factor)  # Decay
                other_keys = [hlist[i][0] for i in range(len(self.hyperparams)) if i != self.hyperparam_idx]
                for other_key in other_keys:
                    self.hyperparams[other_key] = self.hyperparams_backup[other_key]  # Restore
                self.hyperparam_idx += 1  # Next hyperparam
                print("Decayed 1 hyperparam: {} -> {}".format(before, self.hyperparams))


def framework_single_task(args, manager):
    """ Main. """
    if args.task_counter == 1 and not args.train_first_task and not args.wrap_first_task_model:
        print("USING SI AS MODEL FOR FIRST TASK: ", manager.previous_task_model_path)
        return

    # Init
    skip_to_post = args.wrap_first_task_model and args.task_counter == 1  # Put first task SI model in Wrapper model
    hf = HyperparameterFramework(manager.method)

    # Save FT or not
    if args.save_models_FT_heuristic:
        args.save_models_mode = 'all'
    elif manager.method.name == methods.PackNet.name:
        args.save_models_mode = 'only_keep_best'
    else:
        args.save_models_mode = 'keep_none'

    args.phase1_elapsed_time = 0
    args.presteps_elapsed_time = 0
    args.convergence_iteration_elapsed_time = 0
    args.postprocess_time = 0
    print("HEURISTIC BASED METHOD: Task ", args.task_name)

    ########################################
    # DATASETS
    ########################################
    # Dataset for training the task
    if args.task_counter > 1:
        prev_task_name = manager.dataset.get_taskname(args.task_counter - 1)
        args.previous_task_dataset_path = manager.dataset.get_task_dataset_path(task_name=prev_task_name,
                                                                                rnd_transform=False)

        # The importance weights are determined with prev dataset without transforms (e.g. rnd horizontal flips).
        manager.reg_sets = [manager.dataset.get_task_dataset_path(task_name=prev_task_name, rnd_transform=False)]
        print('reg_sets=', manager.reg_sets)

    # LWF/EBLL: Heads are stacked in classifier, define starting idx
    args.classifier_heads_starting_idx = manager.base_model.last_layer_idx
    print("classifier_heads_starting_idx = ", args.classifier_heads_starting_idx)

    if not skip_to_post:
        ##############################################################################
        # PHASE 1
        print("\nPHASE 1 (TASK {})".format(args.task_counter))
        ft_lr, ft_acc = hf.maximalPlasticitySearch(args, manager)

        ##############################################################################
        # PHASE2
        print("\nPHASE 2 (TASK {})".format(args.task_counter))
        print("*" * 20, " FT LR ", ft_lr, "*" * 20)
        hf.stabilityDecay(args, manager, ft_lr, ft_acc)

    ########################################
    # POSTPROCESS
    ########################################
    if hasattr(manager.method, 'poststep'):
        manager.method.poststep(args, manager)

    ########################################
    # NEXT TASK INIT
    ########################################
    if hasattr(manager.method, 'init_next_task'):
        manager.method.init_next_task(manager)
    else:
        manager.previous_task_model_path = manager.best_model_path

    print('phase1_elapsed_time={}, '
          'presteps_elapsed_time={}, '
          'convergence_iteration_elapsed_time={}, '
          'postprocess_time={}'.format(args.phase1_elapsed_time,
                                       args.presteps_elapsed_time,
                                       args.convergence_iteration_elapsed_time,
                                       args.postprocess_time))

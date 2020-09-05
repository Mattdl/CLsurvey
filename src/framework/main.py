"""Main training script."""
import traceback
import argparse
import shutil
import os
import torch

import framework.lr_grid_train as lr_grid_single_task
import framework.framework_train as heuristic_single_task
import framework.eval as test

import models.net as nets
import utilities.utils as utils
import data.dataset as datasets
import methods.method as methods

parser = argparse.ArgumentParser(description='Continual Learning Survey: Hyperparameter Selection Framework')

# REQUIRED ARGS
parser.add_argument('model_name', type=str, help="e.g. base_vgg9_cl_512_512")
parser.add_argument('--method_name', type=str, default=None)
parser.add_argument('--ds_name', type=str, default=None)

# PATHS
parser.add_argument('--gridsearch_name', type=str, default="demo",
                    help="root/<gridsearch_name>/<exp_name>/ grouping experiments.")
parser.add_argument('--exp_name', type=str, default=None,
                    help="root/<gridsearch_name>/<exp_name>/ Default <exp_name> determined by hyperparams")

# DEFAULT SCRIPT VALUES
parser.add_argument('--starting_task_count', type=int, default=1,
                    help="Task number to start training from. Default determined by max in dataset.")
parser.add_argument('--max_task_count', type=int, default=None,
                    help="Max task number to train upon. Default determined by max in dataset.")
parser.add_argument('--finetune_iterations', type=int, default=1,
                    help="Amount of training runs per lr in the max plasticity search")
parser.add_argument('--saving_freq', type=int, default=20,
                    help="How frequently to store a model checkpoint (in terms of epochs)")
parser.add_argument('--save_models_FT_heuristic', action="store_true",
                    help="Save chkpt models during FT phase of heuristic.")

# Special Run Modes
runmodes = ["first_task_basemodel_dump", "timing_mode", "debug"]
parser.add_argument('--runmode', default=None, choices=runmodes,
                    help="first_task_basemodel_dump: Train SI firs task model, reference model for other methods.\n"
                         "timing_mode: Measure timing setup.\n"
                         "debug: Debug mode.")
parser.add_argument('--cleanup_exp', action="store_true",
                    help="Remove all results of the experiment if existed before.")

# FRAMEWORK PARAMS
parser.add_argument('--drop_margin', type=float, default=0.2, help="Max drop best max-plasticity-search accuracy.")
parser.add_argument('--decaying_factor', type=float, default=0.5)
parser.add_argument('--max_attempts_per_task', type=int, default=10,
                    help="Max amount of training attempts to stop infinite erroneous training.")
# HYPERPARAMS
parser.add_argument('--hyperparams', type=str, default="",
                    help="'param1,param2,...'  with 'def' giving default value. Order defined in methods/method.py")
parser.add_argument('--static_hyperparams', type=str, default="",
                    help="'p1,p2,...' Method hyperparams not included in the hyperparam decay set.")
parser.add_argument('--lr_grid', type=str, default="1e-2,5e-3,1e-3,5e-4,1e-4",
                    help="List of learning rates (comma-separated)")
parser.add_argument('--boot_lr_grid', type=str, default=None,
                    help="Learning rates list to bootstrap first task (comma-separated)")
parser.add_argument('--num_epochs', type=int, default=70, help="Epochs for both framework phases.")
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=200)

# EVALUATION
parser.add_argument('--test', action="store_true", help="Evaluate after training (possible in separate run).")
parser.add_argument('--test_max_task_count', type=int, default=None)
parser.add_argument('--test_starting_task_count', type=int, default=1, help="Task dset to start eval from.")
parser.add_argument('--test_overwrite_mode', action="store_true", help="Overwrite eval results if exists.")
parser.add_argument('--test_set', choices=['test', 'val', 'train'], type=str, default='test')


def main(method=None, dataset=None):
    """Training pipeline."""
    utils.init()
    args = parser.parse_args()

    # ADD EXTERNAL PROJECT PATHS
    config = utils.get_parsed_config()
    args.tr_results_root_path = utils.read_from_config(config, 'tr_results_root_path')
    args.models_root_path = utils.read_from_config(config, 'models_root_path')
    args.test_results_root_path = utils.read_from_config(config, 'test_results_root_path')

    # Init method/dataset/model wrapper objects
    if method is None:
        method = methods.parse(args.method_name)
    if dataset is None:
        dataset = datasets.parse(args.ds_name)
    base_model = nets.parse_model_name(args.models_root_path, args.model_name, dataset.input_size)  # Model wrapper init
    print("RUNNING MODEL: ", base_model.name)
    init_checks(args, dataset)  # CHECKS

    # Parse lr
    args.lr_grid = utils.parse_str_to_floatlist(args.lr_grid)
    args.boot_lr_grid = utils.parse_str_to_floatlist(args.boot_lr_grid) if args.boot_lr_grid else args.lr_grid

    # STATIC DEFAULT ARGS
    args.data_dir = None
    args.init_model_path = None
    args.max_task_count = dataset.task_count if args.max_task_count is None else args.max_task_count
    args.inv_drop_margin = 1 - args.drop_margin
    args.first_task_modelname = nets.get_init_modelname(args)

    # Method specific properties
    args.train_first_task = hasattr(method, 'start_scratch') and method.start_scratch
    args.wrap_first_task_model = hasattr(method, 'wrap_first_task_model') and method.wrap_first_task_model
    args.no_framework = hasattr(method, 'no_framework') and method.no_framework  # Baselines/outlier methods

    # Parse Runmodes
    for option in runmodes:  # Set runmode args
        setattr(args, option, args.runmode == option)
    if args.first_task_basemodel_dump:
        overwrite_dump_args(args, method)
    elif args.timing_mode:
        overwrite_timing_args(args)
    elif args.debug:
        overwrite_debug_args(args)

    # Method specific overwrites: e.g. Joint -> 1 task
    if hasattr(method, 'train_args_overwrite'):
        method.train_args_overwrite(args)

    # EXP
    methods.set_hyperparams(method, args.hyperparams)
    methods.set_hyperparams(method, args.static_hyperparams, static_params=True)
    if args.exp_name is None:
        args.exp_name = utils.get_exp_name(args, method)
    print("Experiment name: ", args.exp_name)

    ########################################
    # Config Data and Model paths
    ########################################
    # Where specific training results go (specific tuning dir)
    parent_exp_dir = utils.get_train_results_path(args.tr_results_root_path, dataset, method.name,
                                                  model_name=base_model.name, gridsearch_name=args.gridsearch_name,
                                                  exp_name=args.exp_name)

    # CLEANUP
    if args.cleanup_exp and os.path.isdir(parent_exp_dir):
        assert not args.test, "Can't remove experiment results while evaluating."
        shutil.rmtree(parent_exp_dir)
        print("=====> CLEANING UP EXP: Removing all to start from scratch <=====")

    # INIT TASK
    prev_task_model_path = get_init_model_path(args, base_model, dataset, parent_exp_dir, args.first_task_modelname)
    check_dump(args, parent_exp_dir, dataset, base_model)
    manager = Manager(dataset, method, prev_task_model_path, parent_exp_dir, base_model)  # Manager container

    ########################################
    # Train each of the Tasks
    ########################################
    print("Starting with ARGS={}\nManager={}".format(vars(args), vars(manager)))
    ds_paths, model_paths = [], []
    for task_counter in range(args.starting_task_count, args.max_task_count + 1):
        print("\n", "*" * 80, "\n", "TRAINING Task {}".format(task_counter), "\n", "*" * 80)
        args.task_counter = task_counter
        args.task_name = manager.dataset.get_taskname(args.task_counter)
        args.lrs = args.boot_lr_grid if task_counter == 1 else args.lr_grid
        manager.set_dataset(args)  # Set current task dataset
        try:
            if args.no_framework:
                lr_grid_single_task.lr_grid_single_task(args, manager, save_models_mode='all')
            else:
                heuristic_single_task.framework_single_task(args, manager)
            ds_paths.append(manager.current_task_dataset_path)
            model_paths.append(manager.previous_task_model_path)
        except RuntimeError as e:
            print("ERROR:", e)
            traceback.print_exc()
            break
    utils.print_stats()

    if args.test:
        test.main(args, manager, ds_paths, model_paths)


class Manager(object):
    """Holder object for single task required info."""
    token_name = 'SUCCESS.FLAG'

    def __init__(self, dataset, method, previous_task_model_path, parent_exp_dir, base_model):
        self.dataset = dataset
        self.method = method
        self.previous_task_model_path = previous_task_model_path
        self.parent_exp_dir = parent_exp_dir
        self.base_model = base_model
        self.current_task_dataset_path = None  # Set per task

        # Method specifics
        self.best_finetuned_model_path = None  # e.g. PackNet
        self.autoencoder_model_path = None

    def set_dataset(self, args, rnd_transform=False):
        if hasattr(self.method, 'grid_datafetch'):  # Joint
            self.current_task_dataset_path = self.method.grid_datafetch(args, self.dataset)
        else:
            self.current_task_dataset_path = self.dataset.get_task_dataset_path(
                task_name=args.task_name, rnd_transform=rnd_transform)

    def save_hyperparams(self, output_dir, hyperparams):
        """ Add extra stats (memory,...) and save to output_dir.
        :param output_dir: Dir to export the dictionary to.
        :param hyperparams: Dictionary with hyperparams to save
        """
        utils.create_dir(output_dir)
        hyperparams_outpath = os.path.join(output_dir, utils.get_hyperparams_output_filename())
        torch.save(hyperparams, hyperparams_outpath)
        print("Saved hyperparams to: ", hyperparams_outpath)

    def get_success_token_path(self, exp_dir):
        """ Creates a token for exp_dir, representing successful finishing of training. """
        return os.path.join(exp_dir, self.token_name)

    def create_success_token(self, exp_dir):
        if not os.path.exists(self.get_success_token_path(exp_dir)):
            torch.save('', self.get_success_token_path(exp_dir))


########################################
# CONFIG
########################################
def get_init_model_path(args, base_model, dataset, parent_exp_dir, first_task_modelname):
    if args.starting_task_count == 1:  # Starting from first task
        model_SI_path = utils.get_starting_model_path(args.tr_results_root_path, dataset, base_model.name,
                                                      first_task_modelname, method_name=methods.SI.name)
        if args.train_first_task or args.first_task_basemodel_dump:  # Starting from scratch
            prev_task_model_path = base_model.path
        else:  # STARTING FROM THE FIRST MODEL LEARNED BY PATH INTEGRAL (SI)
            prev_task_model_path = model_SI_path
    else:  # Task > 1: Resume from previous task best model
        model_path = os.path.join(parent_exp_dir, 'task_{}', 'TASK_TRAINING', 'best_model.pth.tar')
        prev_task_model_path = model_path.format(args.starting_task_count - 1)

    if not os.path.exists(prev_task_model_path) and not args.first_task_basemodel_dump:
        raise Exception("NOT EXISTING previous_task_model_path = ", prev_task_model_path)
    print("Starting from model = ", prev_task_model_path)
    return prev_task_model_path


########################################
# CHECKS
########################################
def init_checks(args, dataset):
    if args.starting_task_count < 1 or args.starting_task_count > dataset.task_count:
        raise ValueError("ERROR: Starting task count should be in appropriate range for dataset! Value = ",
                         args.starting_task_count)
    assert 0 <= args.drop_margin <= 1
    assert 0 <= args.decaying_factor <= 1


def check_dump(args, parent_exp_dir, dataset, base_model):
    if args.first_task_basemodel_dump:
        model_path = os.path.join(parent_exp_dir, 'task_1', 'TASK_TRAINING', 'best_model.pth.tar')
        link_path = utils.get_starting_model_path(args.tr_results_root_path, dataset, base_model.name, args.exp_name,
                                                  method_name=methods.SI.name, append_filename=True)
        if os.path.exists(model_path) or os.path.exists(link_path):
            raise Exception("Basemodel/link for SI first task already exists!\n"
                            "Not overwriting, because reference to all other methods.\n"
                            "Manually remove model for a new dump:{}".format(model_path))


########################################
# ARG OVERWRITES
########################################
def overwrite_debug_args(args):
    if args.debug:  # Quick go-through setup
        print('#' * 20, " RUNNING IN DEBUG MODE ", '#' * 20)
        args.finetune_iterations = 1
        args.lrs = [0.01]
        args.num_epochs = 1
        args.batch_size = 200
        args.mem_per_task = 20
        args.saving_freq = 200


def overwrite_dump_args(args, method):
    args.train_first_task = True
    args.starting_task_count = 1
    args.max_task_count = 1
    args.gridsearch_name = "first_task_basemodel"
    args.exp_name = args.first_task_modelname
    assert isinstance(method, methods.SI), "Define SI method to train first task common model."


def overwrite_timing_args(args):
    args.max_task_count = 4  # Avg over 3
    args.lrs = [5e-3]  # reduce gridsearches to 1 iteration
    args.batch_size = 200

    args.save_models_FT_heuristic = False
    args.finetune_iterations = 1
    args.num_epochs = 10  # Avoid early stopping

    args.encoder_dims = [100]  # EBLL
    args.encoder_alphas = [1e-2]
    args.autoencoder_epochs = 10


if __name__ == "__main__":
    main()

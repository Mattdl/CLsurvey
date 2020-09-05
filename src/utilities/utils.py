import os
import configparser
import sys
import random
import numpy
import warnings
import shutil
import datetime
import copy

import torch.nn as nn
import torch


def init():
    set_random()


########################################
# CONFIG PARSING
########################################
def get_root_src_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


def get_parsed_config():
    """ Using this file as reference to get src root: src/<dir>/utils.py"""
    config = configparser.ConfigParser()
    src_path = get_root_src_path()
    config.read(os.path.join(src_path, 'config.init'))

    # Replace "./" with abs src root path
    for key, path in config['DEFAULT'].items():
        if '.' == read_from_config(config, key).split(os.path.sep)[0]:
            config['DEFAULT'][key] = os.path.join(src_path, read_from_config(config, key)[2:])
            create_dir(config['DEFAULT'][key], key)

    return config


def read_from_config(config, key_value):
    return os.path.normpath(config['DEFAULT'][key_value]).replace("'", "").replace('"', "")


def parse_str_to_floatlist(str_in):
    return list(map(float, str_in.replace(' ', '').split(',')))


########################################
# DETERMINISTIC
########################################
def set_random(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_now():
    return str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])


########################################
# PYTORCH UTILS
########################################
def replace_last_classifier_layer(model, out_dim):
    last_layer_index = str(len(model.classifier._modules) - 1)
    num_ftrs = model.classifier._modules[last_layer_index].in_features
    model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, out_dim).cuda()
    return model


def get_first_FC_layer(seq_module):
    """
    :param seq_module: e.g. classifier or feature Sequential module of a model.
    """
    for module in seq_module.modules():
        if isinstance(module, nn.Linear):
            return module
    raise Exception("No LINEAR module in sequential found...")


def save_cuda_mem_req(out_dir, out_filename='cuda_mem_req.pth.tar'):
    """
    :param out_dir: /path/to/best_model.pth.tar
    """
    out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)

    mem_req = {}
    mem_req['cuda_memory_allocated'] = torch.cuda.memory_allocated(device=None)
    mem_req['cuda_memory_cached'] = torch.cuda.memory_cached(device=None)

    torch.save(mem_req, out_path)
    print("SAVED CUDA MEM REQ {} to path: {}".format(mem_req, out_path))


def save_preprocessing_time(out_dir, time, out_filename='preprocess_time.pth.tar'):
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)
    torch.save(time, out_path)
    print_timing(time, "PREPROCESSING")


def print_timing(timing, title=''):
    title = title.strip() + ' ' if title != '' else title
    print("{}TIMING >>> {} <<<".format(title, str(timing)))


def reset_stats():
    try:
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        print("RESETTED STATS")
    except:
        print("PYTORCH VERSION NOT ENABLING RESET STATS")


def print_stats():
    print("CUDA MAX MEM ALLOC >>> {} <<<".format(torch.cuda.max_memory_allocated()))
    print("CUDA MAX MEM CACHE >>> {} <<<".format(torch.cuda.max_memory_cached()))


########################################
# EXP PATHS
########################################
def get_exp_name(args, method):
    exp_name = ["dm={}".format(args.drop_margin),
                "df={}".format(args.decaying_factor),  # framework_hyperparams
                "e={}".format(args.num_epochs),
                "bs={}".format(args.batch_size)]
    if args.weight_decay != 0:
        exp_name.append("L2={}".format(args.weight_decay))
    for h_key, h_val in method.hyperparams.items():
        exp_name.append("{}={}".format(h_key, h_val))
    if hasattr(method, 'static_hyperparams'):
        for h_key, h_val in method.static_hyperparams.items():
            exp_name.append("{}={}".format(h_key, h_val))
    exp_name = '_'.join(exp_name)
    return exp_name


def get_starting_model_path(root_path, dataset_obj, model_name, exp_name, method_name, append_filename=True):
    """
    All methods have the same model for the same task, as no forgetting mechanism is applied.
    Nevertheless, SI estimates during training time, therefore we train SI first model and use it as starting model for
    all other methods sharing the same model and data set.

    Target, e.g.:
        /survey/exp_results/tiny_imgnet/SI/small_VGG9_cl_128_128/gridsearch/
        first_task_basemodel/<exp_name>/best_model.pth.tar
    :param exp_name: vanilla, L2=0.01,...
    """

    path = os.path.join(root_path, dataset_obj.train_exp_results_dir, method_name, model_name)
    path = os.path.join(path, 'gridsearch', 'first_task_basemodel', exp_name, 'task_1', 'TASK_TRAINING')

    if append_filename:
        path = os.path.join(path, 'best_model.pth.tar')
    return path


def get_test_results_path(root_path, dataset_obj, method_name, model_name,
                          gridsearch_name=None, exp_name=None, subset='test', create=False):
    """
    Util method to get the path of the testing results. (e.g. testing performances,...)
    :param dataset_obj:     CustomDataset object
    :param method_name:      str method name or Method object
    """
    path = os.path.join(root_path, 'results', dataset_obj.test_results_dir, method_name, model_name)

    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        if subset != 'test':
            exp_name = '{}_{}'.format(exp_name, subset)
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    return path


def get_test_results_filename(method_name, task_number):
    return "test_method_performances" + method_name + str(int(task_number) - 1) + ".pth"


def get_train_results_path(tr_results_root_path, dataset_obj, method_name=None, model_name=None, gridsearch=True,
                           gridsearch_name=None, exp_name=None, filename=None, create=False):
    """
    Util method to get the path of the training results. (e.g. models, hyperparams during training,...)

    :param dataset_obj:     CustomDataset object
    :param method_name:      str method name or Method object
    """

    if create and filename is not None:
        print("WARNING: filename is not being created, but superdirs are if not existing.")

    path = os.path.join(tr_results_root_path, dataset_obj.train_exp_results_dir)
    if method_name is not None:
        path = os.path.join(path, method_name)
    if model_name is not None:
        path = os.path.join(path, model_name)
    if gridsearch:
        path = os.path.join(path, 'gridsearch')
    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    if filename is not None:
        path = os.path.join(path, filename)
    return path


def get_perf_output_filename(method_name, dataset_index, joint_full_batch=False):
    """
    Performances filename saved during training. (e.g. accuracy,...)
    :param dataset_index:  task_idx - 1
    """
    if joint_full_batch:
        return 'test_method_performancesJOINT_FULL_BATCH.pth'
    else:
        return 'test_method_performances' + method_name + str(dataset_index) + ".pth"


def get_hyperparams_output_filename():
    return 'hyperparams.pth.tar'


def get_prev_heads(prev_head_model_paths, head_layer_idx):
    """
    Last of the head_model_paths is the target head. Evaluating e.g. on Task 3, means heads of Task 1,2,3 available in
    head_model_paths. A model trained up to a certain task, has all heads available up to this task.
    Can be used to analyse per-head performance.

    :param prev_head_model_paths:   Previous Models to extract head from (the model's last task)
    :param head_layer_idx:      Model idx you want to know accuracy from
    """
    if not isinstance(prev_head_model_paths, list):
        prev_head_model_paths = [prev_head_model_paths]

    if len(prev_head_model_paths) == 0:
        return []

    heads = []
    # Add prev model heads
    for head_model_path in prev_head_model_paths:
        previous_model_ft = torch.load(head_model_path)
        if isinstance(previous_model_ft, dict):
            previous_model_ft = previous_model_ft['model']

        head = previous_model_ft.classifier._modules[head_layer_idx]
        assert isinstance(head, torch.nn.Linear), type(head)
        heads.append(copy.deepcopy(head.cuda()))
        del previous_model_ft

    return heads


########################################
# PATH UTILS
########################################
def get_immediate_subdirectories(parent_dir_path, path_mode=False, sort=False):
    """

    :param parent_dir_path: dir to take subdirs from
    :param path_mode: if true, returns subdir paths instead of names
    :return: List with names (not paths) of immediate subdirs
    """
    if not path_mode:
        dirs = [name for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    else:
        dirs = [os.path.join(parent_dir_path, name) for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    if sort:
        dirs.sort()
    return dirs


def get_relative_path(absolute_path, segments=1):
    """ Returns relative path with last #segments of the absolute path. """
    return os.path.sep.join(list(filter(None, absolute_path.split(os.path.sep)))[-segments:])


def attempt_move(src_path, dest_path):
    try:
        shutil.move(src_path, dest_path)
    except Exception:
        if not os.path.exists(dest_path):  # Don't print if already transfered
            print("Dest path not existing: ", dest_path)
        if not os.path.exists(src_path):
            print("SRC path not existing: ", src_path)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def append_to_file(filepath, msg):
    """ Append a new line to a file, and create if file doesn't exist yet. """
    write_mode = 'w' if not os.path.exists(filepath) else 'a'
    with open(filepath, write_mode) as f:
        f.write(msg + "\n")


def rm_dir(path_to_dir, delete_subdirs=True, content_only=True):
    if path_to_dir is not None and os.path.exists(path_to_dir):
        for the_file in os.listdir(path_to_dir):
            file_path = os.path.join(path_to_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and delete_subdirs:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print("REMOVED CONTENTS FROM DIR: ", path_to_dir)

        if not content_only:
            try:
                shutil.rmtree(path_to_dir)
            except Exception as e:
                print(e)
            print("REMOVED DIR AND ALL ITS CONTENTS: ", path_to_dir)


def create_dir(dirpath, print_description=""):
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, mode=0o750)
    except Exception as e:
        print(e)
        print("ERROR IN CREATING ", print_description, " PATH:", dirpath)


def create_symlink(src, ln):
    if not os.path.exists(ln):
        create_dir(os.path.dirname(ln))
        os.symlink(src, ln)


########################################
# MISCELLANEOUS UTILS
########################################
def float_to_scientific_str(value, sig_count=1):
    """
    {0:.6g}.format(value) also works

    :param value:
    :param sig_count:
    :return:
    """
    from decimal import Decimal
    format_str = '%.' + str(sig_count) + 'E'
    return format_str % Decimal(value)


def debug_add_sys_args(string_cmd, set_debug_option=True):
    """
    Add debug arguments as params, this is for IDE debugging usage.
    :param string_cmd:
    :return:
    """
    warnings.warn("=" * 20 + "SEVERE WARNING: DEBUG CMD ARG USED, TURN OF FOR REAL RUNS" + "=" * 20)
    args = string_cmd.split(' ')
    if set_debug_option:
        args.insert(0, "--debug")
    for arg in args:
        sys.argv.append(str(arg))

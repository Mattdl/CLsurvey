from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import OrderedDict
import os
import time
import warnings
import itertools
import copy

import torch
from torch.autograd import Variable

import utilities.utils
import data.dataset as dataset_utils
import models.net as models
from data.imgfolder import ConcatDatasetDynamicLabels
from models.net import ModelRegularization

import framework.inference as test_network

import methods.EWC.main_EWC as trainEWC
import methods.SI.main_SI as trainSI
import methods.MAS.main_MAS as trainMAS
import methods.LwF.main_LWF as trainLWF
import methods.EBLL.Finetune_SGD_EBLL as trainEBLL
import methods.packnet.main as trainPacknet
import methods.rehearsal.main_rehearsal as trainRehearsal
import methods.HAT.run as trainHAT
import methods.IMM.main_L2transfer as trainIMM
import methods.IMM.merge as mergeIMM
import methods.Finetune.main_SGD as trainFT


# PARSING
def parse(method_name):
    """Parse arg string to actual object."""
    # Exact
    if method_name == YourMethod.name:  # Parsing Your Method name as argument
        return YourMethod()

    elif method_name == EWC.name:
        return EWC()
    elif method_name == MAS.name:
        return MAS()
    elif method_name == SI.name:
        return SI()

    elif method_name == EBLL.name:
        return EBLL()
    elif method_name == LWF.name:
        return LWF()

    elif method_name == GEM.name:
        return GEM()
    elif method_name == ICARL.name:
        return ICARL()

    elif method_name == PackNet.name:
        return PackNet()
    elif method_name == HAT.name:
        return HAT()

    elif method_name == Finetune.name:
        return Finetune()
    elif method_name == FinetuneRehearsalFullMem.name:
        return FinetuneRehearsalFullMem()
    elif method_name == FinetuneRehearsalPartialMem.name:
        return FinetuneRehearsalPartialMem()

    elif method_name == Joint.name:
        return Joint()

    # Modes
    elif IMM.name in method_name:  # modeIMM,meanIMM
        mode = method_name.replace('_', '').replace(IMM.name, '').strip()
        return IMM(mode)
    else:
        raise NotImplementedError("Method not yet parseable")


class Method(ABC):
    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def eval_name(self): pass

    @property
    @abstractmethod
    def category(self): pass

    @property
    @abstractmethod
    def extra_hyperparams_count(self): pass

    @property
    @abstractmethod
    def hyperparams(self): pass

    @classmethod
    def __subclasshook__(cls, C):
        return False

    @abstractmethod
    def get_output(self, images, args): pass

    @staticmethod
    @abstractmethod
    def inference_eval(args, manager): pass


class Category(Enum):
    MODEL_BASED = auto()
    DATA_BASED = auto()
    MASK_BASED = auto()
    BASELINE = auto()
    REHEARSAL_BASED = auto()

    def __eq__(self, other):
        """Compare by equality rather than identity."""
        return self.name == other.name and self.value == other.value


####################################################
################ YOUR METHOD #######################
class YourMethod(Method):
    name = "YourMethodName"
    eval_name = name
    category = Category.REHEARSAL_BASED  # Change to your method
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'stability_related_hyperparam': 1})  # Hyperparams to decay
    static_hyperparams = OrderedDict({'hyperparams_not_to_decay': 1024})  # Hyperparams not to decay (e.g. buffer size)
    wrap_first_task_model = False  # Start SI model/ wrap a scratch model in a custom model

    @staticmethod
    def train_args_overwrite(args):
        """
        Overwrite whatever arguments for your method.
        :return: Nothing
        """
        # e.g. args.starting_task_count = 1 #(joint)
        pass

    # PREPROCESS: MAXIMAL PLASTICITY SEARCH
    def grid_prestep(self, args, manager):
        """Processing before starting first phase. e.g. PackNet modeldump for first task."""
        pass

    # MAXIMAL PLASTICITY SEARCH
    @staticmethod
    def grid_train(args, manager, lr):
        """
        Train for finetuning gridsearch learning rate.
        :return: best model, best accuracy
        """
        return Finetune.grid_train(args, manager, lr)  # Or your own FT-related access point

    # POSTPROCESS: 1st phase
    @staticmethod
    def grid_poststep(args, manager):
        """ Postprocessing after max plasticity search."""
        Finetune.grid_poststep(args, manager)

    # STABILITY DECAY
    def train(self, args, manager, hyperparams):
        """
        Train for stability decay iteration.
        :param args/manager: paths and flags, see other methods and main pipeline.
        :param hyperparams: current hyperparams to use for your method.
        :return: best model and accuracy
        """
        print("Your Method: Training")
        return {}, 100

    # POSTPROCESS 2nd phase
    def poststep(self, args, manager):
        """
        Define some postprocessing after the two framework phases. (e.g. iCaRL define exemplars this task)
        :return: Nothing
        """
        pass

    # INFERENCE ACCESS POINT
    @staticmethod
    def inference_eval(args, manager):
        """
        Loads and defines models and heads for evaluation.
        :param args/manager: paths etc.
        :return: accuracy
        """
        return Finetune.inference_eval(args, manager)

    # INFERENCE
    def get_output(self, images, args):
        """
        Get the output for your method. (e.g. iCaRL first selects subset of the single-head).
        :param images: input images
        :return: the network outputs
        """
        # offset1, offset2 = args.model.compute_offsets(args.current_head_idx, args.model.cum_nc_per_task)  # iCaRL
        # outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return args.model(Variable(images))

    ###################################################
    ###### OPTIONALS = Only define when required ######
    ###################################################

    # OPTIONAL: DATASET MERGING (JOINT): DEFINE DSET LIST
    # @staticmethod
    # def grid_datafetch(args, dataset):
    #     """ Only define for list of datasets to append (see Joint)."""
    #     max_task = dataset.task_count  # Include all datasets in the list
    #     current_task_dataset_path = [dataset.get_task_dataset_path(
    #         task_name=dataset.get_taskname(ds_task_counter), rnd_transform=False)
    #         for ds_task_counter in range(1, max_task + 1)]  # Merge current task dataset with all prev task ones
    #     print("Running JOINT for task ", args.task_name, " on datasets: ", current_task_dataset_path)
    #     return current_task_dataset_path

    # OPTIONAL: DATASET MERGING (JOINT): DEFINE IMGFOLDER
    # @staticmethod
    # def compose_dataset(dataset_path, batch_size):
    #     return Finetune.compose_dataset(dataset_path, batch_size)


##################################################
################ Functions #######################
# Defaults
def get_output_def(model, heads, images, current_head_idx, final_layer_idx):
    head = heads[current_head_idx]
    model.classifier._modules[final_layer_idx] = head  # Change head
    model.eval()
    outputs = model(Variable(images))
    return outputs


def set_hyperparams(method, hyperparams, static_params=False):
    """ Parse hyperparameter string using ';' for hyperparameter list value, single value floats using ','.
        e.g. 0.5,300 -> sets hyperparam1=0.5, hyperparam2=300.0
        e.g. 0.1,0.2;5.2,300 -> sets hyperparam1=[0.1, 0.2], hyperparam2=[5.2, 300.0]
    """
    assert isinstance(hyperparams, str)
    leave_default = lambda x: x == 'def' or x == ''
    hyperparam_vals = []
    split_lists = [x.strip() for x in hyperparams.split(';') if len(x) > 0]
    for split_list in split_lists:
        split_params = [float(x) for x in split_list.split(',') if not leave_default(x)]
        split_params = split_params[0] if len(split_params) == 1 else split_params
        if len(split_lists) == 1:
            hyperparam_vals = split_params
        else:
            hyperparam_vals.append(split_params)

    if static_params:
        if not hasattr(method, 'static_hyperparams'):
            print("No static hyperparams to set.")
            return
        target = method.static_hyperparams
    else:
        target = method.hyperparams

    for hyperparam_idx, (hyperparam_key, def_val) in enumerate(target.items()):
        if hyperparam_idx < len(hyperparam_vals):
            arg_val = hyperparam_vals[hyperparam_idx]
            if leave_default(arg_val):
                continue
            target[hyperparam_key] = arg_val
            print("Set value {}={}".format(hyperparam_key, target[hyperparam_key]))
        else:
            print("Retaining default value {}={}".format(hyperparam_key, def_val))

    method.init_hyperparams = copy.deepcopy(target)  # Backup starting hyperparams
    print("INIT HYPERPARAMETERS: {}".format(target))


#####################################################
################ SOTA Methods #######################

# REHEARSAL
class GEM(Method):
    name = "GEM"
    eval_name = name
    category = Category.REHEARSAL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'margin': 1})
    static_hyperparams = OrderedDict({'mem_per_task': 1024})
    wrap_first_task_model = True

    def train(self, args, manager, hyperparams):
        print("Rehearsal: GEM")
        return _rehearsal_accespoint(args, manager, hyperparams['margin'], self.static_hyperparams['mem_per_task'],
                                     'gem')

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return outputs

    def poststep(self, args, manager):
        """ GEM only needs to collect exemplars for the first SI model. """
        if args.task_counter > 1:
            return

        print("POSTPROCESS PIPELINE")
        start_time = time.time()
        save_path = manager.best_model_path  # Save wrapped SI model in first task best_model_path
        prev_model_path = manager.previous_task_model_path

        if os.path.exists(save_path):
            print("SKIPPING POSTPROCESS: ALREADY DONE")
        else:
            _rehearsal_accespoint(args, manager, self.hyperparams['margin'], self.static_hyperparams['mem_per_task'],
                                  'gem', save_path, prev_model_path,
                                  postprocess=args.task_counter == 1)

        args.postprocess_time = time.time() - start_time
        manager.best_model_path = save_path  # New best model (will be used for next task)

    def grid_train(self, args, manager, lr):
        args.lr = lr
        return _rehearsal_accespoint(args, manager, 0, self.static_hyperparams['mem_per_task'], 'gem',
                                     save_path=manager.gridsearch_exp_dir, finetune=True)

    @staticmethod
    def inference_eval(args, manager):
        return FinetuneRehearsalFullMem.inference_eval(args, manager)


class ICARL(Method):
    name = "ICARL"
    eval_name = name
    category = Category.REHEARSAL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 10})
    static_hyperparams = OrderedDict({'mem_per_task': 1024})
    wrap_first_task_model = True

    def train(self, args, manager, hyperparams):
        print("Rehearsal: ICARL")
        return _rehearsal_accespoint(args, manager, hyperparams['lambda'], self.static_hyperparams['mem_per_task'],
                                     'icarl')

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx, args=args)
        outputs = outputs[:, offset1: offset2]
        return outputs

    def poststep(self, args, manager):
        """ iCARL always needs this step to collect the exemplars. """
        print("POSTPROCESS PIPELINE")
        start_time = time.time()
        if args.task_counter == 1:
            save_path = manager.best_model_path  # Save wrapped SI model in first task best_model_path for iCarl
            prev_model_path = manager.previous_task_model_path  # SI common model first task (shared)
        else:
            save_path = os.path.join(manager.heuristic_exp_dir, 'best_model_postprocessed.pth.tar')
            prev_model_path = manager.best_model_path

        if os.path.exists(save_path):
            print("SKIPPING POSTPROCESS: ALREADY DONE")
        else:
            _rehearsal_accespoint(args, manager,
                                  self.hyperparams['lambda'], self.static_hyperparams['mem_per_task'], 'icarl',
                                  save_path, prev_model_path, postprocess=True)

        args.postprocess_time = time.time() - start_time
        manager.best_model_path = save_path  # New best model (will be used for next task)

    def grid_train(self, args, manager, lr):
        args.lr = lr
        return _rehearsal_accespoint(args, manager, 0, self.static_hyperparams['mem_per_task'], 'icarl',
                                     save_path=manager.gridsearch_exp_dir, finetune=True)

    @staticmethod
    def inference_eval(args, manager):
        return FinetuneRehearsalFullMem.inference_eval(args, manager)


def _rehearsal_accespoint(args, manager, memory_strength, mem_per_task, method_arg,
                          save_path=None, prev_model_path=None, finetune=False, postprocess=False):
    nc_per_task = dataset_utils.get_nc_per_task(manager.dataset)
    total_outputs = sum(nc_per_task)
    print("nc_per_task = {}, TOTAL OUTPUTS = {}".format(nc_per_task, total_outputs))

    save_path = manager.heuristic_exp_dir if save_path is None else save_path
    prev_model_path = manager.previous_task_model_path if prev_model_path is None else prev_model_path

    manager.overwrite_args = {
        'weight_decay': args.weight_decay,
        'task_name': args.task_name,
        'task_count': args.task_counter,
        'prev_model_path': prev_model_path,
        'save_path': save_path,
        'n_outputs': total_outputs,
        'method': method_arg,
        'n_memories': mem_per_task,
        'n_epochs': args.num_epochs,
        'memory_strength': memory_strength,
        'cuda': True,
        'dataset_path': manager.current_task_dataset_path,
        'n_tasks': manager.dataset.task_count,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'finetune': finetune,  # FT mode for iCarl/GEM
        'is_scratch_model': args.task_counter == 1,
        'postprocess': postprocess,
    }
    model, task_lr_acc = trainRehearsal.main(manager.overwrite_args, nc_per_task)
    return model, task_lr_acc


# MASK BASED
class PackNet(Method):
    name = "packnet"
    eval_name = name
    category = Category.MASK_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'prune_perc_per_layer': 0.9})
    grid_chkpt = True
    start_scratch = True

    def __init__(self):
        self.pruned_savename = None

    @staticmethod
    def get_dataset_name(task_name):
        return 'survey_TASK_' + task_name

    def train_init(self, args, manager):
        self.pruned_savename = os.path.join(manager.heuristic_exp_dir, 'best_model_PRUNED')

    def train(self, args, manager, hyperparams):
        prune_lr = args.lr * 0.1  # FT LR, order 10 lower

        print("PACKNET PRUNE PHASE")
        manager.overwrite_args = {
            'weight_decay': args.weight_decay,
            'train_path': manager.current_task_dataset_path,
            'test_path': manager.current_task_dataset_path,
            'mode': 'prune',
            'dataset': self.get_dataset_name(args.task_name),
            'loadname': manager.best_finetuned_model_path,  # Load FT trained model
            'post_prune_epochs': 10,
            'prune_perc_per_layer': hyperparams['prune_perc_per_layer'],
            'lr': prune_lr,
            'finetune_epochs': args.num_epochs,
            'cuda': True,
            'save_prefix': self.pruned_savename,  # exp path filename
            'train_bn': args.train_bn,
            'saving_freq': args.saving_freq,
            'current_dataset_idx': args.task_counter,
        }
        task_lr_acc = trainPacknet.main(manager.overwrite_args)
        return None, task_lr_acc

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    def init_next_task(self, manager):
        assert self.pruned_savename is not None
        if os.path.exists(self.pruned_savename + "_final.pth.tar"):
            manager.previous_task_model_path = self.pruned_savename + "_final.pth.tar"
        elif os.path.exists(self.pruned_savename + "_postprune.pth.tar"):
            warnings.warn("Final file not found(no final file saved if finetune gives no improvement)! Using postprune")
            manager.previous_task_model_path = self.pruned_savename + "_postprune.pth.tar"
        else:
            raise Exception("Previous task pruned model final/postprune non-existing: {}".format(self.pruned_savename))

    def grid_prestep(self, args, manager):
        """ Make modeldump. """
        hyperparams = {}
        manager.dataset_name = self.get_dataset_name(args.task_name)
        manager.disable_pruning_mask = False

        # Make init dump of Wrapper Model object
        if args.task_counter == 1:
            init_wrapper_model_name = os.path.join(
                manager.ft_parent_exp_dir, manager.base_model.name + '_INIT_WRAPPED.pth')

            if not os.path.exists(init_wrapper_model_name):
                if isinstance(manager.base_model, models.AlexNet):
                    arch = 'alexnet'
                else:
                    arch = 'VGGslim_nopretrain'

                print("PACKNET INIT DUMP PHASE")
                overwrite_args = {
                    'arch': arch,
                    'init_dump': True,
                    'cuda': True,
                    'loadname': manager.previous_task_model_path,  # Raw model path
                    'save_prefix': init_wrapper_model_name,  # exp path filename
                    'last_layer_idx': manager.base_model.last_layer_idx,  # classifier last layer idx
                    'current_dataset_idx': args.task_counter,
                }
                hyperparams['pre_phase'] = overwrite_args
                trainPacknet.main(overwrite_args)
            else:
                "PACKNET MODEL DUMP ALREADY EXISTS"

            # Update to wrapper Model path
            manager.previous_task_model_path = init_wrapper_model_name
            manager.disable_pruning_mask = True  # Because packnet assume pretrained

    def grid_train(self, args, manager, lr):
        print("PACKNET TRAIN PHASE")
        ft_savename = os.path.join(manager.gridsearch_exp_dir, 'best_model')
        overwrite_args = {
            'weight_decay': args.weight_decay,
            'disable_pruning_mask': manager.disable_pruning_mask,
            'train_path': manager.current_task_dataset_path,
            'test_path': manager.current_task_dataset_path,
            'mode': 'finetune',
            'dataset': manager.dataset_name,
            'num_outputs': len(manager.dataset.classes_per_task[args.task_name]),
            'loadname': manager.previous_task_model_path,  # Model path
            'lr': lr,
            'finetune_epochs': args.num_epochs,
            'cuda': True,
            'save_prefix': ft_savename,  # exp path filename    # TODO, now only dir, not best_model.pth
            'batch_size': 200,  # batch_size try
            'train_bn': args.train_bn,
            'saving_freq': args.saving_freq,
            'current_dataset_idx': args.task_counter,
        }
        acc = trainPacknet.main(overwrite_args)
        return None, acc

    def grid_poststep(self, args, manager):
        manager.best_finetuned_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')

    @staticmethod
    def train_args_overwrite(args):
        args.train_bn = True if ModelRegularization.batchnorm in args.model_name else False  # train BN params
        print("TRAINING BN PARAMS = ", str(args.train_bn))

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        task_name = manager.dataset.get_taskname(args.eval_dset_idx + 1)
        overwrite_args = {
            'train_path': args.dset_path,
            'test_path': args.dset_path,
            'mode': 'eval',
            'dataset': PackNet.get_dataset_name(task_name),
            'loadname': args.eval_model_path,  # Load model
            'cuda': True,
            'batch_size': args.batch_size,
            'current_dataset_idx': args.eval_dset_idx + 1
        }
        accuracy = trainPacknet.main(overwrite_args)
        return accuracy


class Pathnet(Method):
    name = "pathnet"
    eval_name = name
    category = Category.MASK_BASED
    extra_hyperparams_count = 3  # M,N, gen
    hyperparams = OrderedDict({'N': 3})  # Typically 3,4 defined in paper
    static_hyperparams = OrderedDict({'M': 20, 'generations': 35})  # Allows 2 epochs training per time
    start_scratch = True

    # Do grid generations: [7,35,70]
    def grid_train(self, args, manager, lr):
        args.lr = lr
        parameter = list(self.hyperparams.values()) + list(self.static_hyperparams.values())
        return _modular_accespoint(args, manager, parameter, 'pathnet',
                                   save_path=manager.gridsearch_exp_dir, finetune=True)

    def train(self, args, manager, hyperparams):
        assert args.decaying_factor == 1
        parameter = list(hyperparams.values()) + list(self.static_hyperparams.values())
        return _modular_accespoint(args, manager, parameter, 'pathnet')

    def get_output(self, images, args):
        head = args.heads[args.current_head_idx]
        args.model.classifier = torch.nn.ModuleList()
        args.model.classifier.append(head)  # Change head
        args.model.eval()

        logits = args.model.forward(images, args.task_idx)
        return logits

    @staticmethod
    def decay_operator(a, decaying_factor):
        """ For N, we want it to increment instead of decay, with b >=1"""
        assert decaying_factor == 1
        return int(a + decaying_factor)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class HAT(Method):
    name = "HAT"
    eval_name = name
    category = Category.MASK_BASED
    extra_hyperparams_count = 2  # s,c
    hyperparams = OrderedDict({'smax': 800, 'c': 2.5})  # Paper ranges: smax=[25,800], c=[0.1,2.5] but optimal 0.75
    start_scratch = True

    def grid_train(self, args, manager, lr):
        args.lr = lr
        return _modular_accespoint(args, manager, list(self.hyperparams.values()), 'hat',
                                   save_path=manager.gridsearch_exp_dir, finetune=True)

    def train(self, args, manager, hyperparams):
        return _modular_accespoint(args, manager, list(hyperparams.values()), 'hat')

    def get_output(self, images, args):
        head = args.heads[args.current_head_idx]
        args.model.classifier = torch.nn.ModuleList()
        args.model.classifier.append(head)  # Change head
        args.model.eval()

        logits, masks = args.model.forward(args.task_idx, images, s=args.model.smax)
        return logits

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


def _modular_accespoint(args, manager, parameter, method_arg, save_path=None, prev_model_path=None, finetune=False):
    nc_per_task = dataset_utils.get_nc_per_task(manager.dataset)
    total_outputs = sum(nc_per_task)
    print("nc_per_task = {}, TOTAL OUTPUTS = {}".format(nc_per_task, total_outputs))

    save_path = manager.heuristic_exp_dir if save_path is None else save_path
    prev_model_path = manager.previous_task_model_path if prev_model_path is None else prev_model_path

    manager.overwrite_args = {
        'weight_decay': args.weight_decay,
        'task_name': args.task_name,
        'task_count': args.task_counter,
        'prev_model_path': prev_model_path,
        'model_name': args.model_name,
        'output': save_path,
        'nepochs': args.num_epochs,
        'parameter': parameter,  # CL hyperparam
        'cuda': True,
        'dataset_path': manager.current_task_dataset_path,
        'dataset': manager.dataset,
        'n_tasks': manager.dataset.task_count,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'is_scratch_model': args.task_counter == 1,
        'approach': method_arg,
        'nc_per_task': nc_per_task,
        'finetune_mode': finetune,
        'save_freq': args.saving_freq,
    }
    model, task_lr_acc = trainHAT.main(manager.overwrite_args)
    return model, task_lr_acc


class EWC(Method):
    name = "EWC"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 400})

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        return trainEWC.fine_tune_EWC_acuumelation(dataset_path=manager.current_task_dataset_path,
                                                   previous_task_model_path=manager.previous_task_model_path,
                                                   exp_dir=manager.heuristic_exp_dir,
                                                   data_dir=args.data_dir,
                                                   reg_sets=manager.reg_sets,
                                                   reg_lambda=hyperparams['lambda'],
                                                   batch_size=args.batch_size,
                                                   num_epochs=args.num_epochs,
                                                   lr=args.lr,
                                                   weight_decay=args.weight_decay,
                                                   saving_freq=args.saving_freq)

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class SI(Method):
    name = "SI"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 400})

    # start_scratch = True  # Reference model other methods, should run in basemodel_dump mode

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        return trainSI.fine_tune_elastic(dataset_path=manager.current_task_dataset_path,
                                         num_epochs=args.num_epochs,
                                         exp_dir=manager.heuristic_exp_dir,
                                         model_path=manager.previous_task_model_path,
                                         reg_lambda=hyperparams['lambda'],
                                         batch_size=args.batch_size, lr=args.lr, init_freeze=0,
                                         weight_decay=args.weight_decay,
                                         saving_freq=args.saving_freq)

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class MAS(Method):
    name = "MAS"
    eval_name = name
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 3})

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        return trainMAS.fine_tune_objective_based_acuumelation(
            dataset_path=manager.current_task_dataset_path,
            previous_task_model_path=manager.previous_task_model_path,
            init_model_path=args.init_model_path,
            exp_dir=manager.heuristic_exp_dir,
            data_dir=args.data_dir, reg_sets=manager.reg_sets,
            reg_lambda=hyperparams['lambda'],
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            lr=args.lr, norm='L2', b1=False,
            saving_freq=args.saving_freq,
        )

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class IMM(Method):
    name = "IMM"  # Training name
    eval_name = name  # Altered in init
    modes = ['mean', 'mode']
    category = Category.MODEL_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 0.01})
    grid_chkpt = True
    no_framework = True  # Outlier method (see paper)

    def __init__(self, mode='mode'):
        if mode not in self.modes:
            raise Exception("NO EXISTING IMM MODE: '{}'".format(mode))

        # Only difference is in testing, in training mode/mean IMM are the same
        self.mode = mode  # Set the IMM mode (mean and mode), this is only required after training.
        self.eval_name = self.name + "_" + self.mode

    def set_mode(self, mode):
        """
        Set the IMM mode (mean and mode), this is only required after training.
        :param mode:
        :return:
        """
        if mode not in self.modes:
            raise Exception("TRY TO SET NON EXISTING IMM MODE: ", mode)
        self.mode = mode
        self.eval_name = self.name + "_" + self.mode

    def grid_train(self, args, manager, lr):
        return trainIMM.fine_tune_l2transfer(dataset_path=manager.current_task_dataset_path,
                                             model_path=manager.previous_task_model_path,
                                             exp_dir=manager.gridsearch_exp_dir,
                                             reg_lambda=self.hyperparams['lambda'],
                                             batch_size=args.batch_size,
                                             num_epochs=args.num_epochs,
                                             lr=lr,
                                             weight_decay=args.weight_decay,
                                             saving_freq=args.saving_freq,
                                             )

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def grid_poststep(args, manager):
        manager.previous_task_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')
        print("SINGLE_MODEL MODE: Set previous task model to ", manager.previous_task_model_path)
        Finetune.grid_poststep_symlink(args, manager)

    def eval_model_preprocessing(self, args):
        """ Merging step before evaluation. """
        print("IMM preprocessing: '{}' mode".format(self.mode))
        models_path = mergeIMM.preprocess_merge_IMM(self, args.models_path, args.datasets_path, args.batch_size,
                                                    overwrite=True)
        return models_path

    @staticmethod
    def inference_eval(args, manager):
        return Finetune.inference_eval(args, manager)


class EBLL(Method):
    name = "EBLL"
    eval_name = name
    category = Category.DATA_BASED
    extra_hyperparams_count = 2
    hyperparams = OrderedDict({'reg_lambda': 10, 'ebll_reg_alpha': 1, })
    static_hyperparams = OrderedDict({'autoencoder_lr': [0.01], 'autoencoder_epochs': 50,  # Paper defaults
                                      "encoder_alphas": [1e-1, 1e-2], "encoder_dims": [100, 300]})  # Grid

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def prestep(self, args, manager):
        print("-" * 40)
        print("AUTOENCODER PHASE: for prev task ", args.task_counter - 1)
        manager.autoencoder_model_path = self._autoencoder_grid(args, manager)
        print("AUTOENCODER PHASE DONE")
        print("-" * 40)

    def _autoencoder_grid(self, args, manager):
        """Gridsearch for an autoencoder for the task corresponding with given task counter."""
        autoencoder_parent_exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter - 1),
                                                  'ENCODER_TRAINING')

        # CHECKPOINT
        processed_hyperparams = {'header': ('dim', 'alpha', 'lr')}
        grid_checkpoint_file = os.path.join(autoencoder_parent_exp_dir, 'grid_checkpoint.pth')
        if os.path.exists(grid_checkpoint_file):
            checkpoint = torch.load(grid_checkpoint_file)
            processed_hyperparams = checkpoint
            print("STARTING FROM CHECKPOINT: ", checkpoint)

        # GRID
        best_autoencoder_path = None
        best_autoencoder_acc = 0
        for hyperparam_it in list(itertools.product(self.static_hyperparams['encoder_dims'],
                                                    self.static_hyperparams['encoder_alphas'],
                                                    self.static_hyperparams['autoencoder_lr']
                                                    )):
            encoder_dim, alpha, lr = hyperparam_it
            exp_out_name = "dim={}_alpha={}_lr={}".format(str(encoder_dim), str(alpha), lr)
            autoencoder_exp_dir = os.path.join(autoencoder_parent_exp_dir, exp_out_name)
            print("\n AUTOENCODER SETUP: {}".format(exp_out_name))
            print("Batch size={}, Epochs={}, LR={}, alpha={}, dim={}".format(
                args.batch_size,
                self.static_hyperparams['autoencoder_epochs'],
                lr,
                alpha, encoder_dim))

            if hyperparam_it in processed_hyperparams:
                acc = processed_hyperparams[hyperparam_it]
                print("ALREADY DONE: SKIPPING {}, acc = {}".format(exp_out_name, str(acc)))
            else:
                utilities.utils.create_dir(autoencoder_exp_dir, print_description="AUTOENCODER OUTPUT")

                # autoencoder trained on the previous task dataset
                start_time = time.time()
                _, acc = trainEBLL.fine_tune_Adam_Autoencoder(dataset_path=args.previous_task_dataset_path,
                                                              previous_task_model_path=manager.previous_task_model_path,
                                                              exp_dir=autoencoder_exp_dir,
                                                              batch_size=args.batch_size,
                                                              num_epochs=self.static_hyperparams['autoencoder_epochs'],
                                                              lr=lr,
                                                              alpha=alpha,
                                                              last_layer_name=args.classifier_heads_starting_idx,
                                                              auto_dim=encoder_dim)
                args.presteps_elapsed_time += time.time() - start_time

                processed_hyperparams[hyperparam_it] = acc
                torch.save(processed_hyperparams, grid_checkpoint_file)
                print("Saved to checkpoint")

            print("autoencoder acc={}".format(str(acc)))
            if acc > best_autoencoder_acc:
                utilities.utils.rm_dir(best_autoencoder_path, content_only=False)  # Cleanup
                print("{}(new) > {}(old), New best path: {}".format(str(acc), str(best_autoencoder_acc),
                                                                    autoencoder_exp_dir))
                best_autoencoder_acc = acc
                best_autoencoder_path = autoencoder_exp_dir
            else:
                utilities.utils.rm_dir(autoencoder_exp_dir, content_only=False)  # Cleanup

        if best_autoencoder_acc < 0.40:
            print(
                "[WARNING] Auto-encoder grid not sufficient: max attainable acc = {}".format(str(best_autoencoder_acc)))
        return os.path.join(best_autoencoder_path, 'best_model.pth.tar')

    def train(self, args, manager, hyperparams):
        return trainEBLL.fine_tune_SGD_EBLL(dataset_path=manager.current_task_dataset_path,
                                            previous_task_model_path=manager.previous_task_model_path,
                                            autoencoder_model_path=manager.autoencoder_model_path,
                                            init_model_path=args.init_model_path,
                                            exp_dir=manager.heuristic_exp_dir,
                                            batch_size=args.batch_size,
                                            num_epochs=args.num_epochs,
                                            lr=args.lr,
                                            init_freeze=0,
                                            reg_alpha=hyperparams['ebll_reg_alpha'],
                                            weight_decay=args.weight_decay,
                                            saving_freq=args.saving_freq,
                                            reg_lambda=hyperparams['reg_lambda'])

    def get_output(self, images, args):
        try:
            outputs, _ = args.model(Variable(images))  # disgard autoencoder output codes
        except:
            outputs = args.model(Variable(images))  # SI init model
        if isinstance(outputs, list):
            outputs = outputs[args.current_head_idx]
        return outputs.data

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        return LWF.inference_eval(args, manager)


class LWF(Method):
    name = "LWF"
    eval_name = name
    category = Category.DATA_BASED
    extra_hyperparams_count = 1
    hyperparams = OrderedDict({'lambda': 10})

    def __init__(self, warmup_step=False):
        self.warmup_step = warmup_step

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    def train(self, args, manager, hyperparams):
        # LWF PRE-STEP: WARM-UP (Train only classifier)
        if manager.method.warmup_step:
            print("LWF WARMUP STEP")
            warmup_exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter), 'HEAD_TRAINING')
            trainLWF.fine_tune_freeze(dataset_path=manager.current_task_dataset_path,
                                      model_path=args.previous_task_model_path,
                                      exp_dir=warmup_exp_dir, batch_size=args.batch_size,
                                      num_epochs=int(args.num_epochs / 2),
                                      lr=args.lr)
            args.init_model_path = warmup_exp_dir
            print("LWF WARMUP STEP DONE")
        return trainLWF.fine_tune_SGD_LwF(dataset_path=manager.current_task_dataset_path,
                                          previous_task_model_path=manager.previous_task_model_path,
                                          init_model_path=args.init_model_path,
                                          exp_dir=manager.heuristic_exp_dir,
                                          batch_size=args.batch_size,
                                          num_epochs=args.num_epochs, lr=args.lr, init_freeze=0,
                                          weight_decay=args.weight_decay,
                                          last_layer_name=args.classifier_heads_starting_idx,
                                          saving_freq=args.saving_freq,
                                          reg_lambda=hyperparams['lambda'])

    def get_output(self, images, args):
        outputs = args.model(Variable(images))
        if isinstance(outputs, list):
            outputs = outputs[args.current_head_idx]
        return outputs.data

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        if args.trained_model_idx > 0:
            return FinetuneRehearsalFullMem.inference_eval(args, manager)
        else:  # First is SI model
            return Finetune.inference_eval(args, manager)


##################################################
################ BASELINES #######################
class Finetune(Method):
    name = "finetuning"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True

    def get_output(self, images, args):
        return get_output_def(args.model, args.heads, images, args.current_head_idx, args.final_layer_idx)

    @staticmethod
    def grid_train(args, manager, lr):
        dataset_path = manager.current_task_dataset_path
        print('lr is ' + str(lr))
        print("DATASETS: ", dataset_path)

        if not isinstance(dataset_path, list):  # If single path string
            dataset_path = [dataset_path]

        dset_dataloader, cumsum_dset_sizes, dset_classes = Finetune.compose_dataset(dataset_path, args.batch_size)
        return trainFT.fine_tune_SGD(dset_dataloader, cumsum_dset_sizes, dset_classes,
                                     model_path=manager.previous_task_model_path,
                                     exp_dir=manager.gridsearch_exp_dir,
                                     num_epochs=args.num_epochs, lr=lr,
                                     weight_decay=args.weight_decay,
                                     enable_resume=True,  # Only resume when models saved
                                     save_models_mode=True,
                                     replace_last_classifier_layer=True,
                                     freq=args.saving_freq,
                                     )

    @staticmethod
    def grid_poststep(args, manager):
        manager.previous_task_model_path = os.path.join(manager.best_exp_grid_node_dirname, 'best_model.pth.tar')
        print("SINGLE_MODEL MODE: Set previous task model to ", manager.previous_task_model_path)
        Finetune.grid_poststep_symlink(args, manager)

    @staticmethod
    def grid_poststep_symlink(args, manager):
        """ Create symbolic link to best model in gridsearch. """
        exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(args.task_counter), 'TASK_TRAINING')
        if os.path.exists(exp_dir):
            os.unlink(exp_dir)
        print("Symlink best LR: ", utilities.utils.get_relative_path(manager.best_exp_grid_node_dirname, segments=2))
        os.symlink(utilities.utils.get_relative_path(manager.best_exp_grid_node_dirname, segments=2), exp_dir)

    @staticmethod
    def compose_dataset(dataset_path, batch_size):
        """Append all datasets in list, return single dataloader"""
        dset_imgfolders = {x: [] for x in ['train', 'val']}
        dset_classes = {x: [] for x in ['train', 'val']}
        dset_sizes = {x: [] for x in ['train', 'val']}
        for dset_count in range(0, len(dataset_path)):
            dset_wrapper = torch.load(dataset_path[dset_count])

            for mode in ['train', 'val']:
                dset_imgfolders[mode].append(dset_wrapper[mode])
                dset_classes[mode].append(dset_wrapper[mode].classes)
                dset_sizes[mode].append(len(dset_wrapper[mode]))

        cumsum_dset_sizes = {mode: sum(dset_sizes[mode]) for mode in dset_sizes}
        classes_len = {mode: [len(ds) for ds in dset_classes[mode]] for mode in dset_classes}
        dset_dataloader = {x: torch.utils.data.DataLoader(
            ConcatDatasetDynamicLabels(dset_imgfolders[x], classes_len[x]),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
            for x in ['train', 'val']}  # Concat into 1 dataset
        print("dset_classes: {}, dset_sizes: {}".format(dset_classes, cumsum_dset_sizes))
        return dset_dataloader, cumsum_dset_sizes, dset_classes

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        model = torch.load(args.eval_model_path)

        if isinstance(model, dict):
            model = model['model']

        # Check layer idx correct for current model
        head_layer_idx = str(len(model.classifier._modules) - 1)  # Last head layer of prev model
        current_head = model.classifier._modules[head_layer_idx]
        assert isinstance(current_head, torch.nn.Linear), "NO VALID HEAD IDX"

        # Get head of a prev model corresponding to task
        target_heads = utilities.utils.get_prev_heads(args.head_paths, head_layer_idx)
        target_head_idx = 0  # first in list
        print("EVAL on prev heads: ", args.head_paths)
        assert len(target_heads) == 1

        accuracy = test_network.test_model(manager.method, model, args.dset_path, target_head_idx, subset=args.test_set,
                                           target_head=target_heads, batch_size=args.batch_size,
                                           task_idx=args.eval_dset_idx)
        return accuracy


class FinetuneRehearsalPartialMem(Method):
    name = "finetuning_rehearsal_partial_mem"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    arg_string = 'baseline_rehearsal_partial_mem'
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return outputs

    @staticmethod
    def grid_train(args, manager, lr):
        return FinetuneRehearsalFullMem.grid_train(args, manager, lr)

    @staticmethod
    def grid_poststep(args, manager):
        Finetune.grid_poststep(args, manager)

    @staticmethod
    def inference_eval(args, manager):
        return FinetuneRehearsalFullMem.inference_eval(args, manager)


class FinetuneRehearsalFullMem(Method):
    name = "finetuning_rehearsal_full_mem"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    arg_string = 'baseline_rehearsal_full_mem'
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        offset1, offset2 = args.model.compute_offsets(args.current_head_idx,
                                                      args.model.cum_nc_per_task)  # No shared head
        outputs = args.model(Variable(images), args.current_head_idx)[:, offset1: offset2]
        return outputs

    @staticmethod
    def grid_train(args, manager, lr):
        print("RAW REHEARSAL BASELINE")

        # Need 1 head, because also loss on exemplars of prev tasks is performed
        nc_per_task = manager.datasets.get_nc_per_task(manager.dataset)
        total_outputs = sum(nc_per_task)
        print("nc_per_task = {}, TOTAL OUTPUTS = {}".format(nc_per_task, total_outputs))

        print("RUNNING {} mode".format(manager.method.arg_string))
        overwrite_args = {
            'weight_decay': args.weight_decay,
            'task_name': args.task_name,
            'task_count': args.task_counter,
            'prev_model_path': manager.previous_task_model_path,
            'save_path': manager.gridsearch_exp_dir,
            'n_outputs': total_outputs,
            'method': manager.method.arg_string,
            'n_memories': args.mem_per_task,
            'n_epochs': args.num_epochs,
            'cuda': True,
            'dataset_path': manager.current_task_dataset_path,
            'n_tasks': manager.dataset.task_count,
            'batch_size': args.batch_size,
            'lr': lr,
            'finetune': True,  # Crucial
            'is_scratch_model': args.task_counter == 1
        }
        return trainRehearsal.main(overwrite_args, nc_per_task)

    @staticmethod
    def grid_poststep(args, manager):
        Finetune.grid_poststep(args, manager)

    @staticmethod
    def inference_eval(args, manager):
        """ Inference for testing."""
        model = torch.load(args.eval_model_path)
        target_head_idx = args.eval_dset_idx
        target_heads = None

        print("EVAL on prev head idx: ", target_head_idx)
        accuracy = test_network.test_model(manager.method, model, args.dset_path, target_head_idx, subset=args.test_set,
                                           target_head=target_heads, batch_size=args.batch_size,
                                           task_idx=args.eval_dset_idx)
        return accuracy


class Joint(Method):
    name = "joint"
    eval_name = name
    category = Category.BASELINE
    extra_hyperparams_count = 0
    hyperparams = {}
    grid_chkpt = True
    start_scratch = True
    no_framework = True

    def get_output(self, images, args):
        raise NotImplementedError("JOINT has custom testing method for shared head.")

    @staticmethod
    def grid_train(args, manager, lr):
        return Finetune.grid_train(args, manager, lr)

    @staticmethod
    def grid_datafetch(args, dataset):
        current_task_dataset_path = dataset.get_task_dataset_path(task_name=None, rnd_transform=True)

        if current_task_dataset_path is not None:  # Available preprocessed JOINT dataset
            print("Running JOINT for all tasks as 1 batch, dataset = ", current_task_dataset_path)
            return current_task_dataset_path

        # Merge current task dataset with all prev task ones
        max_task = dataset.task_count  # Include all datasets in the list
        current_task_dataset_path = [dataset.get_task_dataset_path(
            task_name=dataset.get_taskname(ds_task_counter), rnd_transform=False)
            for ds_task_counter in range(1, max_task + 1)]
        print("Running JOINT for task ", args.task_name, " on datasets: ", current_task_dataset_path)
        return current_task_dataset_path

    @staticmethod
    def grid_poststep(args, manager):
        Finetune.grid_poststep(args, manager)

    @staticmethod
    def compose_dataset(dataset_path, batch_size):
        return Finetune.compose_dataset(dataset_path, batch_size)

    @staticmethod
    def train_args_overwrite(args):
        args.starting_task_count = 1
        args.max_task_count = args.starting_task_count

    @staticmethod
    def inference_eval(args, manager):
        return test_network.test_task_joint_model(args.model_path, args.dataset_path, args.dataset_index,
                                                  args.task_lengths, batch_size=args.batch_size, subset='test',
                                                  print_per_class_acc=False, debug=False, tasks_idxes=args.tasks_idxes)

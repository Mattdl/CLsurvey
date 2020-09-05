import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch

import utilities.utils as utils
import data.tinyimgnet_dataprep as dataprep_tiny
import data.inaturalist_dataprep as dataprep_inat
import data.recogseq_dataprep as dataprep_recogseq


def parse(ds_name):
    """Parse arg string to actual object."""
    if ds_name == InaturalistDataset.argname:
        return InaturalistDataset()
    elif ds_name == InaturalistDatasetUnrelToRel.argname:
        return InaturalistDatasetUnrelToRel()
    elif ds_name == InaturalistDatasetRelToUnrel.argname:
        return InaturalistDatasetRelToUnrel()

    elif ds_name == TinyImgnetDataset.argname:
        return TinyImgnetDataset()
    elif ds_name == TinyImgnetDatasetHardToEasy.argname:
        return TinyImgnetDatasetHardToEasy()
    elif ds_name == TinyImgnetDatasetEasyToHard.argname:
        return TinyImgnetDatasetEasyToHard()

    elif ds_name == ObjRecog8TaskSequence.argname:
        return ObjRecog8TaskSequence()

    elif ds_name == LongTinyImgnetDataset.argname:  # Supplemental
        return LongTinyImgnetDataset()

    else:
        raise NotImplementedError("Dataset not parseable: ", ds_name)


def get_nc_per_task(dataset):
    return [len(classes_for_task) for classes_for_task in dataset.classes_per_task.values()]


class CustomDataset(metaclass=ABCMeta):
    """
    Abstract properties/methods that can be used regardless of which subclass the instance is.
    """

    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def argname(self): pass

    @property
    @abstractmethod
    def test_results_dir(self): pass

    @property
    @abstractmethod
    def train_exp_results_dir(self): pass

    @property
    @abstractmethod
    def task_count(self): pass

    @property
    @abstractmethod
    def classes_per_task(self): pass

    @property
    @abstractmethod
    def input_size(self): pass

    @abstractmethod
    def get_task_dataset_path(self, task_name, rnd_transform):
        pass

    @abstractmethod
    def get_taskname(self, task_index):
        pass


class InaturalistDataset(CustomDataset):
    """
    iNaturalist dataset.
    - Raw/NoTransform: The ImageFolder has a transform operator that only resizes (e.g. no RandomHorizontalFlip,...)
    """

    name = 'iNaturalist'
    argname = 'inat'
    test_results_dir = 'inaturalist'
    train_exp_results_dir = 'inaturalist'
    task_count = 10
    classes_per_task = OrderedDict()
    input_size = (224, 224)

    def __init__(self, ordering=None, create=True, overwrite=False):
        config = utils.get_parsed_config()
        self.dataset_root = os.path.join(utils.read_from_config(config, 'ds_root_path'), 'inaturalist', 'train_val2018')

        self.unordered_tasks = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca',
                                'Plantae', 'Reptilia'] if ordering is None else ordering
        print("TASK ORDER: ", self.unordered_tasks)

        # Only the train part of the original iNaturalist is used
        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'

        self.joint_root = self.dataset_root
        self.joint_training_file = 'imgfolder_joint.pth.tar'

        if create:
            # Download the training/validation dataset of iNaturalist
            dataprep_inat.download_dset(os.path.dirname(self.dataset_root))

            # Divide it into our own training/validation/test splits
            dataprep_inat.prepare_inat_trainval(os.path.dirname(self.dataset_root), outfile=self.raw_dataset_file,
                                                # TRAINONLY_trainvaltest_dataset.pth.tar
                                                rnd_transform=False, overwrite=overwrite)
            dataprep_inat.prepare_inat_trainval(os.path.dirname(self.dataset_root),
                                                outfile=self.transformed_dataset_file,
                                                rnd_transform=True, overwrite=overwrite)
            dataprep_inat.prepare_JOINT_dataset(os.path.dirname(self.dataset_root), outfile=self.joint_training_file,
                                                overwrite=overwrite)
        self.min_class_count = 100
        self.random_chances = []

        # Init classes per task
        self.count_total_classes = 0
        print("Task Training-Samples Validation-Samples Classes Random-chance")
        for task_name in self.unordered_tasks:
            dataset_path = self.get_task_dataset_path(task_name=task_name)
            dsets = torch.load(dataset_path)
            dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
            dset_classes = dsets['train'].classes
            del dsets
            self.classes_per_task[task_name] = dset_classes
            self.count_total_classes += len(dset_classes)
            rnd_chance = '%.3f' % (1. / len(dset_classes))
            self.random_chances.append(rnd_chance)
            print("{} {} {} {} {}".format(str(task_name), dset_sizes['train'], dset_sizes['val'], dset_sizes['test'],
                                          len(dset_classes), rnd_chance))
        print("RANDOM CHANCES: ", ", ".join(self.random_chances))
        print("TOTAL CLASSES COUNT = ", self.count_total_classes)

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        # JOINT
        if task_name is None:
            return os.path.join(self.joint_root, self.joint_training_file)

        # PER TASK
        if rnd_transform:
            filename = self.transformed_dataset_file
        else:
            filename = self.raw_dataset_file
        return os.path.join(self.dataset_root, task_name, filename)

    def get_taskname(self, task_count):
        """e.g. Translation of 'Task 1' to the actual name of the first task."""
        if task_count < 1 or task_count > self.task_count:
            raise ValueError('[INATURALIST] TASK COUNT EXCEEDED: count = ', task_count)
        return self.unordered_tasks[task_count - 1]


class InaturalistDatasetRelToUnrel(InaturalistDataset):
    """
    Inaturalsit with diff ordering: from related to unrelated.
    Aves is the largest and taken as init task,
    then each task with highest avg relatedness to all previous tasks is picked.
    """
    task_ordering = ['Aves', 'Mammalia', 'Reptilia', 'Amphibia', 'Animalia', 'Fungi', 'Mollusca', 'Arachnida',
                     'Insecta', 'Plantae']

    suffix = 'ORDERED-rel-to-unrel'
    name = InaturalistDataset.name + ' ' + suffix
    argname = 'inatrelunrel'
    test_results_dir = '_'.join([InaturalistDataset.test_results_dir, suffix])
    train_exp_results_dir = '_'.join([InaturalistDataset.train_exp_results_dir, suffix])

    def __init__(self):
        super().__init__(ordering=self.task_ordering)
        print("INATURALIST ORDERING = ", self.suffix)


class InaturalistDatasetUnrelToRel(InaturalistDataset):
    """
    Inaturalsit with diff ordering: from unrelated to related.
    Starting with biggest: Aves, then based on expert gate: pick most unrelated to all previous tasks (avg).
    """
    task_ordering = ['Aves', 'Fungi', 'Insecta', 'Mollusca', 'Plantae', 'Reptilia', 'Arachnida', 'Mammalia', 'Animalia',
                     'Amphibia']
    suffix = 'ORDERED-unrel-to-rel'
    name = InaturalistDataset.name + ' ' + suffix
    argname = 'inatunrelrel'
    test_results_dir = '_'.join([InaturalistDataset.test_results_dir, suffix])
    train_exp_results_dir = '_'.join([InaturalistDataset.train_exp_results_dir, suffix])

    def __init__(self):
        super().__init__(ordering=self.task_ordering)
        print("INATURALIST ORDERING = ", self.suffix)


class TinyImgnetDataset(CustomDataset):
    name = 'Tiny Imagenet'
    argname = 'tiny'
    test_results_dir = 'tiny_imagenet'
    train_exp_results_dir = 'tiny_imgnet'
    def_task_count, task_count = 10, 10
    classes_per_task = OrderedDict()
    tinyimgnet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    input_size = (64, 64)

    def __init__(self, crop=False, create=True, task_count=10, dataset_root=None, overwrite=False):
        config = utils.get_parsed_config()

        self.dataset_root = dataset_root if dataset_root else os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'tiny-imagenet', 'tiny-imagenet-200')
        print("Dataset root = {}".format(self.dataset_root))
        self.crop = crop
        self.task_count = task_count

        self.transformed_dataset_file = 'imgfolder_trainvaltest_rndtrans.pth.tar'
        self.raw_dataset_file = 'imgfolder_trainvaltest.pth.tar'
        self.joint_dataset_file = 'imgfolder_trainvaltest_joint.pth.tar'

        if create:
            dataprep_tiny.download_dset(os.path.dirname(self.dataset_root))
            dataprep_tiny.prepare_dataset(self, self.dataset_root, task_count=self.task_count, survey_order=True,
                                          overwrite=overwrite)
        # Dataset with bare 64x64, no 56x56 crop
        if not crop:
            self.dataset_root = os.path.join(self.dataset_root, 'no_crop')

        # Version with how many tasks
        self.tasks_subdir = "{}tasks".format(task_count)
        if task_count != self.def_task_count:
            self.test_results_dir += self.tasks_subdir
            self.train_exp_results_dir += self.tasks_subdir

        for task_name in range(1, self.task_count + 1):
            dsets = torch.load(self.get_task_dataset_path(str(task_name)))
            dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
            dset_classes = dsets['train'].classes
            self.classes_per_task[str(task_name)] = dset_classes
            print("Task {}: dset_sizes = {}, #classes = {}".format(str(task_name), dset_sizes, len(dset_classes)))

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name is None:  # JOINT
            return os.path.join(self.dataset_root, self.joint_dataset_file)

        filename = self.transformed_dataset_file if rnd_transform else self.raw_dataset_file
        return os.path.join(self.dataset_root, self.tasks_subdir, task_name, filename)

    def get_taskname(self, task_index):
        return str(task_index)


class LongTinyImgnetDataset(TinyImgnetDataset):
    """Tiny Imagenet split in 40 tasks. (Supplemental exps)"""
    suffix = 'LONG'
    name = 'Tiny Imagenet ' + suffix
    argname = 'longtiny'
    task_count = 40

    def __init__(self, crop=False, create=True, task_count=None, overwrite=False):
        task_count = task_count if task_count else self.task_count
        super().__init__(crop=crop, create=create, task_count=task_count, overwrite=overwrite)


class DifLongTinyImgnetDataset(LongTinyImgnetDataset):
    """ LongTinyImagnet + SVHN task (Supplemental exps)"""
    argname = 'diflongtiny'
    task_count = 41

    def __init__(self, crop=False, create=True, overwrite=False):
        # First 40 tasks
        super().__init__(crop=crop, create=create, task_count=40, overwrite=overwrite)
        # Last task
        self.prepare_extratask(overwrite)

        # Overwrite attributes
        self.task_count = 41
        dsets = torch.load(self.get_task_dataset_path(str(41)))
        dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
        dset_classes = dsets['train'].classes
        self.classes_per_task[str(41)] = dset_classes
        print("Task {}: dset_sizes = {}, #classes = {}".format(str(41), dset_sizes, len(dset_classes)))

    def get_task_dataset_path(self, task_name=None, rnd_transform=False):
        if task_name == super().get_taskname(41):
            return self.outpath
        elif task_name is None:  # Joint
            return None
        else:
            return super().get_task_dataset_path(task_name, rnd_transform)

    def prepare_extratask(self, overwrite):
        from torchvision import transforms
        exp_root = '/path/to/datasets/object_recog_8task_seq'
        dataset_filename = 'dataset_64x64_nornd.pth.tar'
        tr = {x: transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) for x in ['train', 'val', 'test']}
        classes = [str(i) for i in range(1, 11)]
        self.outpath = dataprep_recogseq.prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_SVHN_dataset',
                                                         data_transforms=tr, classes=classes, overwrite=overwrite)
        print("prepared all datasets")


class TinyImgnetDatasetHardToEasy(TinyImgnetDataset):
    """
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/1 -> .../tiny-imagenet-200/no_crop/5
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/2 -> .../tiny-imagenet-200/no_crop/7
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/3 -> .../tiny-imagenet-200/no_crop/10
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/4 -> .../tiny-imagenet-200/no_crop/2
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/5 -> .../tiny-imagenet-200/no_crop/9
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/6 -> .../tiny-imagenet-200/no_crop/8
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/7 -> .../tiny-imagenet-200/no_crop/6
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/8 -> .../tiny-imagenet-200/no_crop/4
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/9 -> .../tiny-imagenet-200/no_crop/3
    CREATE LINK: .../tiny-imagenet-200/no_crop/ORDERED-hard-to-easy/10 -> .../tiny-imagenet-200/no_crop/1
    """
    task_ordering = [5, 7, 10, 2, 9, 8, 6, 4, 3, 1]
    suffix = 'ORDERED-hard-to-easy'
    name = 'Tiny Imagenet ' + suffix
    argname = 'tinyhardeasy'
    test_results_dir = 'tiny_imagenet_' + suffix
    train_exp_results_dir = 'tiny_imgnet_' + suffix

    def __init__(self, crop=False, create=False):
        super().__init__(crop=crop, create=create)
        self.original_dataset_root = self.dataset_root
        self.dataset_root = os.path.join(self.original_dataset_root, self.suffix)
        utils.create_dir(self.dataset_root)
        print(self.dataset_root)

        # Create symbolic links if non-existing
        for task in range(1, self.task_count + 1):
            src_taskdir = os.path.join(self.original_dataset_root, str(self.task_ordering[task - 1]))
            dst_tasklink = os.path.join(self.dataset_root, str(task))
            if not os.path.exists(dst_tasklink):
                os.symlink(src_taskdir, dst_tasklink)
                print("CREATE LINK: {} -> {}".format(dst_tasklink, src_taskdir))
            else:
                print("EXISTING LINK: {} -> {}".format(dst_tasklink, src_taskdir))


class TinyImgnetDatasetEasyToHard(TinyImgnetDataset):
    task_ordering = list(reversed([5, 7, 10, 2, 9, 8, 6, 4, 3, 1]))
    suffix = 'ORDERED-easy-to-hard'
    name = 'Tiny Imagenet ' + suffix
    argname = 'tinyeasyhard'
    test_results_dir = 'tiny_imagenet_' + suffix
    train_exp_results_dir = 'tiny_imgnet_' + suffix

    def __init__(self, crop=False, create=False):
        super().__init__(crop=crop, create=create)
        self.original_dataset_root = self.dataset_root
        self.dataset_root = os.path.join(self.original_dataset_root, self.suffix)
        utils.create_dir(self.dataset_root)
        print(self.dataset_root)

        # Create symbolic links if non-existing
        for task in range(1, self.task_count + 1):
            src_taskdir = os.path.join(self.original_dataset_root, str(self.task_ordering[task - 1]))
            dst_tasklink = os.path.join(self.dataset_root, str(task))
            if not os.path.exists(dst_tasklink):
                os.symlink(src_taskdir, dst_tasklink)
                print("CREATE LINK: {} -> {}".format(dst_tasklink, src_taskdir))
            else:
                print("EXISTING LINK: {} -> {}".format(dst_tasklink, src_taskdir))


class TaskDataset(object):

    def __init__(self, name, imagefolder_path, raw_dataset_path=None, dset_sizes=None, dset_classes=None):
        self.name = name
        self.imagefolder_path = imagefolder_path
        self.raw_dataset_path = raw_dataset_path
        self.dset_sizes = dset_sizes
        self.dset_classes = dset_classes

    def init_size_labels(self, classes_per_task):
        dsets = torch.load(self.imagefolder_path)
        dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
        dset_classes = dsets['train'].classes

        self.dset_sizes = dset_sizes
        self.dset_classes = dset_classes
        classes_per_task[self.name] = dset_classes


class ObjRecog8TaskSequence(CustomDataset):
    """
    Preparation script in rercogseq_dataprep.py (not automated).
    (ImageNet) → Flower → Scenes → Birds → Cars → Aircraft → Actions → Letters → SVHN

    Details:
    Pretrained model on ImageNet
    Task flowers: dset_sizes = {'train': 2040, 'val': 3074, 'test': 3075}, #classes = 102
    Task scenes: dset_sizes = {'train': 5360, 'val': 670, 'test': 670}, #classes = 67
    Task birds: dset_sizes = {'train': 5994, 'val': 2897, 'test': 2897}, #classes = 200
    Task cars: dset_sizes = {'train': 8144, 'val': 4020, 'test': 4021}, #classes = 196
    Task aircraft: dset_sizes = {'train': 6666, 'val': 1666, 'test': 1667}, #classes = 100
    Task actions: dset_sizes = {'train': 3102, 'val': 1554, 'test': 1554}, #classes = 11
    Task letters: dset_sizes = {'train': 6850, 'val': 580, 'test': 570}, #classes = 52
    Task svhn: dset_sizes = {'train': 73257, 'val': 13016, 'test': 13016}, #classes = 11
    """

    name = 'obj_recog_8task_seq'
    argname = 'obj8'
    test_results_dir = 'obj_recog_8task_seq'
    train_exp_results_dir = 'obj_recog_8task_seq'
    task_count = 8
    classes_per_task = OrderedDict()
    input_size = (224, 224)  # For AlexNet

    def __init__(self, crop=False):
        config = utils.get_parsed_config()

        assert not crop, ""
        self.crop = crop
        self.dataset_root = os.path.join(
            utils.read_from_config(config, 'ds_root_path'), 'object_recog_8task_seq')

        # Add Tasks ordered
        dataset_filename = 'dataset.pth.tar'
        self.ordered_tasks = []
        self.ordered_tasks.append(
            TaskDataset('flowers', os.path.join(self.dataset_root, 'Pytorch_Flowers', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('scenes', os.path.join(self.dataset_root, 'Pytorch_Scenes', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('birds', os.path.join(self.dataset_root, 'Pytorch_CUB11', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('cars', os.path.join(self.dataset_root, 'Pytorch_Cars_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('aircraft', os.path.join(self.dataset_root, 'Pytorch_AirCraft_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('actions', os.path.join(self.dataset_root, 'Pytorch_Actions_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('letters', os.path.join(self.dataset_root, 'Pytorch_Letters_dataset', dataset_filename)))
        self.ordered_tasks.append(
            TaskDataset('svhn', os.path.join(self.dataset_root, 'Pytorch_SVHN_dataset', dataset_filename)))

        # Init classes per task
        for task in self.ordered_tasks:
            task.init_size_labels(self.classes_per_task)
            print("{} {} {} {} {}".format(str(task.name), len(task.dset_classes), task.dset_sizes['train'],
                                          task.dset_sizes['val'], task.dset_sizes['test'],
                                          ))
        print("[{}] Initialized".format(self.name))

    def get_task_dataset_path(self, task_name=None, rnd_transform=True):
        if task_name is None:  # JOINT
            print("No JOINT dataset defined!")
            return None
        else:
            filename = [task.imagefolder_path for task in self.ordered_tasks if task_name == task.name]
            assert len(filename) == 1

        return os.path.join(self.dataset_root, task_name, filename[0])

    def get_taskname(self, task_index):
        """
        e.g. Translation of 'Task 1' to the actual name of the first task.
        :param task_index:
        :return:
        """
        if task_index < 1 or task_index > self.task_count:
            raise ValueError('[' + self.name + '] TASK INDEX EXCEEDED: idx = ', task_index)
        return self.ordered_tasks[task_index - 1].name

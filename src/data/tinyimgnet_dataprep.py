"""
Download TinyImageNet from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
"""

import os
import torch
import shutil
import subprocess

from torchvision import transforms

import utilities.utils as utils
from data.imgfolder import random_split, ImageFolderTrainVal


def download_dset(path):
    utils.create_dir(path)

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200.zip')):
        subprocess.call(
            "wget -P {} http://cs231n.stanford.edu/tiny-imagenet-200.zip".format(path),
            shell=True)
        print("Succesfully downloaded TinyImgnet dataset.")
    else:
        print("Already downloaded TinyImgnet dataset in {}".format(path))

    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200')):
        subprocess.call(
            "unzip {} -d {}".format(os.path.join(path, 'tiny-imagenet-200.zip'), path),
            shell=True)
        print("Succesfully extracted TinyImgnet dataset.")
    else:
        print("Already extracted TinyImgnet dataset in {}".format(os.path.join(path, 'tiny-imagenet-200')))


def create_training_classes_file(root_path):
    """
    training dir is ImageFolder like structure.
    Gather all classnames in 1 file for later use.
    Ordering may differ from original classes.txt in project!
    :return:
    """
    with open(os.path.join(root_path, 'classes.txt'), 'w') as classes_file:
        for class_dir in utils.get_immediate_subdirectories(os.path.join(root_path, 'train')):
            classes_file.write(class_dir + "\n")


def preprocess_val(root_path):
    """
    Uses val_annotations.txt to construct ImageFolder like structure.
    Images in 'image' folder are moved into class-folder.
    :return:
    """
    val_path = os.path.join(root_path, 'val')
    annotation_path = os.path.join(val_path, 'val_annotations.txt')

    lines = [line.rstrip('\n') for line in open(annotation_path)]
    for line in lines:
        subs = line.split('\t')
        imagename = subs[0]
        dirname = subs[1]
        this_class_dir = os.path.join(val_path, dirname, 'images')
        if not os.path.isdir(this_class_dir):
            os.makedirs(this_class_dir)

        utils.attempt_move(os.path.join(val_path, 'images', imagename), this_class_dir)


def divide_into_tasks(root_path, task_count=10):
    """
    Divides total subset data into task classes (into dirs "task_x").
    :return:
    """
    print("Be patient: dividing into tasks...")
    nb_classes_task = 200 // task_count
    assert 200 % nb_classes_task == 0, "total 200 classes must be divisible by nb classes per task"

    file_path = os.path.join(root_path, "classes.txt")
    lines = [line.rstrip('\n') for line in open(file_path)]
    assert len(lines) == 200, "Should have 200 classes, but {} lines in classes.txt".format(len(lines))
    subsets = ['train', 'val']
    img_paths = {t: {s: [] for s in subsets + ['classes', 'class_to_idx']} for t in range(1, task_count + 1)}

    for subset in subsets:
        task = 1
        for initial_class in (range(0, len(lines), nb_classes_task)):
            classes = lines[initial_class:initial_class + nb_classes_task]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            if len(img_paths[task]['classes']) == 0:
                img_paths[task]['classes'].extend(classes)
            img_paths[task]['class_to_idx'] = class_to_idx

            # Make subset dataset dir for each task
            for class_index in range(initial_class, initial_class + nb_classes_task):
                target = lines[class_index]
                src_path = os.path.join(root_path, subset, target, 'images')
                imgs = [(os.path.join(src_path, f), class_to_idx[target]) for f in os.listdir(src_path)
                        if os.path.isfile(os.path.join(src_path, f))]  # (label_idx, path)
                img_paths[task][subset].extend(imgs)
            task = task + 1
    return img_paths


def create_train_test_val_imagefolders(img_paths, root, normalize, include_rnd_transform, no_crop):
    # TRAIN
    pre_transf = None
    if include_rnd_transform:
        if no_crop:
            pre_transf = transforms.RandomHorizontalFlip()
        else:
            pre_transf = transforms.Compose([
                transforms.RandomResizedCrop(56),  # Crop
                transforms.RandomHorizontalFlip(), ])
    else:  # No rnd transform
        if not no_crop:
            pre_transf = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),  # Crop
            ])
    sufx_transf = [transforms.ToTensor(), normalize, ]
    train_transf = transforms.Compose([pre_transf] + sufx_transf) if pre_transf else transforms.Compose(sufx_transf)
    train_dataset = ImageFolderTrainVal(root, None, transform=train_transf, classes=img_paths['classes'],
                                        class_to_idx=img_paths['class_to_idx'], imgs=img_paths['train'])

    # Validation
    pre_transf_val = None
    sufx_transf_val = [transforms.ToTensor(), normalize, ]
    if not no_crop:
        pre_transf_val = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(56), ])
    val_transf = transforms.Compose([pre_transf_val] + sufx_transf_val) if pre_transf_val \
        else transforms.Compose(sufx_transf_val)
    test_dataset = ImageFolderTrainVal(root, None, transform=val_transf, classes=img_paths['classes'],
                                       class_to_idx=img_paths['class_to_idx'], imgs=img_paths['val'])

    # Validation set of TinyImgnet is used for testing dataset,
    # Training data set is split into train and validation.
    dsets = {}
    dsets['train'] = train_dataset
    dsets['test'] = test_dataset

    # Split original TinyImgnet trainset into our train and val sets
    dset_trainval = random_split(dsets['train'],
                                 [round(len(dsets['train']) * (0.8)), round(len(dsets['train']) * (0.2))])
    dsets['train'] = dset_trainval[0]
    dsets['val'] = dset_trainval[1]
    dsets['val'].transform = val_transf  # Same transform val/test
    print("Created Dataset:{}".format(dsets))
    return dsets


def create_train_val_test_imagefolder_dict(dataset_root, img_paths, task_count, outfile, no_crop=True, transform=False):
    """
    Makes specific wrapper dictionary with the 3 ImageFolder objects we will use for training, validation and evaluation.
    """
    # Data loading code
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop", "{}tasks".format(task_count))
    else:
        out_dir = os.path.join(dataset_root, "{}tasks".format(task_count))

    for task in range(1, task_count + 1):
        print("\nTASK ", task)

        # Tiny Imgnet total values from pytorch
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dsets = create_train_test_val_imagefolders(img_paths[task], dataset_root, normalize, transform, no_crop)
        utils.create_dir(os.path.join(out_dir, str(task)))
        torch.save(dsets, os.path.join(out_dir, str(task), outfile))
        print("SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                        len(dsets['test'])))
        print("Saved dictionary format of train/val/test dataset Imagefolders.")


def create_train_val_test_imagefolder_dict_joint(dataset_root, img_paths, outfile, no_crop=True):
    """
    For JOINT training: All 10 tasks in 1 data folder.
    Makes specific wrapper dictionary with the 3 ImageFolder objects we will use for training, validation and evaluation.
    """
    # Data loading code
    if no_crop:
        out_dir = os.path.join(dataset_root, "no_crop")
    else:
        out_dir = dataset_root

    # Tiny Imgnet total values from pytorch
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dsets = create_train_test_val_imagefolders(img_paths[1], dataset_root, normalize, True, no_crop=no_crop)

    ################ SAVE ##################
    utils.create_dir(out_dir)
    torch.save(dsets, os.path.join(out_dir, outfile))
    print("JOINT SIZES: train={}, val={}, test={}".format(len(dsets['train']), len(dsets['val']),
                                                          len(dsets['test'])))
    print("JOINT: Saved dictionary format of train/val/test dataset Imagefolders.")


def prepare_dataset(dset, target_path, survey_order=True, joint=True, task_count=10, overwrite=False):
    """
    Main datapreparation code for Tiny Imagenet.
    First download the set and set target_path to unzipped download path.
    See README dataprep.

    :param target_path: Path to Tiny Imagenet dataset
    :param survey_order: Use the original survey ordering of the labels to divide in tasks
    :param joint: Prepare the joint dataset
    """
    print("Preparing dataset")
    if not os.path.isdir(target_path):
        raise Exception("TINYIMGNET PATH IS NON EXISTING DIR: ", target_path)

    if os.path.isdir(os.path.join(target_path, 'train')):
        if survey_order:
            shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "tinyimgnet_classes.txt"),
                            os.path.join(target_path, 'classes.txt'))
        else:
            create_training_classes_file(target_path)
    else:
        print("Already cleaned up original train")

    if not os.path.isfile(os.path.join(target_path, 'VAL_PREPROCESS.TOKEN')):
        preprocess_val(target_path)
        torch.save({}, os.path.join(target_path, 'VAL_PREPROCESS.TOKEN'))
    else:
        print("Already cleaned up original val")

    # Make different subset dataset for each task
    if not os.path.isfile(os.path.join(target_path, "DIV.TOKEN")) or overwrite:
        print("PREPARING DATASET: DIVIDING INTO {} TASKS".format(task_count))
        img_paths = divide_into_tasks(target_path, task_count=task_count)
        torch.save({}, os.path.join(target_path, 'DIV.TOKEN'))
    else:
        print("Already divided into tasks")

    if not os.path.isfile(os.path.join(target_path, "IMGFOLDER.TOKEN")) or overwrite:
        print("PREPARING DATASET: IMAGEFOLDER GENERATION")
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.raw_dataset_file,
                                               no_crop=True, transform=False)
        create_train_val_test_imagefolder_dict(target_path, img_paths, task_count, dset.transformed_dataset_file,
                                               no_crop=True, transform=True)
        torch.save({}, os.path.join(target_path, 'IMGFOLDER.TOKEN'))
    else:
        print("Task imgfolders already present.")

    if joint:
        if not os.path.isfile(os.path.join(target_path, "IMGFOLDER_JOINT.TOKEN")) or overwrite:
            print("PREPARING JOINT DATASET: IMAGEFOLDER GENERATION")
            img_paths = divide_into_tasks(target_path, task_count=1)
            # Create joint
            create_train_val_test_imagefolder_dict_joint(target_path, img_paths, dset.joint_dataset_file, no_crop=True)
            torch.save({}, os.path.join(target_path, 'IMGFOLDER_JOINT.TOKEN'))
        else:
            print("Joint imgfolders already present.")

    print("PREPARED DATASET")

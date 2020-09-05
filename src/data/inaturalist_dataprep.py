"""
Script to set up the iNaturalist dataset.
Info: https://www.kaggle.com/c/inaturalist-2018/data
Download: https://github.com/visipedia/inat_comp/tree/master/2018

Images have a max dimension of 800px and have been converted to JPEG format
Untaring the images creates a directory structure like train_val2018/super category/category/image.jpg. This may take a while.
10 Super categories are selected from the 14 available, based on at least having 100 categories (leaving out Chromista,
Protozoa, Bacteria), and omitting a random super category from the remainder (Actinopterygii).
"""

import os
import torch
import subprocess
from torchvision import transforms
import tqdm

import utilities.utils as utils
from data.imgfolder import ConcatDatasetDynamicLabels, ImageFolderTrainVal


#####################
# DOWNLOAD
#####################
def download_dset(path, location="eu"):
    """
    Europe links are used, replace if Asia or North America.
    Location: eu (Europe), asia,
    """
    assert location in ["eu", "asia", "us"]
    utils.create_dir(path)

    # TRAIN/VAL IMAGES
    train_link = "https://storage.googleapis.com/inat_data_2018_{}/train_val2018.tar.gz".format(location)
    train_tarname = train_link.split('/')[-1]  # train_val2018.tar.gz
    train_dirname = train_tarname.split('.')[0]  # train_val2018

    if not os.path.exists(os.path.join(path, train_tarname)):
        download(path, train_link)
        print("Succesfully downloaded train+val dataset iNaturalist.")
    if not os.path.exists(os.path.join(path, train_dirname)):
        extract(path, os.path.join(path, train_tarname))
        print("Succesfully extracted train+val dataset iNaturalist.")

    # TRAIN JSON
    trainjson_link = "https://storage.googleapis.com/inat_data_2018_{}/train2018.json.tar.gz".format(location)
    trainjson_tarname = trainjson_link.split('/')[-1]  # train2018.json.tar.gz
    trainjson_filename = trainjson_link.split('.')[0]  # train2018

    if not os.path.exists(os.path.join(path, trainjson_tarname)):
        download(path, train_link)
        print("Succesfully downloaded train json iNaturalist.")
    if not os.path.exists(os.path.join(path, trainjson_filename)):
        extract(path, os.path.join(path, os.path.join(path, trainjson_tarname)))
        print("Succesfully extracted train json iNaturalist.")

    # VAL JSON
    trainjson_link = "https://storage.googleapis.com/inat_data_2018_{}/train2018.json.tar.gz".format(location)
    trainjson_tarname = trainjson_link.split('/')[-1]  # train2018.json.tar.gz
    trainjson_filename = trainjson_link.split('.')[0]  # train2018

    if not os.path.exists(os.path.join(path, trainjson_tarname)):
        download(path, train_link)
        print("Succesfully downloaded train json iNaturalist.")
    if not os.path.exists(os.path.join(path, trainjson_filename)):
        extract(path, os.path.join(path, os.path.join(path, trainjson_tarname)))
        print("Succesfully extracted train json iNaturalist.")


def download(dest_path, link):
    subprocess.call(
        "wget -P {} {}".format(dest_path, link),
        shell=True)


def extract(dest_path, src_targz_path):
    subprocess.call(
        "tar -C {} -xzvf {}".format(dest_path, src_targz_path),
        shell=True)


#####################
# PREPARE DATASET
#####################
def prepare_inat_trainval(root_path, inat_v="train_val2018", outfile=None, rnd_transform=False, overwrite=False):
    """
    Divide the iNaturalist training/validation dataset into our custom train/val/test split.
    Using fixed 224 crop for AlexNet.
    """
    token_path = os.path.join(root_path, inat_v, "SUCCES_rnd={}.TOKEN".format(rnd_transform))

    if os.path.exists(token_path) and not overwrite:
        print("Skipping creation, already exists: {}".format(token_path))
        return

    print("Preparing iNaturalist {} dataset (rnd={})".format(inat_v, rnd_transform))
    if rnd_transform:
        data_transforms = get_rnd_transforms()
        outfilename = 'imgfolder_trainvaltest_rndtrans.pth.tar' if outfile is None else outfile
    else:
        data_transforms = get_transforms()
        outfilename = 'imgfolder_trainvaltest.pth.tar' if outfile is None else outfile

    task_dirs = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae',
                 'Reptilia']
    task_classes = get_task_classes(root_path, task_dirs)  # Classes for each task
    make_split_files(root_path, task_dirs, task_classes)  # Write splits in files
    tasks_data_list_dir = os.path.join(root_path, inat_v)
    for task_dir in tqdm.tqdm(task_dirs, "Making imagefolders for train/val/test"):
        data_dir = os.path.join(root_path, inat_v, task_dir)
        out_path = os.path.join(root_path, inat_v, task_dir, outfilename)

        train_list = os.path.join(tasks_data_list_dir, task_dir, 'TrainImages.txt')
        val_list = os.path.join(tasks_data_list_dir, task_dir, 'ValImages.txt')
        test_list = os.path.join(tasks_data_list_dir, task_dir, 'TestImages.txt')

        classes_names = task_classes[task_dir]['classes_names']
        all_files_list = [train_list, val_list, test_list]
        splits = ['train', 'val', 'test']
        dsets = {split: ImageFolderTrainVal(data_dir, all_files_list[splits.index(split)], data_transforms[split],
                                            classes=classes_names) for split in splits}
        torch.save(dsets, out_path)
    torch.save({}, token_path)
    print("Finished iNaturalist {} dataset (rnd={})".format(inat_v, rnd_transform))


def get_task_classes(root_path, task_dirs, inat_v="train_val2018"):
    """ Get classes per super category.
    Only consider classes with minimum 100 images."""
    task_classes = {}
    for task_dir in task_dirs:
        full_task_dir = os.path.join(root_path, inat_v, task_dir)
        task_nb_classes = 0
        train_tasks_classes_names = []
        train_tasks_classes_images = []
        for sub_dir in os.walk(full_task_dir):
            sub_dir = sub_dir[0]
            if not sub_dir == full_task_dir:
                _, _, files = os.walk((sub_dir)).__next__()
                file_count = len(files)
                if file_count >= 100:
                    train_tasks_classes_names.append(sub_dir.split(os.sep)[-1])
                    train_tasks_classes_images.append(file_count)
                    task_nb_classes += 1
        task_classes[task_dir] = {}
        task_classes[task_dir]['classes_names'] = train_tasks_classes_names
        task_classes[task_dir]['classes_images'] = train_tasks_classes_images
        task_classes[task_dir]['nb_classes'] = task_nb_classes
        print("TASK {} has {} classes.".format(task_dir, task_nb_classes))
    return task_classes


def make_split_files(root_path, task_dirs, task_classes, max_number_of_files=500, inat_v="train_val2018"):
    """
    Write split files dividing into train/val/test sets.
    Constrain classes to select a maximum of images: max_number_of_files.
    """

    for task_dir in tqdm.tqdm(task_dirs, "Making train/val/test splits"):
        print("Making train/val/test split for task {}".format(task_dir))
        full_task_dir = os.path.join(root_path, inat_v, task_dir)
        try:
            os.makedirs(full_task_dir)
        except:
            pass
        train_path = os.path.join(root_path, inat_v, task_dir, 'TrainImages.txt')
        val_path = os.path.join(root_path, inat_v, task_dir, 'ValImages.txt')
        test_path = os.path.join(root_path, inat_v, task_dir, 'TestImages.txt')

        # Write splits to files
        with open(train_path, "w") as train_file, open(val_path, 'w') as val_file, open(test_path, 'w') as test_file:
            for index, class_name in enumerate(task_classes[task_dir]['classes_names']):
                nb_images = task_classes[task_dir]['classes_images'][index]
                effective_nb_images = min(max_number_of_files, nb_images)
                train_size = int(round(effective_nb_images * 70 / 100))
                val_size = int(round(effective_nb_images * 10 / 100))
                test_size = int(round(effective_nb_images * 20 / 100))
                if train_size + val_size + test_size != effective_nb_images:
                    val_size -= 1
                this_class_path = os.path.join(root_path, inat_v, task_dir, class_name)
                _, _, images = os.walk(this_class_path).__next__()
                if len(images) != nb_images:
                    raise Exception('Number of images different from what we counted')

                # Write to split files
                for im_index in range(train_size):
                    file_name = images[im_index]
                    train_file.write(os.path.join(class_name, file_name) + '\n')
                for im_index in range(val_size):
                    file_name = images[train_size + im_index]
                    val_file.write(os.path.join(class_name, file_name) + '\n')
                for im_index in range(test_size):
                    file_name = images[train_size + val_size + im_index]
                    test_file.write(os.path.join(class_name, file_name) + '\n')


def prepare_JOINT_dataset(root_path, inat_v="train_val2018", outfile=None, taskfile=None, overwrite=False):
    print("Preparing JOINT iNaturalist {} dataset (rnd={})".format(inat_v, True))
    token_path = os.path.join(root_path, inat_v, "SUCCES_JOINT.TOKEN")

    if os.path.exists(token_path) and not overwrite:
        print("Skipping creation, already exists: {}".format(token_path))
        return

    dataset_parent_dir = os.path.join(root_path, inat_v)
    taskfile = 'imgfolder_trainvaltest_rndtrans.pth.tar' if taskfile is None else taskfile
    outfile = 'imgfolder_joint.pth.tar' if outfile is None else outfile
    tasks_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca',
                   'Plantae', 'Reptilia']
    joindsets = {'train': [], 'val': [], 'test': []}
    all_classes = {'train': [], 'val': [], 'test': []}
    for task_name in tasks_names:
        dataset_path = os.path.join(dataset_parent_dir, task_name, taskfile)
        dsets = torch.load(dataset_path)
        for phase in ['train', 'val', 'test']:
            joindsets[phase].append(dsets[phase])
            all_classes[phase] += dsets[phase].classes
    inaturalist_dset = {}
    for phase in ['train', 'val', 'test']:
        classes_len = [len(joindsets[phase][x].classes) for x in
                       list(range(len(joindsets[phase])))]  # number of classes in each dataset
        inaturalist_dset[phase] = ConcatDatasetDynamicLabels(joindsets[phase], classes_len)
        inaturalist_dset[phase].classes = all_classes[phase]
    torch.save(inaturalist_dset, os.path.join(dataset_parent_dir, outfile))
    torch.save({}, token_path)
    print("Finished JOINT iNaturalist {} dataset (rnd={})".format(inat_v, True))


#####################
# TRANSFORMS
#####################
def get_rnd_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


if __name__ == "__main__":
    config = utils.get_parsed_config()
    parent_path = utils.read_from_config(config, 'ds_root_path')
    root_path = os.path.join(parent_path, "inaturalist")

    # Download the training/validation dataset of iNaturalist
    download_dset(root_path)

    # Divide it into our own training/validation/test splits
    prepare_inat_trainval(root_path, rnd_transform=False)
    prepare_inat_trainval(root_path, rnd_transform=True)
    prepare_JOINT_dataset(root_path)

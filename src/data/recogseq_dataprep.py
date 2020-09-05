"""
For preparation of the 8 task datasets. Each of the dataset paths must be configured.
Imagefolders are created from the datasets, division between train/test is in 2 txt files.
"""

import random

from torchvision import transforms
from data.imgfolder import *


def split_file(file_to_split, out_val, out_test, percentage=0.9, isShuffle=True, seed=123):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    print("START SPLIT")

    random.seed(seed)
    with open(file_to_split, 'r', encoding="utf-8") as fin, \
            open(out_val, 'w') as foutBig, \
            open(out_test, 'w') as foutSmall:
        nLines = sum(1 for line in fin)
        fin.seek(0)

        nValid = int(nLines * percentage)
        nTest = nLines - nValid

        val_lines = 0
        for line in fin:
            r = random.random() if isShuffle else 0  # so that always evaluated to true when not isShuffle
            if (val_lines < nValid and r < percentage) or (nLines - val_lines > nTest):
                foutBig.write(line)
                val_lines += 1
            else:
                foutSmall.write(line)
    print("DONE SPLIT")


def prepare_dataset(exp_root, data_root, dataset_filename, ds_dir, imgdir='images', overwrite=False,
                    data_transforms=None, classes=None):
    """
    Output: splits into OrigTestImagesPartialForVal.txt and OrigTestImagesPartialForTest.txt of original test split.
            Saves imgfolder to exp_file.
    """
    data_dir = os.path.join(data_root, ds_dir, imgdir)
    train_list = os.path.join(data_root, ds_dir, 'TrainImages.txt')
    test_list = os.path.join(data_root, ds_dir, 'TestImages.txt')
    exp_file = os.path.join(exp_root, ds_dir, dataset_filename)  # Output
    name = ds_dir

    if os.path.exists(exp_file) and not overwrite:
        print("Imagefolder already exists: {}".format(exp_file))
        return exp_file

    data_transforms = data_transforms if data_transforms else {
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
        ]),
    }

    out_val = os.path.join(os.path.dirname(exp_file), 'OrigTestImagesPartialForVal.txt')
    out_test = os.path.join(os.path.dirname(exp_file), 'OrigTestImagesPartialForTest.txt')
    print("out_val=", out_val)
    print("out_test=", out_test)

    if not os.path.exists(out_val) or not os.path.exists(out_test) or overwrite:
        print("SPLITTING TEST INTO: val and test")
        split_file(test_list, out_val=out_val, out_test=out_test, percentage=0.5)

    all_files_list = [train_list, out_val, out_test]
    modes = ['train', 'val', 'test']

    print("Creating imgfolders")
    dsets = {modes[x]: ImageFolderTrainVal(data_dir, all_files_list[x], data_transforms[modes[x]], classes=classes)
             for x in range(0, len(modes))}

    # Check
    print("Checking sizes")
    dset_sizes = {x: len(dsets[x]) for x in modes}
    dset_classes = dsets['train'].classes
    print("Task {}: dset_sizes = {}, #classes = {}".format(name, dset_sizes, len(dset_classes)))

    torch.save(dsets, exp_file)
    print("SAVED TO: ", exp_file)
    return exp_file


def main():
    exp_root = '/path/to/datasets/object_recog_8task_seq'
    dataset_filename = 'dataset.pth.tar'  # Output imgfolder size

    # Download: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_Flowers', imgdir='Images')

    # Download: http://web.mit.edu/torralba/www/indoor.html
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_Scenes', imgdir='Images')

    # Download: http://www.vision.caltech.edu/visipedia
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_CUB11', imgdir='CUB11f_dataset/images/images')

    # Download: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_Cars_dataset')

    # Download: http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_AirCraft_dataset')

    # Download: http://host.robots.ox.ac.uk/pascal/VOC/
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_Actions_dataset')

    # Download: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_Letters_dataset')

    # Download: http://ufldl.stanford.edu/housenumbers/ (Or use Pytorch build-in dataloader)
    prepare_dataset(exp_root, exp_root, dataset_filename, 'Pytorch_SVHN_dataset',
                    classes=[str(i) for i in range(1, 11)])
    print("prepared all datasets")


if __name__ == "__main__":
    main()

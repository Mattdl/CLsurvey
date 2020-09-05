import bisect
import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data
from torchvision import datasets

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []
    # print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname
                    # print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    return images


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainVal(datasets.ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None, class_to_idx=None, imgs=None):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("Creating Imgfolder with root: {}".format(root))
        imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                                format(root, ",".join(IMG_EXTENSIONS))))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


class ImageFolder_Subset(ImageFolderTrainVal):
    """
    Wrapper of ImageFolderTrainVal, subsetting based on indices.
    """

    def __init__(self, dataset, indices):
        self.__dict__ = copy.deepcopy(dataset).__dict__
        self.indices = indices  # Extra

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])  # Only return from subset

    def __len__(self):
        return len(self.indices)


class ImageFolder_Subset_ClassIncremental(ImageFolder_Subset):
    """
    ClassIncremental to only choose samples of specific label.
    Need to subclass in order to retain compatibility with saved ImageFolder_Subset objects.
    (Can't add new attributes...)
    """

    def __init__(self, imgfolder_subset, target_idx):
        """
        Subsets an ImageFolder_Subset object for only the target idx.
        :param imgfolder_subset: ImageFolder_Subset object
        :param target_idx: target int output idx
        """
        if not isinstance(imgfolder_subset, ImageFolder_Subset):
            print("Not a subset={}".format(imgfolder_subset))
            imagefolder_subset = random_split(imgfolder_subset, [len(imgfolder_subset)])[0]
            print("A subset={}".format(imagefolder_subset))

        # Creation of this object shouldn't interfere with original object
        imgfolder_subset = copy.deepcopy(imgfolder_subset)

        # Change ds classes here, to avoid any misuse
        imgfolder_subset.class_to_idx = {label: idx for label, idx in imgfolder_subset.class_to_idx.items()
                                         if idx == target_idx}
        assert len(imgfolder_subset.class_to_idx) == 1
        imgfolder_subset.classes = next(iter(imgfolder_subset.class_to_idx))

        # (path, FC_idx) => from (path, class_to_idx[class]) pairs
        orig_samples = np.asarray(imgfolder_subset.samples)
        subset_samples = orig_samples[imgfolder_subset.indices.numpy()]
        print("SUBSETTING 1 CLASS FROM DSET WITH SIZE: ", subset_samples.shape[0])

        # Filter these samples to only those with certain label
        label_idxs = np.where(subset_samples[:, 1] == str(target_idx))[0]  # indices row
        print("#SAMPLES WITH LABEL {}: {}".format(target_idx, label_idxs.shape[0]))

        # Filter the corresponding indices
        final_indices = imgfolder_subset.indices[label_idxs]

        # Sanity check
        # is first label equal to all others
        is_all_same_label = str(target_idx) == orig_samples[final_indices, 1]
        assert np.all(is_all_same_label)

        # Make a ImageFolder of the whole
        super().__init__(imgfolder_subset, final_indices)


class ImageFolder_Subset_PathRetriever(ImageFolder_Subset):
    """
    Wrapper for Imagefolder_Subset: Also returns path of the images.
    """

    def __init__(self, imagefolder_subset):
        if not isinstance(imagefolder_subset, ImageFolder_Subset):
            print("Transforming into Subset Wrapper={}".format(imagefolder_subset))
            imagefolder_subset = random_split(imagefolder_subset, [len(imagefolder_subset)])[0]
        super().__init__(imagefolder_subset, imagefolder_subset.indices)

    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolder_Subset_PathRetriever, self).__getitem__(index)
        # the image file path
        path = self.samples[self.indices[index]][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


class ImagePathlist(data.Dataset):
    """
    Adapted from: https://github.com/pytorch/vision/issues/81
    Load images from a list with paths (no labels).
    """

    def __init__(self, imlist, targetlist=None, root='', transform=None, loader=default_loader):
        self.imlist = imlist
        self.targetlist = targetlist
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]

        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        if self.targetlist is not None:
            target = self.targetlist[index]
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.imlist)


def random_split(dataset, lengths):
    """
    Creates ImageFolder_Subset subsets from the dataset, by altering the indices.
    :param dataset:
    :param lengths:
    :return: array of ImageFolder_Subset objects
    """
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(sum(lengths))
    return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(accumulate(lengths), lengths)]


class ConcatDatasetDynamicLabels(torch.utils.data.ConcatDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
        the output labels are shifted by the dataset index which differs from the pytorch implementation that return the original labels
    """

    def __init__(self, datasets, classes_len):
        """
        :param datasets: List of Imagefolders
        :param classes_len: List of class lengths for each imagefolder
        """
        super(ConcatDatasetDynamicLabels, self).__init__(datasets)
        self.cumulative_classes_len = list(accumulate(classes_len))

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            img, label = self.datasets[dataset_idx][sample_idx]
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            img, label = self.datasets[dataset_idx][sample_idx]
            label = label + self.cumulative_classes_len[dataset_idx - 1]  # Shift Labels
        return img, label

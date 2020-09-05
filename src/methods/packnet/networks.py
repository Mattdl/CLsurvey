"""Contains various network definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class ModifiedWrapperModel(nn.Module):
    """VGGSlim with different classifiers."""

    def __init__(self, raw_model, classifier_last_layer_idx, input_size):
        super(ModifiedWrapperModel, self).__init__()
        self.make_model(raw_model, classifier_last_layer_idx)
        self.input_size = input_size

    def make_model(self, raw_model, classifier_last_layer_idx, is_pretrained=False):
        """Creates the model."""
        # Get the pretrained model.
        self.datasets, self.classifiers = [], nn.ModuleList()  # Classifiers = stack of heads for diff tasks
        self.last_layer_in_feats = None

        # For classifier without dropout
        classifier_input_size = None
        prunable_post_feat_extractor = []
        start_idx = 0
        found_classifier_input = False
        for idx, module in enumerate(raw_model.classifier.children()):
            if not found_classifier_input:
                if idx == start_idx and isinstance(module, nn.Linear):
                    classifier_input_size = module.in_features
                    found_classifier_input = True
                else:
                    start_idx += 1

            if idx != classifier_last_layer_idx:
                prunable_post_feat_extractor.append(module)
            else:
                # Add task_dataset and corresponding head to list of first (pretrain) task
                if isinstance(module, nn.Linear):
                    self.last_layer_in_feats = module.in_features

                    if is_pretrained:
                        self.datasets.append('task_1')
                        self.classifiers.append(module)
                else:
                    raise Exception("Defined hardcoded last layer idx is not Linear")

        assert isinstance(classifier_input_size, int)
        assert isinstance(self.last_layer_in_feats, int)

        # Features are all prunable: So add all but last layer from classifier
        features = list(raw_model.features.children())
        features.append(View(-1, classifier_input_size))
        features.extend(prunable_post_feat_extractor)

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)
        print("MODEL WRAPPER FEAT EXTRACTOR = ", self.shared)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(self.last_layer_in_feats, num_outputs))
            print("ADDED HEAD for dataset '{}': {}".format(dataset, self.classifiers[-1]))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedWrapperModel, self).train(mode)

    def check_correctness(self, raw_model):
        """Ensures that conversion of fc layers to conv is correct."""

        # Test to make sure outputs match.
        raw_model.eval()
        self.shared.eval()
        self.classifier.eval()
        rand_input = Variable(torch.rand(1, 3, self.input_size, self.input_size))
        fc_output = raw_model(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier(x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).item() < 1e-8
        print('CHECK PASSED')
        # raw_input()


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ModifiedVGG16(nn.Module):
    """VGG16 with different classifiers."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16 = models.vgg16(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(4096, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        # raw_input()


class ModifiedVGG16BN(ModifiedVGG16):
    """VGG16 with batch norm."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16BN, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16BN, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Get classifiers.
        idx = 6
        for module in vgg16_bn.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1

        features = list(vgg16_bn.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)


class ModifiedResNet(ModifiedVGG16):
    """ResNet-50."""

    def __init__(self, make_model=True):
        super(ModifiedResNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        resnet = models.resnet50(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(resnet.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(2048, num_outputs))


class ModifiedDenseNet(ModifiedVGG16):
    """DenseNet-121."""

    def __init__(self, make_model=True):
        super(ModifiedDenseNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedDenseNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        densenet = models.densenet121(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = densenet.features

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(densenet.classifier)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def forward(self, x):
        features = self.shared(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(1024, num_outputs))

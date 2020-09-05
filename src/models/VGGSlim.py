"""
See: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import torch.nn as nn
import torch
import torchvision

#############################
# Static params: Config
#############################
conv_kernel_size = 3
img_input_channels = 3
cfg = {
    '19normal': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    '16normal': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '11normal': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    # models TinyImgnet
    'small_VGG9': [64, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M'],  # 334,016 feat params,
    'base_VGG9': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M'],  # 1.145.408 feat params
    'wide_VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],  # 4.500.864 feat params
    'deep_VGG22': [64, 'M', 64, 64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 128, 'M',
                   256, 256, 256, 256, 256, 256, 'M'],  # 4.280.704 feat params
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = img_input_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_kernel_size, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGSlim(torchvision.models.VGG):
    """
    Creates VGG feature extractor from config and custom classifier.
    """

    def __init__(self, config='11Slim', num_classes=50, init_weights=True,
                 classifier_inputdim=512 * 7 * 7, classifier_dim1=512, classifier_dim2=512, batch_norm=False,
                 dropout=False):
        features = make_layers(cfg[config], batch_norm=batch_norm)
        super(VGGSlim, self).__init__(features)

        if hasattr(self, 'avgpool'):  # Compat Pytorch>1.0.0
            self.avgpool = torch.nn.Identity()

        if dropout:  # Same as in Pytorch default: print(models.vgg11_bn())
            self.classifier = nn.Sequential(
                nn.Linear(classifier_inputdim, classifier_dim1),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_dim1, classifier_dim2),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_dim2, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_inputdim, classifier_dim1),
                nn.ReLU(True),
                nn.Linear(classifier_dim1, classifier_dim2),
                nn.ReLU(True),
                nn.Linear(classifier_dim2, num_classes),
            )
        if init_weights:
            self._initialize_weights()

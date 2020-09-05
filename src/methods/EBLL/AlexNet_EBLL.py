"""
AlexNet wrapper to use EBLL.
"""

import torch
import torch.nn as nn


class AutoEncoder(torch.nn.Module):
    def __init__(self, x_dim, h1_dim):
        super(AutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            # fc layers for the encoder
            nn.Linear(x_dim, h1_dim),
            nn.Sigmoid())

        # fc layers for the decoder
        self.decode = nn.Sequential(
            nn.Linear(h1_dim, x_dim),

        )

    def forward(self, x):
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon


class autoencoders(torch.nn.Module):
    def __init__(self, autoencoder):
        super(autoencoders, self).__init__()
        self.add_module('0', autoencoder.encode)

    def forward(self, x):
        outputs = []

        for name, module in self._modules.items():
            outputs.append(module(x))

        return outputs


class AlexNet_ENCODER(nn.Module):
    """
    EBLL ENCODER, AlexNet Defaults
    """

    def __init__(self, alexnet, dim=100, last_layer_name=6, num_ftrs=256 * 6 * 6):
        """
        Net with autoencoder inserted between feature extractor and classifier.

        :param dim: Reduced code dim in autoencoder
        :param last_layer_name: int
        :param num_ftrs: Autoencoder input/output size
        """
        super(AlexNet_ENCODER, self).__init__()

        self.add_module('features', alexnet.features)
        # replace it with the number of features
        self.add_module('autoencoder', AutoEncoder(num_ftrs, dim))
        self.add_module('classifier', alexnet.classifier)
        self.last_layer_name = last_layer_name

    def forward(self, x):

        sub_index = 0

        last_layer = False
        for name, module in self._modules.items():

            for namex, modulex in module._modules.items():
                if last_layer:
                    out = modulex(x)
                else:
                    x = modulex(x)
                if name == 'classifier' and namex == str(int(self.last_layer_name) - 1):
                    last_layer = True
            if name == 'autoencoder':
                encoder_output = x

            # for reshaping the fully connected layers
            # need to be changed for
            if sub_index == 0:
                x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
                encoder_input = x.data.clone()
                # forward to the encoder
            sub_index += 1
        return out, encoder_input, encoder_output


class AlexNet_EBLL(nn.Module):
    """
    EBLL MODEL WRAPPER, AlexNet Defaults
    """

    def __init__(self, model, autoencoder, last_layer_name=6):
        super(AlexNet_EBLL, self).__init__()
        self.add_module('features', model.features)
        self.add_module('autoencoders', autoencoders(autoencoder))
        self.add_module('classifier', model.classifier)
        self.last_layer_name = last_layer_name
        self.finetune_mode = False

    def set_finetune_mode(self, mode):
        self.finetune_mode = mode

    def forward(self, x):

        sub_index = 0
        last_layer = False
        for name, module in self._modules.items():
            if name == 'autoencoders':
                codes = module(x)
            else:
                for namex, modulex in module._modules.items():

                    if last_layer:
                        outputs.append(modulex(x))
                    else:
                        x = modulex(x)

                    if name == 'classifier' and namex == str(self.last_layer_name - 1):
                        last_layer = True
                        outputs = []

            # for reshaping the fully connected layers
            # need to be changed for
            if sub_index == 0:
                x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
            sub_index += 1

        # In this case only return for one head an output
        if self.finetune_mode:
            # print("FINETUNING MODE, so outputting single value")
            assert len(outputs) == 1
            outputs = outputs[0]

        return outputs, codes

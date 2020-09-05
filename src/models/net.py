import os
from abc import ABCMeta, abstractmethod

import torch
from torchvision import models

import models.VGGSlim as VGGcreator
import utilities.utils


########################################
# PARSING
########################################

def parse_model_name(models_root_path, model_name, input_size):
    """
    Parses model name into model type object.
    :param model_name: e.g. small_VGG9_cl_512_512
    :param input_size: Size of the input: (w , h)
    :return: the actual model type (not pytorch model)
    """
    pretrained = "pretrained" in model_name
    if "alexnet" in model_name:
        base_model = AlexNet(models_root_path, pretrained=pretrained, create=True)
    elif SmallVGG9.vgg_config in model_name:
        base_model = SmallVGG9(models_root_path, input_size, model_name, create=True)
    elif WideVGG9.vgg_config in model_name:
        base_model = WideVGG9(models_root_path, input_size, model_name, create=True)
    elif DeepVGG22.vgg_config in model_name:
        base_model = DeepVGG22(models_root_path, input_size, model_name, create=True)
    elif BaseVGG9.vgg_config in model_name:
        base_model = BaseVGG9(models_root_path, input_size, model_name, create=True)
    else:
        raise NotImplementedError("MODEL NOT IMPLEMENTED YET: ", model_name)

    return base_model


def get_init_modelname(args):
    """
    The model_name of the first-task model in SI.
    Needs different 1st task model if using regularization: e.g. L2, dropout, BN, dropout+BN
    """
    name = ["e={}".format(args.num_epochs),
            "bs={}".format(args.batch_size),
            "lr={}".format(sorted(args.lr_grid))]
    if args.weight_decay != 0:
        name.append("{}={}".format(ModelRegularization.weight_decay, args.weight_decay))
    if ModelRegularization.batchnorm in args.model_name:
        name.append(ModelRegularization.batchnorm)
    if ModelRegularization.dropout in args.model_name:
        name.append(ModelRegularization.dropout)
    return '_'.join(name)


def extract_modelname_val(seg, tr_exp_dir):
    seg_found = [tr_seg.split('=')[-1] for tr_seg in tr_exp_dir.split('_') if seg == tr_seg.split('=')[0]]
    if len(seg_found) == 1:
        return seg_found[0]
    elif len(seg_found) > 1:
        raise Exception("Ambiguity in exp name: {}".format(seg_found))
    else:
        return None


class ModelRegularization(object):
    vanilla = 'vanilla'
    weight_decay = 'L2'
    dropout = 'DROP'
    batchnorm = 'BN'


########################################
# MODELS
########################################

class Model(metaclass=ABCMeta):
    @property
    @abstractmethod
    def last_layer_idx(self):
        """ Used in data-based methods LWF/EBLL to know where heads start."""
        pass

    @abstractmethod
    def name(self): pass

    @abstractmethod
    def path(self): pass


############################################################
############################################################
# AlexNet
############################################################
############################################################
class AlexNet(Model):
    last_layer_idx = 6

    def __init__(self, models_root_path, pretrained=True, create=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR ALEXNET DOES NOT EXIST: ", models_root_path)

        name = ["alexnet"]
        if pretrained:
            name.append("pretrained_imgnet")
        else:
            name.append("scratch")
        self.name = '_'.join(name)
        self.path = os.path.join(models_root_path,
                                 self.name + ".pth.tar")  # In training scripts: AlexNet pretrained on Imgnet when empty

        if not os.path.exists(self.path):
            if create:
                torch.save(models.alexnet(pretrained=pretrained), self.path)
                print("SAVED NEW ALEXNET MODEL (name=", self.name, ") to ", self.path)
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("STARTING FROM EXISTING ALEXNET MODEL (name=", self.name, ") to ", self.path)

    def name(self):
        return self.name

    def path(self):
        return self.path


############################################################
############################################################
# VGG MODELS
############################################################
############################################################
class VGGModel(Model):
    """
    VGG based models.
    base_vgg9_cl_512_512_DROP_BN
    """
    last_layer_idx = 4  # vgg_classifier_last_layer_idx
    pooling_layers = 4  # in all our models 4 max pooling layers with stride 2

    def __init__(self, models_root_path, input_size, model_name, vgg_config, overwrite_mode=False, create=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR ", model_name, " DOES NOT EXIST: ", models_root_path)

        self.name = model_name
        self.final_featmap_count = VGGcreator.cfg[vgg_config][-2]
        parent_path = os.path.join(models_root_path,
                                   "customVGG_input={}x{}".format(str(input_size[0]), str(input_size[1])))
        self.path = os.path.join(parent_path, self.name + ".pth.tar")

        # After classifier name
        dropout = ModelRegularization.dropout in model_name.split("_")
        batch_norm = ModelRegularization.batchnorm in model_name.split("_")

        if dropout:
            self.last_layer_idx = 6

        if overwrite_mode or not os.path.exists(self.path):
            classifier = parse_classifier_name(model_name)

            last_featmap_size = (
                int(input_size[0] / 2 ** self.pooling_layers), int(input_size[1] / 2 ** self.pooling_layers))
            print("CREATING MODEL, with FC classifier size {}*{}*{}".format(self.final_featmap_count,
                                                                            last_featmap_size[0],
                                                                            last_featmap_size[1]))
            if create:
                utilities.utils.create_dir(parent_path)
                make_VGGmodel(last_featmap_size, vgg_config, self.path, classifier, self.final_featmap_count,
                              batch_norm, dropout)
                print("CREATED MODEL:")
                print(view_saved_model(self.path))
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("MODEL ", model_name, " already exist in path = ", self.path)

    def name(self):
        return self.name

    def path(self):
        return self.path


class SmallVGG9(VGGModel):
    vgg_config = "small_VGG9"
    def_classifier_suffix = "_cl_128_128"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. small_VGG9_cl_128_128
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


class BaseVGG9(VGGModel):
    vgg_config = "base_VGG9"
    def_classifier_suffix = "_cl_512_512"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. base_VGG9_cl_512_512
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


class WideVGG9(VGGModel):
    vgg_config = "wide_VGG9"
    def_classifier_suffix = "_cl_512_512"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. base_vgg9_cl_512_512
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


class DeepVGG22(VGGModel):
    vgg_config = "deep_VGG22"
    def_classifier_suffix = "_cl_512_512"

    def __init__(self, models_root_path, input_size, model_name=(vgg_config + def_classifier_suffix),
                 overwrite_mode=False, create=False):
        """
        :param model_name: defined in main script, e.g. base_vgg9_cl_512_512
        :param overwrite_mode: Overwrite if model already exists
        """
        super().__init__(models_root_path, input_size, model_name, vgg_config=self.vgg_config,
                         overwrite_mode=overwrite_mode, create=create)


############################################################
# FUNCTIONS
############################################################
def make_VGGmodel(last_featmap_size, name, path, classifier, final_featmap_count, batch_norm, dropout):
    """
    Creates custom VGG model with specified classifier array.

    :param last_featmap_size: (w , h ) tupple showing last feature map size.
    :param name: custom VGG config name for feature extraction
    :param path:
    :param classifier: array of length 2, with sizes of 2 FC layers
    :param final_featmap_count: amount of feat maps in the last non-pooling layer. Used to calc classifier input.
    :return:
    """
    # Create and save the model in data root path
    model = VGGcreator.VGGSlim(config=name, num_classes=20,
                               classifier_inputdim=final_featmap_count * last_featmap_size[0] * last_featmap_size[1],
                               classifier_dim1=int(classifier[0]),
                               classifier_dim2=int(classifier[1]),
                               batch_norm=batch_norm,
                               dropout=dropout)
    torch.save(model, path)
    print("SAVED NEW MODEL (name=", name, ", classifier=", classifier, ") to ", path)


def parse_classifier_name(model_name, classifier_layers=3):
    """
    Takes in model name (e.g. base_vgg9_cl_512_512_BN), and returns classifier sizes: [512,512]
    :param model_name:
    :return:
    """
    return model_name[model_name.index("cl_"):].split("_")[1:classifier_layers]


def get_vgg_classifier_postfix(classifier):
    return "_cl_" + '_'.join(str(classifier))


def save_model_to_path(self, model):
    torch.save(model, self.path)


def print_module_composition(vgg_config_name):
    """
    Prints the amount of weights and biase parameters in the feat extractor.
    Formatted in a per module basis.
    :param vgg_config_name:
    :return:
    """
    vgg_config = VGGcreator.cfg[vgg_config_name]

    # Print Weights
    weight_str = []
    weight_str.append("(" + str(VGGcreator.conv_kernel_size) + "*" + str(VGGcreator.conv_kernel_size) + ") * {(")
    bias_str = []

    weightlist = vgg_config
    weightlist.insert(0, VGGcreator.img_input_channels)

    for idx in range(1, len(weightlist)):
        if 'M' == weightlist[idx]:
            weight_str.append(")")
            if idx != len(weightlist) - 1:
                weight_str.append(" + (")

        else:
            prev = str(weightlist[idx - 1])
            if prev == "M":
                prev = str(weightlist[idx - 2])
            elif idx > 1:
                weight_str.append(" + ")
            current_layer_size = str(weightlist[idx])
            weight_str.append(prev + "*" + current_layer_size)
            bias_str.append(current_layer_size)

    weight_str.append("}")

    print("=> Weights = ", "".join(weight_str))
    print("=> Biases = ", " + ".join(bias_str))


def count_parameters(model_type, loaded_model=None, print_module=True):
    """
    Returns the number of trainable parameters in the model.

    :param model:
    :return:
    """
    if loaded_model is None:
        model = torch.load(model_type.path)
    else:
        model = loaded_model
    classifier = model.classifier
    feat = model.features

    classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    feat_params = sum(p.numel() for p in feat.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 10, "MODEL ", model_type.name, "=" * 10)
    # '{:,}'.format(1234567890.001)
    print('%12s  %12s  %12s' % ('Feat', 'Classifier', 'TOTAL'))
    print('%12s  %12s  %12s' % (
        '{:,}'.format(feat_params), '{:,}'.format(classifier_params), '{:,}'.format(total_params)))
    if print_module and hasattr(model_type, 'vgg_config'):
        print_module_composition(model_type.vgg_config)


def view_saved_model(path):
    """
    View model architecture of a saved model, by specifiying the path.
    :param path:
    :return:
    """
    print(torch.load(path))

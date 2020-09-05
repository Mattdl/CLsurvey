import os

import torch
import torch.nn as nn
import torch.optim as optim

import utilities.utils as utils
from methods.LwF.AlexNet_LwF import AlexNet_LwF
from methods.EBLL.AlexNet_EBLL import AlexNet_EBLL
import methods.Finetune.train_SGD as SGD_Training


def fine_tune_SGD(dset_dataloader, cumsum_dset_sizes, dset_classes, model_path, exp_dir, num_epochs=100, lr=0.0004,
                  freeze_mode=0, weight_decay=0, enable_resume=True, replace_last_classifier_layer=True,
                  save_models_mode=True, freq=5):
    """
    Finetune training pipeline with SGD optimizer.
    (1) Performs training setup: dataloading, init.
    (2) Actual training: SGD_training.py

    :param dataset_path:     path to preprocessed .pth file, containing the val and training datasets
    :param model_path:       input model to start from
    :param exp_dir:          where to output new model
    :param freeze_mode:     true if only training classification layer
    :param enable_resume:   resume from existing epoch.pth.tar file, overwrite otherwise
    :param freq:            epoch frequency of saving model checkpoints
    """

    # Resume
    resume = os.path.join(exp_dir, 'epoch.pth.tar') if enable_resume else ''
    if os.path.isfile(resume):  # Resume if there is already a model in the expdir!
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        print("Resumed from model: ", resume)
    else:
        if not os.path.exists(exp_dir) and save_models_mode:
            os.makedirs(exp_dir)
        if not os.path.isfile(model_path):
            raise Exception("Model path non-existing: {}".format(model_path))
        else:
            model_ft = torch.load(model_path)
            print("Starting from model path: ", model_path)

    # GPU
    use_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()

    # Unpack Wrapper objects
    if isinstance(model_ft, AlexNet_LwF):
        model_ft.model.classifier = nn.Sequential(
            *list(model_ft.model.classifier.children())[:model_ft.last_layer_name + 1])
        model_ft = model_ft.model
    elif isinstance(model_ft, AlexNet_EBLL):
        model_ft.classifier = nn.Sequential(*list(model_ft.classifier.children())[:model_ft.last_layer_name + 1])
        model_ft.set_finetune_mode(True)

    # Reset last classifier layer
    if freeze_mode or replace_last_classifier_layer:
        labels_per_task = [len(task_labels) for task_labels in dset_classes['train']]
        output_labels = sum(labels_per_task)
        model_ft = utils.replace_last_classifier_layer(model_ft, output_labels)
        print("REPLACED LAST LAYER with {} new output nodes".format(output_labels))

    if use_gpu:
        model_ft = model_ft.cuda()
        print("MODEL LOADED IN CUDA GPU")

    if freeze_mode:
        # Freeze net, only train last classifier layer (warmup training).
        # In freeze mode, only the last classification layer is optimized (rest of net is frozen)
        optimizer_ft = optim.SGD(model_ft.classifier._modules['6'].parameters(), lr, momentum=0.9)
    else:
        optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)

    # Start training
    model_ft, best_acc = SGD_Training.train_model(model_ft, criterion, optimizer_ft, lr, dset_dataloader,
                                                  cumsum_dset_sizes, use_gpu, num_epochs, exp_dir, resume,
                                                  save_models_mode=save_models_mode,
                                                  saving_freq=freq)

    return model_ft, best_acc

import os
import time
import copy

import torch
from torch.nn import functional as F
from torch.autograd import Variable

import utilities.utils as utils


def preprocess_merge_IMM(method, model_paths, datasets_path, batch_size, overwrite=False, debug=True):
    """
    Create and save all merged models.
    :param model_paths: list of chronologically all paths, of the models per trained task.
    """
    merged_model_paths = []
    IMM_mode = method.mode
    merge_model_name = 'best_model_' + IMM_mode + '_merge.pth.tar'

    last_task_idx = len(model_paths) - 1

    # Avoiding memory overload when merged models already exist
    if not overwrite:
        for task_list_index in range(len(model_paths) - 1, 0, -1):
            merged_model_path = os.path.join(os.path.dirname(model_paths[task_list_index]), merge_model_name)

            if os.path.exists(merged_model_path):
                print("SKIPPING, MERGE ALREADY EXISTS for task ", task_list_index)
                last_task_idx = task_list_index
            else:
                break

    # Load models in memory
    models = [torch.load(model_path) for model_path in model_paths]
    print("MODELS TO PROCESS:")
    print('\n'.join(model_paths[:last_task_idx + 1]))
    print("LOADED ", len(models), " MODELS in MEMORY")

    # Keep first model (no merge needed)
    merged_model_paths.append(model_paths[0])

    # Head param names
    last_layer_index = str(len(models[0].classifier._modules) - 1)
    head_param_names = ['classifier.{}.{}'.format(last_layer_index, name) for name, p in
                        models[0].classifier._modules[last_layer_index].named_parameters()]

    if debug:
        print("HEAD PARAM NAMES")
        [print(name) for name in head_param_names]

    # equal_alpha = 1 / len(model_paths)
    # alphas = [equal_alpha for model_path in range(0, len(model_paths))]

    # Calculate precisions and sum of all precisions
    if IMM_mode == method.modes[1]:
        start_time = time.time()
        print("MODE IMM PREPROCESSING")
        precision_matrices = []
        sum_precision_matrices = []  # All summed of previous tasks (first task not included)
        precision_name = 'precision_' + IMM_mode + '.pth.tar'
        sum_precision_matrix = None
        for task_list_index in range(0, last_task_idx + 1):
            print("TASK ", task_list_index)
            precision_out_file_path = os.path.join(os.path.dirname(model_paths[task_list_index]), precision_name)
            sum_precision_out_file_path = os.path.join(os.path.dirname(model_paths[task_list_index]),
                                                       "sum_" + precision_name)

            if os.path.exists(precision_out_file_path) and not overwrite:
                precision_matrix = torch.load(precision_out_file_path)
                print('LOADED PRECISION MATRIX FOR TASK {} : {}'.format(task_list_index, precision_out_file_path))
            else:
                # get model and data
                model = models[task_list_index]
                dsets = torch.load(datasets_path[task_list_index])

                dset_loaders = {
                    x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=8,
                                                   pin_memory=True)
                    for x in ['train', 'val']}

                # get parameters precision estimation
                if debug:
                    print("PARAM NAMES")
                    [print(n) for n, p in model.named_parameters() if p.requires_grad]
                model.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
                precision_matrix = diag_fisher(model, dset_loaders, exclude_params=head_param_names)

                assert [precision_matrix.keys()] == [
                    {name for name, p in model.named_parameters() if name not in head_param_names}]
                del model, dset_loaders, dsets

                print("Saving precision matrix: ", precision_out_file_path)
                torch.save(precision_matrix, precision_out_file_path)
            precision_matrices.append(precision_matrix)

            # Update sum
            # Make sum matrix for each of the tasks! (incremental sum)
            if sum_precision_matrix is None:
                sum_precision_matrix = precision_matrix
            else:
                if os.path.exists(sum_precision_out_file_path) and not overwrite:
                    sum_precision_matrix = torch.load(sum_precision_out_file_path)
                    print('LOADED SUM-PRECISION MATRIX FOR TASK {} : {}'.format(task_list_index,
                                                                                sum_precision_out_file_path))
                else:
                    if debug:
                        for name, p in sum_precision_matrix.items():
                            print("{}: {} -> {}".format(name, p.shape, precision_matrix[name].shape))
                    sum_precision_matrix = {name: p + precision_matrix[name]
                                            for name, p in sum_precision_matrix.items()}
                    assert len([precision_matrix[name] != p for name, p in sum_precision_matrix.items()]) > 0

                    # Save
                    torch.save(sum_precision_matrix, sum_precision_out_file_path)
                    print("Saving SUM precision matrix: ", sum_precision_out_file_path)

                sum_precision_matrices.append(sum_precision_matrix)
        elapsed_time = time.time() - start_time
        utils.print_timing(elapsed_time, title="MODE IMM IWS")

    # Create merged model for each task (except first)
    start_time = time.time()
    for task_list_index in range(1, last_task_idx + 1):
        out_file_path = os.path.join(os.path.dirname(model_paths[task_list_index]), merge_model_name)

        # Mean IMM
        if IMM_mode == method.modes[0]:
            merged_model = IMM_merge_models(models, task_list_index, head_param_names, mean_mode=True)
        # Mode IMM
        elif IMM_mode == method.modes[1]:
            merged_model = IMM_merge_models(models, task_list_index, head_param_names, precision=precision_matrices,
                                            sum_precision=sum_precision_matrices[task_list_index - 1], mean_mode=False)
        else:
            raise ValueError("IMM mode is not supported: ", str(IMM_mode))

        # Save merged model on same spot as best_model
        torch.save(merged_model, out_file_path)
        merged_model_paths.append(out_file_path)
        print(" => SAVED MERGED MODEL: ", out_file_path)

        del merged_model
    del models
    elapsed_time = time.time() - start_time
    utils.print_timing(elapsed_time, title="IMM MERGING")

    print("MERGED MODELS:")
    print('\n'.join(merged_model_paths))

    return merged_model_paths


################## MODE IMM  ###############
# here the ewc code goes
def diag_fisher(model, dataset, exclude_params=None):
    print("Calculating precision matrix")
    # initialize space for precision_matrix
    precision = {}
    for n, p in copy.deepcopy(model.params).items():
        if n in exclude_params:
            continue
        p.data.zero_()
        precision[n] = Variable(p.data + 1e-8)

    # fill matrix
    model.eval()
    for phase in dataset.keys():
        for input in dataset[phase]:
            model.zero_grad()
            x, label = input
            x, label = Variable(x).cuda(), Variable(label, requires_grad=False).cuda()
            output = model(x)
            temp = F.softmax(output).data

            targets = Variable(torch.multinomial(temp, 1).clone().squeeze()).cuda()
            loss = F.nll_loss(F.log_softmax(output, dim=1), targets, size_average=True)
            loss.backward()

            for n, p in model.named_parameters():
                if n in exclude_params:
                    continue
                precision[n].data += p.grad.data ** 2 / len(dataset[phase])

    precision_param = {n: p for n, p in precision.items()}
    return precision_param


def IMM_merge_models(models, task_list_idx, head_param_names, precision=None, sum_precision=None, mean_mode=True):
    """
    Mean-IMM:, averaging all the parameters of the trained models up to the given task.
    Mode-IMM: dividing task precision matrix by sum of all task precision matrices

    Here alphas are all equal (1/ #models). All alphas must sum to 1.

    :param models: list with all models preceding and current model of param task
    :param task_list_idx: up to and including which task the models should be merged
    :return: new merged model
    """

    if not mean_mode and (precision is None or sum_precision is None):
        raise Exception("Can only use precision for MODE IMM, not mean IMM")

    print("Merging models for TASK ", str(task_list_idx + 1))
    merged_model = copy.deepcopy(models[task_list_idx])

    total_task_count = task_list_idx + 1  # e.g. task_idx 1, means to avg over task_idx 0 and 1 => 2 tasks
    # alpha_inverse = total_task_count  # comp efficient

    # Iterate params
    for param_name, param_value in merged_model.named_parameters():
        # Don't merge heads (we use separate heads)
        if param_name in head_param_names:
            print("NOT MERGING PARAM {}, as it is a head param name".format(param_name))
            continue

        # Calculate Mean
        mean_param = torch.zeros(param_value.data.size()).cuda()
        for merge_task_idx in range(0, total_task_count):  # Avg over all preceding + including current task

            # Error check
            if models[merge_task_idx].state_dict()[param_name].size() != mean_param.size():
                print("ERROR WHEN MERGING MODELS")
                raise Exception("ERROR WHEN MERGING MODELS: PRECEDING MODEL PARAMS TASK",
                                str(merge_task_idx), " != PARAM SIZE OF REF TASK", str(task_list_idx))

            if mean_mode:  # MEAN IMM
                state_dict = models[merge_task_idx].state_dict()
                param_value = state_dict[param_name]
                mean_param = mean_param + param_value
            else:  # MODE IMM
                merge_weighting = precision[merge_task_idx][param_name] / sum_precision[param_name]
                d_mean_param = merge_weighting.data * models[merge_task_idx].state_dict()[param_name]
                mean_param += d_mean_param

        # Task_idx is count of how many iterated
        if mean_mode:
            mean_param = mean_param / total_task_count  # Cancels out in mode IMM

        # Update avged param
        param_value.data = mean_param.clone()

    return merged_model

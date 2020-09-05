"""Formatting and defining plots for eval gridsearch results"""

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import traceback

from utilities import plot as plot
from methods.method import *
import utilities.utils as utils


def analyze_experiments(experiment_data_entries, hyperparams_selection=None, plot_seq_acc=True,
                        plot_seq_forgetting=False, save_img_parent_dir=None, img_extention='png', legend_location='top',
                        all_diff_color_force=False, ylim=None, taskcount=10):
    """ Pipeline data collection and plotting/summary."""

    # Collect data
    experiment_data_entries, hyperparams_counts, max_task_count = collect_dataframe(experiment_data_entries,
                                                                                    hyperparams_selection, taskcount)
    # Pad entries
    pad_dataframe(experiment_data_entries, hyperparams_counts)

    # Plot
    if save_img_parent_dir is not None:
        filename_template = save_img_parent_dir + "_TASK{}." + img_extention
        filename_template = filename_template.replace(" ", "")
        save_img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imgs',
                                     save_img_parent_dir, img_extention)
        utils.create_dir(save_img_path)
        save_img_path = os.path.join(save_img_path, filename_template)

        plot_multigraphs(experiment_data_entries, save_img_path, max_task_count,
                         plot_seq_forgetting=plot_seq_forgetting,
                         plot_seq_acc=plot_seq_acc,
                         legend_location=legend_location,
                         all_diff_color_force=all_diff_color_force,
                         ylim=ylim,
                         taskcount=taskcount)

    # Collect some statistics
    print_exp_statistics(experiment_data_entries)


class ExperimentDataEntry(object):
    """
    Single object where curve/table will be generated for.
    """

    def __init__(self, dataset, method, model, gridsearch_name, experiment_name, exp_results_parent_dir,
                 label_prefix, exp_name_segment_idxs, between_head_acc, tr_results_root_path, color=None):
        self.dataset = dataset
        self.method = method
        self.model = model
        self.experiment_name = experiment_name
        self.label = self.create_label(label_prefix, exp_name_segment_idxs, method)
        self.between_head_acc = between_head_acc  # Plot between-head acc instead of normal acc

        if color is None:
            self.color = self.get_method_color(method)
        else:
            self.color = color
        self.linestyle = self.get_family_linestyle(method)
        self.marker, self.markersize = self.get_family_marker_and_size(method)

        # Paths
        self.exp_results_parent_dir = exp_results_parent_dir  # Test_paths
        self.hyperparam_parent_path = utils.get_train_results_path(tr_results_root_path, dataset, method.name,
                                                                   model_name=model.name,
                                                                   gridsearch_name=gridsearch_name)  # Train Path results

        # Metrics (all tasks)
        self.seq_acc = {}
        self.seq_forgetting = {}
        self.final_model_seq_test_acc = []

        # Avg (over all tasks final model)
        self.avg_acc = 0
        self.avg_forgetting = 0

        # Additional info
        self.hyperparams = {}

    def get_method_color(self, method):
        # Matplotlib colormaps: https://matplotlib.org/tutorials/colors/colormaps.html
        cmap = plt.cm.get_cmap('tab20')
        step = 0.05
        # Data based
        if method.name == LWF.name:
            return 'dodgerblue'
        elif method.name == EBLL.name:
            return 'b'
        # Other regul
        elif method.name == MAS.name:
            return 'red'
        elif method.name == EWC.name:
            return 'gold'
        elif method.name == SI.name:
            return 'darkorange'
        elif method.eval_name == IMM('mean').eval_name:
            return cmap(step * 11)
        elif method.eval_name == IMM('mode').eval_name:
            return cmap(step * 10)

        # Mask
        elif method.name == PackNet.name:
            return cmap(step * 4)

        # Rehearsal
        elif method.name == GEM.name:
            return cmap(step * 0)
        elif method.name == ICARL.name:
            return cmap(step * 3)

        # Baseline
        elif method.name == Finetune.name:
            return 'black'
        elif method.name == Joint.name:
            return cmap(step * 14)

        elif method.name == HAT.name:
            return cmap(step * 12)

        elif method.name == FinetuneRehearsalFullMem.name:
            return cmap(step * 12)
        elif method.name == FinetuneRehearsalPartialMem.name:
            return cmap(step * 12)
        else:
            return Exception("Method color not defined in here")

    def get_family_linestyle(self, method):
        if method.category is Category.BASELINE:
            return ':'
        return '-'

    def get_family_marker_and_size(self, method):
        default = 3
        if method.category is Category.BASELINE:
            if method.name == Joint.name:
                return "4", default
            else:
                return "4", default
        elif method.category is Category.MASK_BASED:
            return 'x', default
        elif method.category is Category.DATA_BASED:
            return 11, default
        elif method.category is Category.MODEL_BASED:
            if method.name == IMM.name:
                return '+', default
            else:
                return '1', default
        return '1', default

    def __str__(self):
        return ', '.join([self.dataset.name, self.method.name, self.model.name, self.experiment_name])

    def create_label(self, label_prefix, exp_name_segment_idxs, method, join_arg=','):
        if label_prefix is None:
            label_prefix = [self.method.name, self.model.name]
        elif not isinstance(label_prefix, list):
            label_prefix = [label_prefix]
        label_suffix = self.experiment_name.split("_")
        label_segments = label_prefix + label_suffix
        if exp_name_segment_idxs is not None:
            label_segments = [label_segment for idx, label_segment in enumerate(label_segments)
                              if idx in exp_name_segment_idxs and idx < len(label_segments)]
        label = join_arg.join(label_segments)

        # Not using heuristic
        if method.name is Joint.name:
            label += "*"

        return label


def print_exp_statistics(experiment_data_entries, table_sep='\t'):
    print()
    print("-" * 50)
    print("SUMMARY")
    print("-" * 50)

    print(table_sep.join(["'EXPERIMENT'", "'AVG ACC(FINAL MODEL)'", "'AVG FORGETTING(FINAL MODEL)'"]))
    for experiment_data_entry in experiment_data_entries:
        print(
            str(experiment_data_entry.label) + table_sep +
            str(format(experiment_data_entry.avg_acc, '.2f')) +
            ' (' + str(format(experiment_data_entry.avg_forgetting, '.2f')) + ')'
        )


def collect_gridsearch_exp_entries(test_results_root_path, tr_results_root_path,
                                   dataset, method, gridsearch_name, model,
                                   experiment_selection=None,
                                   exp_name_contains=None,
                                   exp_name_not_containing=None,
                                   label_prefix=None,
                                   label_segment_idxs=None,
                                   task_agnostic_mode=False,
                                   between_head_acc=False,
                                   colors=None,
                                   subset='test',
                                   label_func=None):
    """
    Collects experiments from the gridsearch, or only the specified experiment, and makes reformatted entries from them.

    :param label_prefix: Prefix of the label (exp_name is suffix), default: <method.name>_<model.name>
    :param label_segment_idxs: Which segments of the label to keep. (On Split)
    :param linestyle: linestyles = ['-', '--', '-.', ':']
    :param task_agnostic_mode: True if task agnostic eval experiment
    :param between_head_acc: plot between_head_acc
    :return formatted experiment entries
    """
    if between_head_acc:
        assert task_agnostic_mode, 'Can only plot between-head acc if in task_agnostic mode'

    model_name = model.name
    exp_results_parent_dir = utils.get_test_results_path(test_results_root_path, dataset, method.eval_name,
                                                         model_name, gridsearch_name, subset=subset)
    # Experiments to analyse
    if experiment_selection is None or len(experiment_selection) == 0:
        experiment_names = utils.get_immediate_subdirectories(exp_results_parent_dir,
                                                              path_mode=False, sort=True)
    else:
        if isinstance(experiment_selection, list):
            experiment_names = [x.strip() for x in experiment_selection]
        else:
            experiment_names = [experiment_selection.strip()]

    if exp_name_contains is not None:
        if exp_name_not_containing is not None:
            experiment_names = [experiment_name for experiment_name in experiment_names
                                if exp_name_contains in experiment_name
                                and exp_name_not_containing not in experiment_name]
        else:
            experiment_names = [experiment_name for experiment_name in experiment_names
                                if exp_name_contains in experiment_name]
    elif exp_name_not_containing is not None:
        experiment_names = [experiment_name for experiment_name in experiment_names
                            if exp_name_not_containing not in experiment_name]

    if label_func:
        labels = [label_func(experiment_name) for experiment_name in experiment_names]
    else:
        labels = [label_prefix for _ in experiment_names]

    if colors is not None:
        experiment_data_entries = [ExperimentDataEntry(dataset, method, model, gridsearch_name,
                                                       experiment_name, exp_results_parent_dir, labels[idx],
                                                       label_segment_idxs, between_head_acc,
                                                       tr_results_root_path,
                                                       color=colors[idx])
                                   for idx, experiment_name in enumerate(experiment_names)]
    else:
        experiment_data_entries = [ExperimentDataEntry(dataset, method, model, gridsearch_name,
                                                       experiment_name, exp_results_parent_dir, labels[idx],
                                                       label_segment_idxs, between_head_acc,
                                                       tr_results_root_path)
                                   for idx, experiment_name in enumerate(experiment_names)]
    return experiment_data_entries


def collect_dataframe(exp_data_entries, hyperparams_selection=None, taskcount=None):
    """
    Read dict eval results and put data in the entries.
    """

    hyperparams_counts = {}
    max_task_count = 0
    for exp_data_entry_idx, exp_data_entry in enumerate(exp_data_entries[:]):
        print("preprocessing experiment: ", exp_data_entry)
        taskcount = taskcount if taskcount else exp_data_entry.dataset.task_count

        if taskcount > max_task_count:
            max_task_count = taskcount

        # single perf file
        joint_full_batch = True if exp_data_entry.method.name == Joint.name else False

        # seq perf files
        for dataset_index in range(taskcount):
            # Define paths with original exp_name
            acc_filename = utils.get_perf_output_filename(exp_data_entry.method.eval_name, dataset_index,
                                                          joint_full_batch)
            exp_results_file = os.path.join(exp_data_entry.exp_results_parent_dir, exp_data_entry.experiment_name,
                                            acc_filename)

            # LOAD EVAL RESULTS
            try:
                method_performances = torch.load(exp_results_file)
            except Exception:
                print("LOADING performance ERROR: REMOVING FROM PLOT EXPS")
                del exp_data_entries[(exp_data_entry_idx)]
                traceback.print_exc(5)
                break  # stop iterating other idxes

            metric_dict_key = 'seq_res'  # ACC
            if exp_data_entry.between_head_acc:
                metric_dict_key = 'seq_head_acc'

            if exp_data_entry.method.name not in method_performances:
                assert len(method_performances.keys()) == 1
                for key in method_performances:
                    eval_results = method_performances[key][metric_dict_key]  # Hack for Rahaf EBLL LWF results
            else:
                eval_results = method_performances[exp_data_entry.method.name][metric_dict_key]

            if joint_full_batch:
                eval_results = reformat_single_sequence(eval_results, dataset_index, repeatings_for_curve=taskcount)

            # PARSE AND STORE EVAL metrics
            collect_eval_metrics(exp_data_entry, eval_results, dataset_index, taskcount)

            # LOAD HYPERPARAMS
            load_hyperparams = True
            if joint_full_batch:
                load_hyperparams = False
            if exp_data_entry.method.name == Finetune.name:
                load_hyperparams = False
            if exp_data_entry.method.name == EBLL.name and dataset_index == 0:
                load_hyperparams = False

            if load_hyperparams:
                hyperparams_task_dir = os.path.join(exp_data_entry.hyperparam_parent_path,
                                                    exp_data_entry.experiment_name,
                                                    "task_" + str(dataset_index + 1))

                if not dataset_index == 0 or os.path.exists(hyperparams_task_dir):
                    try:
                        hyperparam_path = os.path.join(hyperparams_task_dir, 'TASK_TRAINING', 'hyperparams.pth.tar')
                        hyperparams_dict = torch.load(hyperparam_path)
                    except Exception:
                        # print("LOADING HYPERPARAMS FAILED: ", hyperparam_path)
                        # traceback.print_exc(5)
                        continue

                    collect_hyperparams(exp_data_entry, hyperparams_dict, hyperparams_counts, hyperparams_selection)

        exp_data_entry.avg_acc /= exp_data_entry.dataset.task_count
        exp_data_entry.avg_forgetting /= exp_data_entry.dataset.task_count
    return exp_data_entries, hyperparams_counts, max_task_count


def collect_eval_metrics(exp_data_entry, eval_results, dataset_index, taskcount):
    """Update values of an entry."""
    # Collect EVAL metrics
    if isinstance(eval_results, list):
        eval_results = {'': eval_results}
    assert len(eval_results.keys()) == 1
    for result_key in eval_results:
        res = eval_results[result_key][:taskcount - dataset_index]
        exp_data_entry.seq_acc[dataset_index] = res
        exp_data_entry.final_model_seq_test_acc.append(res[-1])
        exp_data_entry.avg_acc += exp_data_entry.final_model_seq_test_acc[-1]

        if len(res) > 1:
            exp_data_entry.seq_forgetting[dataset_index] = [res[0] - task_res
                                                            for task_res in res[1:]]
            assert len(exp_data_entry.seq_forgetting[dataset_index]) + 1 == len(exp_data_entry.seq_acc[dataset_index])
            exp_data_entry.avg_forgetting += exp_data_entry.seq_forgetting[dataset_index][-1]
        else:
            exp_data_entry.seq_forgetting[dataset_index] = []


def reformat_single_sequence(eval_results, dataset_index, plot_full_curve=False, repeatings_for_curve=None):
    """Reformat for methods with only 1 performance sequence (from 1 model), e.g. Joint."""
    if repeatings_for_curve is None:
        repeatings_for_curve = len(eval_results)
    if not plot_full_curve:
        repeatings_for_curve -= dataset_index
    extended_result = {'acc': [eval_results[dataset_index]] * repeatings_for_curve}
    return extended_result


def collect_hyperparams(exp_data_entry, hyperparams_dict, hyperparams_counts, hyperparams_selection):
    """Collects hyperparams and stores them in the exp_data_entry."""
    if hyperparams_selection is None:
        hyperparams_keys = list(hyperparams_dict.keys())  # Print all
    else:
        hyperparams_keys = hyperparams_selection

    for hyperparam_key in hyperparams_keys:
        value = hyperparams_dict[hyperparam_key]

        if hyperparam_key not in exp_data_entry.hyperparams:
            exp_data_entry.hyperparams[hyperparam_key] = []
        exp_data_entry.hyperparams[hyperparam_key].append(value)

        # Count how many from each hyperparam to pad afterwards
        if hyperparam_key not in hyperparams_counts:
            hyperparams_counts[hyperparam_key] = 0
        count = len(exp_data_entry.hyperparams[hyperparam_key])
        if count > hyperparams_counts[hyperparam_key]:
            hyperparams_counts[hyperparam_key] = count


def pad_dataframe(experiment_data_entries, hyperparams_counts, pad_value=0):
    """Tables require all lists to have same length. Therefore padding to max value."""
    for hyperparam, count in hyperparams_counts.items():
        for exp_data_entry in experiment_data_entries:
            # Add zero row
            if hyperparam not in exp_data_entry.hyperparams:
                exp_data_entry.hyperparams[hyperparam] = [pad_value] * count

            # Padd row
            elif len(exp_data_entry.hyperparams[hyperparam]) < count:
                pad_size = count - len(exp_data_entry.hyperparams[hyperparam])
                exp_data_entry.hyperparams[hyperparam] += [pad_value] * pad_size

            elif len(exp_data_entry.hyperparams[hyperparam]) > count:
                raise Exception("Should've been counted in collection step")


def get_colors(experiment_data_entries):
    # Define colors
    colors = ['C0', 'C2', 'C1', 'C4', 'C6', 'C7', 'C3', 'C9', 'C8', 'C3']
    extra_colors_count = len(experiment_data_entries) - 10
    if extra_colors_count > 0:
        color_selection = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        for count, color in enumerate(color_selection):
            colors.append(color)
            if count == extra_colors_count:
                break
    return colors


def get_plot_label(experiment_data_entry):
    """Label appended in legend after the method name."""
    if experiment_data_entry.method.name == Joint.name:
        return ": " + '%.2f (n/a)' % (experiment_data_entry.avg_acc)  # plot final model avg acc
    else:
        return ": " + '%.2f (%.2f)' % (
            experiment_data_entry.avg_acc, experiment_data_entry.avg_forgetting
        )  # plot final model avg acc


def plot_multigraphs(experiment_data_entries, save_img_path, max_task_count,
                     label_avg_plot_acc=True, plot_seq_acc=True, plot_seq_forgetting=False,
                     legend_location='top', all_diff_color_force=False, ylim=None, taskcount=10):
    """All eval data for all tasks in one graph (horizontally stacked task performances)."""
    acc_plots = []
    forgetting_plots = []
    labels = []
    linestyles = []
    colors = []
    single_dot_idxes = []
    markers = []
    markersizes = []
    for dataset_index in range(max_task_count):
        print("PLOTTING TASK NUMBER :", str(dataset_index + 1))

        # Fetch labels for this dataset testing plot
        acc_curves = []
        forgetting_curves = []
        for idx, experiment_data_entry in enumerate(experiment_data_entries):
            try:
                acc_curve = experiment_data_entry.seq_acc[dataset_index]
                acc_curves.append(acc_curve)

                forgetting_curve = experiment_data_entry.seq_acc[dataset_index]
                forgetting_curves.append(forgetting_curve)

                # Only need to init all exps once
                if dataset_index == 0:
                    label = experiment_data_entry.label
                    if label_avg_plot_acc:
                        label += get_plot_label(experiment_data_entry)
                    labels.append(label)
                    linestyles.append(experiment_data_entry.linestyle)
                    colors.append(experiment_data_entry.color)
                    markers.append(experiment_data_entry.marker)
                    markersizes.append(experiment_data_entry.markersize)
                    if experiment_data_entry.method.name == Joint.name:
                        single_dot_idxes.append(idx)
            except:
                print("NOT PLOTTING IDX ", idx)

        acc_plots.append(acc_curves)
        forgetting_plots.append(forgetting_curves)

    if all_diff_color_force:
        colors = get_colors(experiment_data_entries)

    if save_img_path is not None:
        suffix = ''
        suffix_nr = 1
        while os.path.exists(save_img_path.format("ALL" + suffix)):
            suffix_nr += 1
            suffix = '_v{}'.format(str(suffix_nr))
        save_img_path = save_img_path.format("ALL" + suffix)

        print("Saving path: ", save_img_path)

    if plot_seq_acc:
        try:
            plot.plot_line_horizontal_sequence(acc_plots, colors, linestyles, labels, markers, markersizes,
                                               legend=legend_location,
                                               ylabel="Accuracy %",
                                               save_img_path=save_img_path,
                                               single_dot_idxes=single_dot_idxes,
                                               ylim=ylim,
                                               taskcount=taskcount)
        except Exception as e:
            print("ACC PLOT ERROR: ", e)
            traceback.print_exc()

    if plot_seq_forgetting:
        try:
            plot.plot_line_horizontal_sequence(forgetting_plots, colors, linestyles, labels, markers, markersizes,
                                               legend=legend_location,
                                               ylabel="Forgetting %",
                                               save_img_path=save_img_path)
        except Exception as e:
            print("FORGETTING PLOT ERROR: ", e)
            traceback.print_exc()

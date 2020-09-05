import numpy as np
import pylab
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Serif'
rcParams['font.sans-serif'] = ['DejaVuSerif']
import matplotlib.pyplot as plt


def plot_line_horizontal_sequence(plots_data, colors, linestyles, labels, markers, markersizes, save_img_path=None,
                                  ylim=None,
                                  legend="out",
                                  ylabel="Accuracy % after learning all tasks",
                                  y_label_fontsize=19,
                                  xlabel="Training Sequence Per Task",
                                  x_label_fontsize=19,
                                  start_y_zero=False,
                                  labelmode='minor',
                                  single_dot_idxes=None,
                                  taskcount=10):
    """
    Checkout for markers: https://matplotlib.org/api/markers_api.html

    :param curves_data: Ordered array of arrays [ [<data_seq_curve1>], [<data_seq_curve2>], ... ]
    :param labels: Ordered array of labels
    :param legend: best or "upper/lower/center right/left/center"
    """
    legend_col = 4  # 5
    height_inch = 8
    width_inch = 20
    x_tick_fontsize = 16
    y_tick_fontsize = 18
    legendsize = 16
    bg_alpha = 1
    bg_color = 'whitesmoke'
    plt.ion()

    task_idxs = [0, 4, 9, 14, 19] if taskcount > 10 else [i for i in range(0, taskcount)]
    print("task_idxs={}".format(task_idxs))

    panel_length = len(plots_data[0][0])  # Length of 1 plot panel
    curves_per_plot = len(plots_data[0])  # Curves in 1 plot
    plot_count = len(task_idxs)  # Amount of stacked plots next to eachother

    print("panel_length={}".format(panel_length))
    print("curves_per_plot={}".format(curves_per_plot))
    print("plot_count={}".format(plot_count))

    if single_dot_idxes is None:
        single_dot_idxes = []

    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    print('Adding plot data')
    for i, plot_idx in enumerate(task_idxs):  # horizontal subplots
        curves_data = plots_data[plot_idx]
        print("Plot idx = {}".format(plot_idx))
        for curve_idx, curve_data in enumerate(curves_data):  # curves in 1 subplot
            # Shift to graph + offset of not testing on prev task
            plot_Xshift = i * panel_length + 1 * plot_idx

            X = np.arange(len(curve_data)) + plot_Xshift
            label = labels[curve_idx] if i == 0 else None
            marker = markers[curve_idx]
            markersize = markersizes[curve_idx]
            print("Plot X = {}".format(X))
            print("Xshift={}".format(plot_Xshift))

            if curve_idx in single_dot_idxes:  # Plot e.g. JOINT as single point at the end
                X = X[-1]
                curve_data = curve_data[-1]
                markersize = 12

            ax.plot(X, curve_data, color=colors[curve_idx], label=label, linewidth=1.5, marker=marker,
                    linestyle=linestyles[curve_idx], markersize=markersize)
    # No Top/right border
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # Put X-axis ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    subplot_offset = 0.1  # TODO

    if ylim is not None:
        ax.set_ylim(top=ylim)

    # Background
    print('Adding plot span')
    for idx, task_idx in enumerate(task_idxs):
        ax.axvspan(idx * panel_length + subplot_offset, (1 + idx) * panel_length - subplot_offset,
                   facecolor=bg_color, alpha=bg_alpha)

    ##############################
    # X-axis gridline positions
    # Major
    # XgridlinesPosMajor = np.linspace(0, (panel_length) * plot_count, num=plot_count)

    # Minor
    # offset_idx = 0
    upper_ticksoffset = -4
    XgridlinesPosMinor, XgridlinesPosMajor = [], []
    for idx, task_idx in enumerate(task_idxs):
        XgridlinesPosMinor.append(idx * panel_length + task_idx)
        XgridlinesPosMajor.append(int(idx * panel_length + panel_length / 2 + upper_ticksoffset))
        # offset_idx += 1
    XgridlinesPosMajor = np.asarray(XgridlinesPosMajor)
    print("XgridlinesPosMinor={}".format(XgridlinesPosMinor))
    print("XgridlinesPosMajor={}".format(XgridlinesPosMajor))

    ###############################
    # Labels
    print("Setting labels")
    Xtick_minorlabels = ['T{}'.format(idx + 1) for idx in task_idxs]
    Xtick_majorlabels = np.repeat('T1', len(XgridlinesPosMajor))

    if labelmode == 'major':
        # Labels Major labeling only
        # Xticks = np.linspace(0, 10, ticks_per_plot)
        Xticks = XgridlinesPosMajor

        # Set
        ax.set_xticks(Xticks, minor=False)
        ax.set_xticklabels(Xtick_majorlabels, minor=False)
    elif labelmode == 'both':
        # Labels both major minor gridlines
        offset_idx = 0
        Xticks_gridlines = []
        Xtick_labels = []
        for idx, task_idx in enumerate(task_idxs):
            Xticks_gridlines.append(idx * panel_length)
            Xtick_labels.append('T{}'.format((task_idx + 1) % panel_length))
            if offset_idx > 0:
                Xticks_gridlines.append(idx * panel_length + offset_idx)
                Xtick_labels.append('T{}'.format((task_idx + 1) % panel_length))
            offset_idx += 1
        Xticks = Xticks_gridlines
        # Set
        plt.xticks(Xticks, Xtick_labels, fontsize=10, color='black')

    elif labelmode == 'minor':
        # Labels only on minor gridlines
        Xticks = XgridlinesPosMinor

        # Set
        ax.set_xticks(Xticks, minor=True)
        ax.set_xticklabels(Xtick_minorlabels, minor=True)
        ax.set_xticklabels([], minor=False)

    print("Setting ticks")
    # Actual Ticks with Labels
    ax.tick_params(axis='y', which='major', labelsize=y_tick_fontsize)
    ax.tick_params(axis='x', which='minor', labelsize=x_tick_fontsize)
    ax.tick_params(axis='x', which='major', labelsize=x_tick_fontsize, length=0)

    # Axis titles
    ax.set_xlabel(xlabel, fontsize=x_label_fontsize, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=y_label_fontsize, labelpad=5)
    ax.set_xlim(-1, len(task_idxs) * taskcount + 1)

    # Grid lines
    ax.set_xticks(XgridlinesPosMinor, minor=True)
    ax.set_xticks(XgridlinesPosMajor, minor=False)

    ax.xaxis.grid(True, linestyle='--', alpha=0.4, which='minor')
    ax.xaxis.grid(True, linestyle='-', alpha=0.8, which='major', color='white')

    # y-axis
    if start_y_zero:
        ax.set_ylim(bottom=0)

    # Legend
    print("Setting legend")
    if legend == "top":
        leg = ax.legend(bbox_to_anchor=(0., 1.20, 1., 0.1), loc='upper center', ncol=legend_col,
                        prop={'size': legendsize},
                        mode="expand", fancybox=True, fontsize=24)  # best or "upper/lower/center right/left/center"
    else:
        leg = ax.legend(bbox_to_anchor=(0., -0.36, 1., -.136), loc='upper center', ncol=legend_col,
                        prop={'size': legendsize},
                        mode="expand", fancybox=True, fontsize=24)  # best or "upper/lower/center right/left/center"

    # Update legend linewidths
    for idx, legobj in enumerate(leg.legendHandles):
        if idx not in single_dot_idxes:
            legobj.set_linewidth(2.0)
        else:
            legobj.set_linewidth(0)
        legobj._legmarker.set_markersize(8.0)

    # TOP axis
    print("Setting axes")
    ax_top = ax.twiny()
    print("chkpt{}".format(1))
    ax_top.set_xlim(-1, taskcount * len(task_idxs) + 1)  # MUST BE SAME AS ax
    top_ticks = XgridlinesPosMajor + 5
    print("chkpt{}".format(2))

    ax_top.set_xticks(top_ticks, minor=False)
    ax_top.set_xticklabels(Xtick_minorlabels, minor=False)
    print("chkpt{}".format(3))

    ax_top.tick_params(axis=u'both', which=u'both', length=0)
    ax_top.tick_params(axis='x', which='major', labelsize=x_tick_fontsize)
    print("chkpt{}".format(4))

    ax_top.set_xlabel('Evaluation on Task', fontsize=x_label_fontsize, labelpad=10)
    plt.setp(ax_top.get_xaxis().get_offset_text(), visible=False)
    print("chkpt{}".format(5))

    # Format Plot
    # plt.tight_layout()

    print("Saving to {}".format(save_img_path))
    if save_img_path is not None:
        plt.axis('on')
        pylab.savefig(save_img_path, bbox_inches='tight')
        pylab.clf()
    else:
        pylab.show()  # Show always also when saving


def imshow_tensor(inp, title=None, denormalize=True,
                  mean=np.array([0.485, 0.456, 0.406]),
                  std=np.array([0.229, 0.224, 0.225])):
    """
    Imshow for Tensor.

    :param inp: input Tensor of img
    :param title:
    :param denormalize: denormalize input or not
    :param mean: imgnet mean by default
    :param std: imgnet std by default
    :return:
    """

    inp = inp.cpu().numpy().transpose((1, 2, 0))
    if denormalize:
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import methods.rehearsal.model.baseline_rehearsal_partial_mem as baseline_rehearsal_partial_mem


class Net(baseline_rehearsal_partial_mem.Net):
    """
    Shared head, with all amounts of classes.
    Heads are masked out
    """

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        args.full_mem_mode = True
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)

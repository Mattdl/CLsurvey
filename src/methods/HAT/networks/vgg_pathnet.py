import torch
import numpy as np
import methods.HAT.HAT_utils as HATutils
import copy


# Also defined in Pytorch 1.3
class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Net(torch.nn.Module):

    def __init__(self, rawmodel, inputsize, taskcla, args=None):
        super(Net, self).__init__()
        print("Original raw model = {}".format(rawmodel))

        ncha, size, _ = inputsize
        self.taskcla = taskcla
        self.ntasks = len(self.taskcla)

        # # Better config found by HAT
        # expand_factor = 0.258  # match num params -> HAT matches to nb params 7.1M
        # self.N = 3
        # self.M = 16
        # self.L = 5  # our architecture has 5 layers

        # PathNet Params
        self.N = args.parameter[0]  # To be set by PathNet method
        self.M = args.parameter[1]  # Fixed based on paper classification exps

        # Model params
        self.convs = torch.nn.ModuleList()
        self.maxpool_idxs = []  # Conv maxpool idxs
        self.fcs = torch.nn.ModuleList()
        s = size  # Guarantee the modules input/outputs fit

        # ConvLayers
        conv_idx = 0
        out_channels = None
        in_channels = None
        self.maxpool = None
        self.relu = torch.nn.ReLU(inplace=True)
        for mod in rawmodel.features.children():
            if isinstance(mod, torch.nn.Conv2d):

                # Subdivide in modules
                new_conv = torch.nn.ModuleList()
                out_channels = int(mod.out_channels / self.M)  # Based on part of the original model
                in_channels = mod.in_channels if in_channels is None else in_channels  # Init from prev out
                for j in range(self.M):
                    new_conv.append(torch.nn.Conv2d(in_channels, out_channels,
                                                    kernel_size=mod.kernel_size,
                                                    stride=mod.stride,
                                                    padding=mod.padding))
                self.convs.append(new_conv)
                in_channels = out_channels
                conv_idx += 1
            elif isinstance(mod, torch.nn.MaxPool2d):  # Init Maxpool
                if self.maxpool is None:
                    self.maxpool = copy.deepcopy(mod)
                self.maxpool_idxs.append(conv_idx - 1)
                s = HATutils.compute_conv_output_size(s, self.maxpool.kernel_size, stride=self.maxpool.stride)

        # FC
        self.drop_fc = Identity()
        self.classifier = torch.nn.ModuleList()
        in_feats = out_channels * s * s
        fc_idx = 0
        fc_cnt_total = sum(1 for mod in rawmodel.classifier.children() if isinstance(mod, torch.nn.Linear))
        for mod in rawmodel.classifier.children():
            if isinstance(mod, torch.nn.Linear):  # Don't include last head
                if fc_idx < fc_cnt_total - 1:  # If not head
                    # Subdivide in modules
                    new_fc = torch.nn.ModuleList()
                    out_features = int(mod.out_features / self.M)
                    for j in range(self.M):
                        new_fc.append(torch.nn.Linear(in_feats, out_features))
                    self.fcs.append(new_fc)
                    in_feats = out_features
                else:  # Final head layer
                    self.classifier.append(torch.nn.Linear(in_feats, mod.out_features))
                fc_idx += 1
            elif isinstance(mod, torch.nn.Dropout) and not isinstance(self.drop_fc, torch.nn.Dropout):  # Init Dropouts
                self.drop_fc = copy.deepcopy(mod)  # All our dropout layers are defined in same configuration

        # Pathnet vars
        self.L = len(self.convs) + len(self.fcs)

        self.bestPath = -1 * np.ones((self.ntasks, self.L, self.N), dtype=np.int)  # Propagate this between tasks
        self.initial_model = copy.deepcopy(self)  # For later rand initialization
        print("Pathnet model: N={}, M={}, L={}".format(self.N, self.M, self.L))
        return

    def forward(self, x, t, P=None):
        layer_idx = 0
        bs = x.size(0)

        # P is the genotype path matrix shaped LxN(no.layers x no.permitted modules)
        if P is None:
            P = self.bestPath[t]

        # Conv
        for conv_idx, conv in enumerate(self.convs):
            maxpool = self.maxpool if conv_idx in self.maxpool_idxs else Identity()
            sum_out = maxpool(self.relu(conv[P[layer_idx, 0]](x)))  # First module activation
            for mod_idx in range(1, self.N):  # Sum over the different module activations
                sum_out = sum_out + maxpool(self.relu(conv[P[layer_idx, mod_idx]](x)))  # sum activations
            layer_idx += 1
            x = sum_out  # Input next layer

        x = x.view(bs, -1)  # Flatten

        # FC
        for fc_idx, fc in enumerate(self.fcs):
            sum_out = self.drop_fc(self.relu(fc[P[layer_idx, 0]](x)))  # First module activation
            for mod_idx in range(1, self.N):  # Sum over the different module activations
                sum_out = sum_out + self.drop_fc(self.relu(fc[P[layer_idx, mod_idx]](x)))  # sum activations
            layer_idx += 1
            x = sum_out  # Input next layer

        # Head output
        y = self.classifier[0](x)
        return y

    def unfreeze_path(self, t, Path):
        """ Path = LxN numpy matrix. """
        # freeze modules not in path P and the ones in bestPath paths for the previous tasks
        for i in range(self.M):
            layer_idx = 0
            for conv_idx, conv in enumerate(self.convs):
                bp = self.bestPath[0:t, layer_idx, :][0] if t > 0 else []
                self.unfreeze_module(conv, i, Path[layer_idx, :], bp)
                layer_idx += 1
            for fc_idx, fc in enumerate(self.fcs):
                bp = self.bestPath[0:t, layer_idx, :][0] if t > 0 else []
                self.unfreeze_module(fc, i, Path[layer_idx, :], bp)
                layer_idx += 1
        return

    def unfreeze_module(self, layer, i, Path, bestPath):
        if (i in Path) and (i not in bestPath):  # if the current module is in the path and not in the bestPath
            HATutils.set_req_grad(layer[i], True)
        else:
            HATutils.set_req_grad(layer[i], False)
        return

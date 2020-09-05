import torch
import copy

# Also defined in Pytorch 1.3
class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Net(torch.nn.Module):
    """
    All VGGs models differ, so need dynamic allocation of the embeddings etc.
    """

    def __init__(self, rawmodel, inputsize, taskcla, args=None, uniform_init=True):
        super(Net, self).__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla

        self.convs = torch.nn.ModuleList()
        self.conv_embs = torch.nn.ModuleList()
        self.maxpool_idxs = []  # Conv maxpool idxs
        self.fcs = torch.nn.ModuleList()
        self.fc_embs = torch.nn.ModuleList()

        # ConvLayers
        # All embedding stuff should start with 'e'
        conv_idx = 0
        self.maxpool = None
        self.relu = torch.nn.ReLU(inplace=True)
        for mod in rawmodel.features.children():
            if isinstance(mod, torch.nn.Conv2d):
                mod_c = copy.deepcopy(mod)
                self.convs.append(mod_c)
                self.conv_embs.append(torch.nn.Embedding(len(self.taskcla), mod_c.out_channels))
                conv_idx += 1
            elif isinstance(mod, torch.nn.MaxPool2d):  # Init Maxpool
                if self.maxpool is None:
                    self.maxpool = copy.deepcopy(mod)
                self.maxpool_idxs.append(conv_idx - 1)

        # FC
        # All embedding stuff should start with 'e'
        self.drop_fc = Identity()
        self.classifier = torch.nn.ModuleList()
        fc_idx = 0
        fc_cnt_total = sum(1 for mod in rawmodel.classifier.children() if isinstance(mod, torch.nn.Linear))
        for mod in rawmodel.classifier.children():
            if isinstance(mod, torch.nn.Linear):  # Don't include last head
                mod_c = copy.deepcopy(mod)
                if fc_idx < fc_cnt_total - 1:
                    self.fcs.append(mod_c)
                    self.fc_embs.append(torch.nn.Embedding(len(self.taskcla), mod_c.out_features))
                else:  # Final head layer
                    self.classifier.append(mod_c)
                fc_idx += 1
            elif isinstance(mod, torch.nn.Dropout) and not isinstance(self.drop_fc, torch.nn.Dropout):  # Init Dropouts
                self.drop_fc = copy.deepcopy(mod)  # All our dropout layers are defined in same configuration

        # After conv layers feature map size: (smid, smid)
        smid_sq = self.fcs[0].in_features / self.convs[-1].out_channels
        self.smid = int(smid_sq ** 0.5)
        assert self.convs[-1].out_channels * self.smid * self.smid == self.fcs[0].in_features  # Check

        # HAT specific
        self.gate = torch.nn.Sigmoid()
        self.enable_warmup = True
        self.smax = None
        self.lamb = None

        if uniform_init:
            lo, hi = 0, 2
            for emb in self.conv_embs:
                emb.weight.data.uniform_(lo, hi)
            for emb in self.fc_embs:
                emb.weight.data.uniform_(lo, hi)
        return

    def forward(self, t, x, s=1, masks=None, first_drop=False):
        """
        :param t: task idx
        :param x: inputs
        :param s: s for annealing
        :param masks: overwrite masks only for maximal plasticity search, s is not used
        """
        if isinstance(t, int):
            t = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)

        # Gates
        bs = x.size(0)

        if masks is None:
            masks = self.mask(t, s=s)
        conv_masks = masks[:len(self.conv_embs)]
        fc_masks = masks[len(self.conv_embs):]

        # Conv
        for conv_idx, conv in enumerate(self.convs):
            maxpool = self.maxpool if conv_idx in self.maxpool_idxs else Identity()
            x = maxpool(self.relu(conv(x)))
            x = x * conv_masks[conv_idx].view(1, -1, 1, 1).expand_as(x)

        x = x.view(bs, -1)  # Flatten

        # FC
        for fc_idx, fc in enumerate(self.fcs):
            if first_drop:
                x = self.relu(fc(self.drop_fc(x)))  # Alexnett
            else:
                x = self.drop_fc(self.relu(fc(x)))  # VGG
            x = x * fc_masks[fc_idx].expand_as(x)

        # Head output
        y = self.classifier[0](x)
        return y, masks

    def mask(self, t, s=1):
        layer_masks = []
        for conv_emb in self.conv_embs:
            layer_masks.append(self.gate(s * conv_emb(t)))
        for fc_emb in self.fc_embs:
            layer_masks.append(self.gate(s * fc_emb(t)))
        return layer_masks

    def premask_summary(self, t, smax):
        """ Both embedding + mask summary. """
        print("=" * 80)
        print("Task {}: PREMASK SUMMARY (smax={})".format(t, smax))
        print("=" * 80)

        if isinstance(t, int):
            t = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)

        for idx, conv_emb in enumerate(self.conv_embs):
            print("Conv_layer={},\n "
                  "Emb: u={:.4f}, std={:.4f} \n "
                  "Mask: <0.1={}, >0.9={}".format(
                idx,
                (conv_emb(t).mean()),
                (conv_emb(t).std()),
                (self.gate(smax * conv_emb(t)) < 0.1).nonzero().shape[0],
                (self.gate(smax * conv_emb(t)) > 0.9).nonzero().shape[0]))
            print("-" * 80)

        for idx, fc_emb in enumerate(self.fc_embs):
            print("FC_layer={},\n "
                  "Emb: u={:.4f}, std={:.4f} \n "
                  "Mask: <0.1={}, >0.9={}".format(
                idx,
                fc_emb(t).mean(),
                fc_emb(t).std(),
                (self.gate(smax * fc_emb(t)) < 0.1).nonzero().shape[0],
                (self.gate(smax * fc_emb(t)) > 0.9).nonzero().shape[0]))
            print("-" * 80)

        print("=" * 80)
        return

    def premask_summary_ext(self, t, smax):
        """ Both embedding + mask summary. """
        print("=" * 80)
        print("Task {}: PREMASK SUMMARY (smax={})".format(t, smax))
        print("=" * 80)

        if isinstance(t, int):
            t = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)

        for idx, conv_emb in enumerate(self.conv_embs):
            print("Conv_layer={},\n "
                  "Emb: u={:.4f}, std={:.4f} \n "
                  "Mask: <0.1={}, >0.9={}".format(
                idx,
                (conv_emb(t).mean()),
                (conv_emb(t).std()),
                (self.gate(smax * conv_emb(t)) < 0.1).nonzero().shape[0],
                (self.gate(smax * conv_emb(t)) > 0.9).nonzero().shape[0]))
            print("-" * 80)

        for idx, fc_emb in enumerate(self.fc_embs):
            print("FC_layer={},\n "
                  "Emb: u={:.4f}, std={:.4f} \n "
                  "Mask: <0.1={}, >0.9={}".format(
                idx,
                fc_emb(t).mean(),
                fc_emb(t).std(),
                (self.gate(smax * fc_emb(t)) < 0.1).nonzero().shape[0],
                (self.gate(smax * fc_emb(t)) > 0.9).nonzero().shape[0]))
            print("-" * 80)

        print("=" * 80)
        return

    def backmask_summary(self, t, smax, mask_back, avg_only=False, include_bias=False):
        cap_left, max_cap_left, min_cap_left = 100, 100, 100
        names = []
        res = []

        if len(mask_back) > 0:  # MASK BACK STORED AS '1 - x' --> here we print x
            print("=" * 80)
            print("Task {}: STARTING WITH BACKMASK (smax={})".format(t, smax))
            print("=" * 80)
            sum_cap_left = 0
            cnt_cap_left = 0
            max_cap_left = 0
            min_cap_left = 101
            total = 0
            active = 0
            deactive = 0
            for n, _ in self.named_parameters():
                if 'bias' in n and not include_bias:
                    continue
                if n in mask_back:
                    deactive_l = (mask_back[n] > 0.9).nonzero().shape[0]
                    active_l = (mask_back[n] < 0.1).nonzero().shape[0]
                    total_l = mask_back[n].numel()
                    rem = total_l - deactive_l - active_l
                    cap_left = deactive_l / total_l * 100
                    cap_used = 100 - cap_left

                    # Task results in sequence
                    names.append(n)
                    res.append(cap_used)
                    if not avg_only:
                        print("Layer={},\n "
                              "Mask: <0.1={}, >0.9={}, [0.1,0.9]={}, total={} | cap_left={:.1f}%".format(
                            n, deactive_l, active_l, rem, total_l, cap_left
                        ))
                        print("-" * 80)
                    # Update avgs
                    max_cap_left = max(max_cap_left, cap_left)
                    min_cap_left = min(min_cap_left, cap_left)
                    sum_cap_left += cap_left
                    cnt_cap_left += 1

                    # Update total cnts
                    total += total_l
                    active += active_l
                    deactive += deactive_l

            rem = total - deactive - active
            cap_left = deactive / total * 100
            print("TOTAL: Mask: <0.1={}, >0.9={}, [0.1,0.9]={}, total={} | cap_left={:.1f}%".format(
                deactive, active, rem, total, cap_left
            ))
            print("Per-layer AVG: cap_left={:.1f}%".format(sum_cap_left / cnt_cap_left))
            print("Max: cap_left={:.1f}%".format(max_cap_left))
            print("Min: cap_left={:.1f}%".format(min_cap_left))
            print("-" * 80)
        else:
            print("NO BACKMASK")
        print("=" * 80)
        return cap_left, max_cap_left, min_cap_left, names, res # res are the backmasks!!

    def get_view_for(self, n, masks):
        # print("Getting view for {}".format(n))
        conv_masks = masks[:len(self.conv_embs)]
        fc_masks = masks[len(self.conv_embs):]

        try:
            segs = n.split('.')  # e.g. 'convs.0.weight', 'conv_embs.5.weight', 'fcs.0.weight', 'fcs.0.bias'
            idx = int(segs[1])
            assert len(segs) == 3 and idx >= 0
        except:
            return None

        if segs[0] == 'convs':
            if n == "convs.0.weight":  # First layer doesn't consider input activations
                return conv_masks[0].data.view(-1, 1, 1, 1).expand_as(self.convs[0].weight)  # Only return post
            else:
                if segs[-1] == 'weight':
                    post = conv_masks[idx].data.view(-1, 1, 1, 1).expand_as(self.convs[idx].weight)
                    pre = conv_masks[idx - 1].data.view(1, -1, 1, 1).expand_as(self.convs[idx].weight)
                    return torch.min(post, pre)
                elif segs[-1] == 'bias':
                    return conv_masks[idx].data.view(-1)
        elif segs[0] == 'fcs':
            if n == "fcs.0.weight":  # First layer doesn't consider input activations
                post = fc_masks[0].data.view(-1, 1).expand_as(self.fcs[0].weight)
                pre = conv_masks[-1].data.view(-1, 1, 1).expand(
                    (self.conv_embs[-1].weight.size(1), self.smid, self.smid)
                ).contiguous().view(1, -1).expand_as(self.fcs[0].weight)
                return torch.min(post, pre)
            else:
                if segs[-1] == 'weight':
                    post = fc_masks[idx].data.view(-1, 1).expand_as(self.fcs[idx].weight)
                    pre = fc_masks[idx - 1].data.view(1, -1).expand_as(self.fcs[idx].weight)
                    return torch.min(post, pre)
                elif segs[-1] == 'bias':
                    return fc_masks[idx].data.view(-1)

        return None

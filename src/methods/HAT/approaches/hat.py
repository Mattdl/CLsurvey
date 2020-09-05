import time
import numpy as np
import torch
import os
from copy import deepcopy

import methods.HAT.HAT_utils as HATutils
from torch.autograd import Variable


########################################################################################################################

class Appr(object):

    def __init__(self, model, exp_dir, nepochs=100, sbatch=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=10,
                 clipgrad=10000, args=None):
        self.model = model
        self.exp_dir = exp_dir
        self.save_freq = args.save_freq

        self.momentum = 0.9  # Prev 0.9 req to train
        self.weight_decay = args.weight_decay
        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        # Orig paper
        # self.lamb = lamb  # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        # self.smax = smax  # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400

        assert len(args.parameter) == 2
        self.smax = args.parameter[0]  # s_max
        self.post_lamb = args.parameter[1]  # c in paper --> Sparsity regularization strength
        self.warmup_lamb = 0  # The first
        # self.pre_lamb = self.post_lamb  # No warmup
        self.lamb = None

        # Warmup
        self.warmup_lr = 0.01
        self.enable_warmup = self.model.enable_warmup

        # Based on epochs
        self.warmup_epochs = 10
        self.min_epochs = int(self.nepochs / 2)  # Min epochs for init task model
        print("smax={},post_lamb={}, enable_warmup={}, warmup_lamb={}, warmup_epochs={}".format(
            self.smax, self.post_lamb, self.enable_warmup, self.warmup_lamb, self.warmup_epochs))

        return

    @staticmethod
    def init_masks(current_task, model, smax):
        """
        Init the masks again for every new task, as the s_max might change in our framework.
        This allows for less steep sigmoid slopes, allowing more plasticity.
        """
        mask_pre = None
        for t in range(current_task):  # Don't include current task
            # At the end of the task: Update activations mask for next task
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)  # Embedding idx

            # a^{t} -> mask (Retrieve from current task embedding)
            mask = model.mask(task, s=smax)  # per-layer masks [gc1, gc2, gc3, gfc1, gfc2]
            for i in range(len(mask)):  # detach data for each layer
                mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)

            # Only need to store 2 masks, the cumulative prev ones, and the current mask
            # a^{<t} -> self.mask_pre (prediction mask in forward pass), prev task final mask
            if t == 0:
                mask_pre = mask
            else:
                for i in range(len(mask_pre)):
                    mask_pre[i] = torch.max(mask_pre[i], mask[i])
            # a^{<=t} -> self.mask_pre (updated)

        mask_back = {}  # Grads/Weights mask
        if mask_pre is not None:  # Not for first task
            for n, _ in model.named_parameters():
                vals = model.get_view_for(n, mask_pre)  # Adapt to gradient shape
                if vals is not None:
                    mask_back[n] = 1 - vals  # backprop mask: 1 - a^{<t}

        return mask_pre, mask_back

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return HATutils.HAT_SGD(self.model.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def train(self, t, dset_loaders, eps=1e-6):

        # Restore chkpt
        loaded_chkpt = False
        chkpt_path = os.path.join(self.exp_dir, 'epoch.pth.tar')
        if os.path.exists(chkpt_path):
            print('Chkpt at: {}'.format(chkpt_path))
            chkpt = torch.load(chkpt_path)

            try:
                assert abs(self.smax - chkpt['smax']) < eps  # Avoid inconsistencies with unexpected interuptions
                assert abs(self.post_lamb - chkpt['post_lamb']) < eps

                init_e = chkpt['e']
                self.model.load_state_dict(chkpt['model'])
                self.optimizer.load_state_dict(chkpt['optimizer'])
                best_acc = deepcopy(chkpt['best_acc'])
                lr = deepcopy(chkpt['lr'])
                patience = deepcopy(chkpt['patience'])
                warmup = deepcopy(chkpt['warmup'])
                print('Loaded chkpt: {}'.format("".join(["{}={}".format(x, chkpt[x])
                                                         for x in ['best_acc', 'lr', 'patience', 'warmup']])))
                del chkpt
                loaded_chkpt = True
            except Exception as e:
                print('No chkpt loaded:{}'.format(e))

        if not loaded_chkpt:  # No chkpt
            patience = self.lr_patience
            best_acc = 0
            init_e = 0
            warmup = t == 0 and self.enable_warmup
            lr = self.lr if not warmup else self.warmup_lr
            self.optimizer = self._get_optimizer(lr)

        best_model = HATutils.get_model(self.model)
        HATutils.print_optimizer_config(self.optimizer)

        # Init masks
        self.mask_pre, self.mask_back = Appr.init_masks(t, self.model, self.smax)
        self.model.backmask_summary(t, self.smax, self.mask_back)  # How much capacity starting?

        # Loop epochs
        for e in range(init_e, self.nepochs):
            self.lamb = self.warmup_lamb if warmup else self.post_lamb

            # Train
            clock0 = time.time()
            train_loss, train_acc = self.train_epoch(t, dset_loaders['train'])
            clock1 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms | Train: loss={:.6f}, acc={:5.1f}% |'.format(
                e + 1,
                1000 * self.sbatch * (clock1 - clock0) / len(dset_loaders['train']),
                train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, dset_loaders['val'])
            print(' Valid: loss={:.6f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            print(' lamb={:.4f} |'.format(self.lamb), end='')

            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = HATutils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
                torch.save(best_model, os.path.join(self.exp_dir, 'best_model.pth.tar'))
            elif not warmup:
                patience -= 1
                if patience == self.lr_patience // 2:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    HATutils.set_lr_(self.optimizer, lr)
                elif patience <= 0:
                    if e < self.min_epochs and t == 0:
                        print("[BREAK SUSPEND] need at least {} epochs".format(self.min_epochs), end='')
                    else:
                        print("[BREAK] Patience=0/{}, with lr={:.1e}".format(self.lr_patience, lr))
                        break

            # End warmup for initial task
            if warmup and e >= self.warmup_epochs:
                warmup = False
                patience = self.lr_patience
                HATutils.set_lr_(self.optimizer, self.lr)
                print("[WARMUP END] Lambda_pre -> lambda_post (lr={})".format(self.lr), end='')

            if (e + 1) % self.save_freq == 0:
                torch.save(
                    {'post_lamb': self.post_lamb, 'smax': self.smax, 'warmup': warmup,
                     'e': e + 1, 'patience': patience, 'best_acc': best_acc, 'lr': lr,
                     'optimizer': self.optimizer.state_dict(), 'model': self.model.state_dict(), },
                    chkpt_path)
                print(" -> chkpt", end='')
            print()

        # AFTER TASK POSTPROCESSING
        self.model = best_model  # Restore best validation model from state dict
        self.model.smax = self.smax  # Hyperparams the model was trained with (To easily retrieve at inference for eval)
        self.model.lamb = self.lamb

        self.model.premask_summary(t, self.smax)  # How much capacity used for current task?
        torch.save(self.model, os.path.join(self.exp_dir, 'best_model.pth.tar'))
        print("Saving self.model={}".format(self.model))
        return self.model, best_acc

    def train_epoch(self, t, dset_loader, thres_cosh=50, thres_emb=6, use_gpu=True, debug=False):
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_num = 0
        task = torch.autograd.Variable(torch.LongTensor([t]).cuda())

        # Iterate over data.
        batch_idx = 0
        for data in dset_loader:
            images, targets = data  # get the inputs
            bs = images.shape[0]
            images = images.cuda() if use_gpu else images
            targets = targets.cuda() if use_gpu else targets
            images, targets = Variable(images), Variable(targets)

            progress_ratio = batch_idx / (len(dset_loader) - 1)
            batch_idx += 1
            assert 0 <= progress_ratio <= 1
            s = (self.smax - 1 / self.smax) * progress_ratio + 1 / self.smax
            if debug:
                print('| s={:.3f}, smax={:.3f}, pr={:.3f} |'.format(s, self.smax, progress_ratio), end='')

            # Forward
            output, masks = self.model.forward(task, images, s=s)  # a^t
            loss, reg = self.criterion(output, targets,
                                       masks)
            if debug:
                print('| loss={:.3f}, reg={:.3f} |'.format(loss, reg))

            # Backward
            self.optimizer.zero_grad()  # Zero the grads
            loss.backward()

            self.optimizer.step(self.model, self.mask_back, t, s, thres_cosh, self.smax, self.clipgrad)

            # Constrain embeddings
            for n, p in self.model.named_parameters():
                if 'embs' in n:
                    p.data = torch.clamp(p.data, -thres_emb, thres_emb)

            # Predict for running acc/
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * bs
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += bs
        return total_loss / total_num, total_acc / total_num

    def eval(self, t, dset_loader, use_gpu=True):
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            total_num = 0
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), requires_grad=False)
            self.model.eval()

            total_reg = 0
            for data in dset_loader:
                images, targets = data  # get the inputs
                bs = images.shape[0]

                # wrap them in Variable
                images = images.cuda() if use_gpu else images
                targets = targets.cuda() if use_gpu else targets
                images, targets = Variable(images, requires_grad=False), Variable(targets, requires_grad=False)

                # Forward
                logits, masks = self.model.forward(task, images, s=self.smax)
                loss, reg = self.criterion(logits, targets, masks)
                _, pred = logits.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy().item() * bs
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += bs
                total_reg += reg.data.cpu().numpy().item() * bs

        print('<reg={:.6f}/ce={:.6f}>'.format(total_reg / total_num, (total_loss - total_reg) / total_num), end='')

        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets, masks):
        """ Add reg to CrossEntropy for sparsity. """
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):  # R term
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:  # if task == 0 -> L1 loss of attention masks
            for m in masks:
                reg += m.sum()  # numerator
                count += np.prod(m.size()).item()  # denominator
        reg /= count
        return self.ce(outputs, targets) + self.lamb * reg, self.lamb * reg

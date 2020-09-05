import time
import torch
import os
from copy import deepcopy

import methods.HAT.approaches.hat as hat
import methods.HAT.HAT_utils as HATutils
from torch.autograd import Variable


########################################################################################################################

class Appr(hat.Appr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.model, "mask_pre"):  # Cleanup, free mem
            del self.model.mask_pre
        self.ft_mask = None
        self.momentum = 0.9
        self.nepochs += self.warmup_epochs
        return

    def get_ft_mask(self, task_idx):
        """
        Return mask which enables all units.
        """
        task = torch.autograd.Variable(torch.LongTensor([task_idx]).cuda(), volatile=False)  # Embedding idx
        mask = self.model.mask(task, s=self.smax)  # per-layer masks [gc1, gc2, gc3, gfc1, gfc2]
        for i in range(len(mask)):  # detach data for each layer
            mask[i] = torch.autograd.Variable(torch.ones_like((mask[i]), requires_grad=False))
        return mask

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return HATutils.HAT_SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def train(self, t, dset_loaders):
        self.mask_pre, self.mask_back = self.init_masks(t, self.model, self.smax)  # Back-mask
        self.ft_mask = self.get_ft_mask(t)  # Important HAT-FT specific

        # Restore chkpt
        chkpt_path = os.path.join(self.exp_dir, 'epoch.pth.tar')
        if os.path.exists(chkpt_path):
            print('Chkpt at: {}'.format(chkpt_path))
            chkpt = torch.load(chkpt_path)
            init_e = deepcopy(chkpt['e'])
            self.model.load_state_dict(chkpt['model'])
            self.optimizer.load_state_dict(chkpt['optimizer'])
            best_acc = deepcopy(chkpt['best_acc'])
            lr = deepcopy(chkpt['lr'])
            patience = deepcopy(chkpt['patience'])
            print('Loaded chkpt: {}'.format("".join(["{}={}".format(x, chkpt[x])
                                                     for x in ['best_acc', 'lr', 'patience', 'warmup']])))
            del chkpt
        else:  # No chkpt
            patience = self.lr_patience
            best_acc = 0
            init_e = 0
            lr = self.lr
            self.optimizer = self._get_optimizer(lr)

        best_model = HATutils.get_model(self.model)
        HATutils.print_optimizer_config(self.optimizer)

        # Loop epochs
        for e in range(init_e, self.nepochs):
            # Train
            clock0 = time.time()
            train_loss, train_acc = self.train_epoch(t, dset_loaders['train'])
            clock1 = time.time()
            # train_loss, train_acc = self.eval(t, dset_loaders['train'])
            # clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1,
                1000 * self.sbatch * (clock1 - clock0) / len(dset_loaders['train']),
                train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, dset_loaders['val'])
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = HATutils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
                if os.path.exists(self.exp_dir):  # Created in framework if we want to store models
                    torch.save(best_model, os.path.join(self.exp_dir, 'best_model.pth.tar'))
            else:
                patience -= 1
                if patience == self.lr_patience // 2:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    HATutils.set_lr_(self.optimizer, lr)
                elif patience == 0:
                    print("[BREAK] Patience=0/{}, with lr={:.1e}".format(self.lr_patience, lr))
                    break

            if (e + 1) % self.save_freq == 0:
                torch.save(
                    {'model': self.model.state_dict(), 'e': e + 1, 'patience': patience, 'best_acc': best_acc, 'lr': lr,
                     'optimizer': self.optimizer.state_dict()}, chkpt_path)
                print(" -> chkpt", end='')
            print()
        return best_model, best_acc

    def train_epoch(self, t, dset_loader, thres_cosh=50, thres_emb=6, use_gpu=True):
        """
        Fixing prev task units by constraining the gradient.
        We assume the prev task masks are near-binary, acts as gate:
            -> Zero grad the prev task important units. (grad x 0)
            -> Allow others to be altered (grad x 1)
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_num = 0

        # Iterate over data.
        batch_idx = -1
        for data in dset_loader:
            batch_idx += 1
            images, targets = data  # get the inputs
            bs = images.shape[0]
            images = images.cuda() if use_gpu else images
            targets = targets.cuda() if use_gpu else targets

            images, targets = Variable(images, volatile=False), Variable(targets, volatile=False)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)

            # Forward
            output, _ = self.model.forward(task, images, masks=self.ft_mask)  # a^t
            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step(self.model, self.mask_back, t, finetune=True)
            # self.optimizer.step()  # Regular SGD

            # Predict for running acc/
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * bs
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += bs
        return total_loss / total_num, total_acc / total_num

    def eval(self, t, dset_loader, use_gpu=True):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        for data in dset_loader:
            images, targets = data  # get the inputs
            bs = images.shape[0]
            images = images.cuda() if use_gpu else images
            targets = targets.cuda() if use_gpu else targets

            images, targets = Variable(images, volatile=True), Variable(targets, volatile=True)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=True)

            # Forward
            output, masks = self.model.forward(task, images, masks=self.ft_mask)
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * bs
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += bs
        return total_loss / total_num, total_acc / total_num

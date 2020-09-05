import time
import numpy as np
import os
from copy import deepcopy

import torch
from torch.autograd import Variable

import methods.HAT.HAT_utils as HATutils


class Appr(object):
    # Based on paper and largely on https://github.com/dai-dao/pathnet-pytorch and https://github.com/kimhc6028/pathnet-pytorch

    def __init__(self, model, exp_dir, nepochs=100, sbatch=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=1000, generations=20, args=None):
        self.momentum = 0.9
        self.weight_decay = args.weight_decay  # Weight decay should already work! Optimizer only does current path

        # Params
        if len(args.parameter) >= 1:
            assert len(args.parameter) == 3
            self.N = args.parameter[0]
            self.M = args.parameter[1]
            self.generations = args.parameter[2]

        # Adapt Model
        self.model = model
        # self.model.N = self.N # Don't adapt N here yet!
        self.L = self.model.L  # layers with paths in the network
        self.ntasks = self.model.ntasks
        assert self.model.M == self.M
        self.init_bestPath()  # Init N, extend bestPath

        # Init
        self.exp_dir = exp_dir
        # self.generations = generations  # Grid search = [5,10,20,50,100,200]; best was 20
        self.P = 2  # from paper Secs 2.4 and 2.5, numbers of the individuals in each generation/paths to be trained

        self.nepochs = nepochs // self.generations  # To maintain same number of training updates
        self.sbatch = sbatch
        self.lr = lr
        # self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        # TODO RM
        # self.nepochs = 2
        # self.lr = 0.1
        # self.generations = 1

        print("Pathnet init with: epochs/gen={}, lr={}, N={}, M={}, gen={}".format(
            self.nepochs, self.lr, self.N, self.M, self.generations))

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                               momentum=self.momentum, weight_decay=self.weight_decay)

    def init_bestPath(self):
        """ Extend numpy array dynamically. """
        if self.N > self.model.N:
            print("Extending bestPath array N={} -> {}".format(self.model.N, self.N))
            assert self.model.N == self.model.bestPath.shape[-1]
            new_bestPath = -1 * np.ones((np.array(self.model.bestPath).shape[0:2]) + (self.N,), dtype=np.int)
            for task_idx, task in enumerate(self.model.bestPath):  # Fill inCopy bestpath data here
                for layer_idx, layer in enumerate(task):
                    mod = self.model.bestPath[task_idx][layer_idx]
                    new_bestPath[task_idx][layer_idx][:len(mod)] = mod
        elif self.model.N == self.N:
            print("N={} remained the same.".format(self.N))
        else:
            raise ValueError("N should only increase in the framework! {} -> {}".format(self.model.N, self.N))
        # Update
        self.model.N = self.N

    def reinit_modules(self, t):
        # Conv layers
        offset = 0
        self.copy_mod_(self.model.convs, self.model.initial_model.convs, offset, t)

        # FC layers
        offset = len(self.model.convs)
        self.copy_mod_(self.model.fcs, self.model.initial_model.fcs, offset, t)

    def copy_mod_(self, model1, model2, offset, t):
        for (n, p), (m, q) in zip(model1.named_parameters(), model2.named_parameters()):
            if n == m:
                layer, module, par = n.split(".")
                module = int(module)
                layer = int(layer)
                if module not in self.model.bestPath[0:t, layer + offset]:
                    p.data = deepcopy(q.data)  # copy rand init from init model if not in any best path of prev tasks

    def train(self, t, dset_loaders):
        if t > 0:  # reinit modules not in bestpath with random, according to the paper
            self.reinit_modules(t)

        # init path for this task
        Path = np.random.randint(0, self.M - 1, size=(self.P, self.L, self.N))
        guesses = list(range(self.M))
        optim = []
        lr = []
        patience = []
        best_acc = []
        for p in range(self.P):  # For all participants in the tournament (here binary, P=2)
            optim.append(self._get_optimizer(self.lr))
            lr.append(self.lr)
            patience.append(self.lr_patience)
            best_acc.append(0)
            for j in range(self.L):  # Init the random paths
                np.random.shuffle(guesses)
                Path[p, j, :] = guesses[:self.N]  # Max N distinct modules/pathway(participant)

        winner = 0
        best_path_model = HATutils.get_model_state(self.model)
        best_acc_overall = 0

        for g in range(self.generations):
            battle_win = -1
            for p in range(self.P):

                # train only the modules in the current path, minus the ones in the model.bestPath
                self.model.unfreeze_path(t, Path[p])

                # the optimizer trains solely the params for the current task
                if p != battle_win:
                    optim[p] = self._get_optimizer(lr[p])  # Renew loser optimizer
                self.optimizer = optim[p]  # Keep optimizer of winner for momentum

                # Loop epochs
                for e in range(self.nepochs):
                    # Train
                    clock0 = time.time()
                    self.train_epoch(t, dset_loaders['train'], Path[p])
                    clock1 = time.time()
                    train_loss, train_acc = self.eval(t, dset_loaders['train'], Path[p])
                    clock2 = time.time()
                    print('| Generation {:3d} | Path {:3d} | Epoch {:3d},'
                          ' time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                        g + 1, p + 1, e + 1,
                        1000 * self.sbatch * (clock1 - clock0) / len(dset_loaders['train']),
                        1000 * self.sbatch * (clock2 - clock1) / len(dset_loaders['train']),
                        train_loss, 100 * train_acc),
                        end='')
                    # Valid
                    valid_loss, valid_acc = self.eval(t, dset_loaders['val'], Path[p])
                    print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                    # Save the winner
                    if valid_acc > best_acc_overall:
                        best_acc_overall = valid_acc
                        best_path_model = HATutils.get_model_state(self.model)
                        winner = p
                        print(' B', end='')

                    # Adapt lr
                    if valid_acc > best_acc[p]:
                        print("val > best ({} > {})".format(valid_acc, best_acc[p]), end='')
                        best_acc[p] = valid_acc
                        patience[p] = self.lr_patience
                        print(' *', end='')
                    else:
                        print("val <= best ({} <= {})".format(valid_acc, best_acc[p]), end='')
                        patience[p] -= 1
                        if patience[p] == self.lr_patience // 2:
                            lr[p] /= self.lr_factor
                            print(' lr={:.1e}'.format(lr[p]), end='')
                            HATutils.set_lr_(optim[p], lr[p])  # Use same optimizer
                        elif patience == 0:
                            print("[BREAK] Patience=0/{}, with lr={:.1e}".format(self.lr_patience, lr))
                            break
                    print()
                battle_win = winner

            # Restore winner model
            HATutils.set_model_state_(self.model, best_path_model)
            print('| Winning path: {:3d} | Best overall acc: {:.3f} |'.format(winner + 1, best_acc_overall))

            # Keep the winner and mutate it
            print('Mutating')
            probability = 1 / (self.N * self.L)  # probability to mutate
            for p in range(self.P):
                if p != winner:
                    best_acc[p] = 0
                    lr[p] = lr[winner]
                    patience[p] = self.lr_patience
                    for j in range(self.L):
                        for k in range(self.N):
                            Path[p, j, k] = Path[winner, j, k]
                            if np.random.rand() < probability:
                                Path[p, j, k] = (Path[p, j, k] + np.random.randint(-2, 2)) % self.M  # add int in [-2,2]

        # save the best path into the model
        self.model.bestPath[t] = Path[winner]
        torch.save(self.model, os.path.join(self.exp_dir, 'best_model.pth.tar'))
        print("-> Saving best model with best paths")
        print(self.model.bestPath[t])

        return self.model, best_acc_overall

    def train_epoch(self, t, dset_loader, Path, use_gpu=True):
        self.model.train()

        for data in dset_loader:
            images, targets = data  # get the inputs

            if use_gpu:  # wrap them in Variable
                images, targets = Variable(images.cuda(), volatile=False), Variable(targets.cuda(), volatile=False)
            else:
                images, targets = Variable(images, volatile=False), Variable(targets, volatile=False)

            # Forward
            output = self.model.forward(images, t, Path)
            loss = self.criterion(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.clipgrad)
            self.optimizer.step()

        return

    def eval(self, t, dset_loader, Path=None, use_gpu=True):
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            total_num = 0
            self.model.eval()

            # Loop batches
            for data in dset_loader:
                images, targets = data  # get the inputs
                bs = images.shape[0]

                if use_gpu:  # wrap them in Variable
                    images, targets = Variable(images.cuda(), volatile=True), Variable(targets.cuda(), volatile=True)
                else:
                    images, targets = Variable(images, volatile=True), Variable(targets, volatile=True)

                # Forward
                output = self.model.forward(images, t, Path)
                loss = self.criterion(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy().item() * bs
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += bs

        return total_loss / total_num, total_acc / total_num

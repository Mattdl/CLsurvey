import time
import numpy as np
import os
from copy import deepcopy

import torch

import methods.HAT.HAT_utils as HATutils
import methods.HAT.approaches.pathnet as pathnet


class Appr(pathnet.Appr):

    def __init__(self, model, exp_dir, nepochs=70, sbatch=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=1000, generations=20, args=None):
        self.momentum = 0.9
        self.weight_decay = args.weight_decay  # Weight decay should already work! Optimizer only does current path

        self.model = model
        assert len(args.parameter) == 3
        self.N = args.parameter[0]
        self.M = args.parameter[1]
        self.generations = None
        assert self.model.M == self.M

        self.L = self.model.L  # layers with paths in the network
        self.ntasks = self.model.ntasks

        # Init
        self.exp_dir = exp_dir
        self.initial_model = deepcopy(model)
        self.P = 1  # Only 1 participant

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        print("FT-Pathnet init with: epochs{}, lr={}".format(self.nepochs, self.lr))
        return

    def train(self, t, dset_loaders):
        if t > 0:  # reinit modules not in bestpath with random, according to the paper
            self.reinit_modules(t)

        # init path for this task (P=1)
        Path = np.random.randint(0, self.M - 1, size=(self.L, self.M))
        for j in range(self.L):  # Init the random paths
            Path[j, :] = list(range(self.M))  # Include all modules

        # Init
        best_acc = 0
        lr = self.lr
        patience = self.lr_patience
        best_model = HATutils.get_model(self.model)
        best_acc_overall = 0

        # train all modules, minus the ones in the model.bestPath
        self.model.unfreeze_path(t, Path)

        # the optimizer trains solely the params for the current task
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):  # Train
            clock0 = time.time()
            self.train_epoch(t, dset_loaders['train'], Path)
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, dset_loaders['train'], Path)
            clock2 = time.time()
            print('| Epoch {:3d},time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1,
                1000 * self.sbatch * (clock1 - clock0) / len(dset_loaders['train']),
                1000 * self.sbatch * (clock2 - clock1) / len(dset_loaders['train']),
                train_loss, 100 * train_acc),
                end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, dset_loaders['val'], Path)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                patience = self.lr_patience
                best_model = HATutils.get_model_state(self.model)
                print(' *', end='')
            else:
                patience -= 1
                if patience == self.lr_patience // 2:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    HATutils.set_lr_(self.optimizer, lr)
                elif patience == 0:
                    print("[BREAK] Patience=0/{}, with lr={:.1e}".format(self.lr_patience, lr))
                    break
            print()

        # save the best path into the model
        if os.path.exists(self.exp_dir):  # Created in framework if we want to store models
            torch.save(best_model, os.path.join(self.exp_dir, 'best_model.pth.tar'))
        print('| Best val acc: {:.3f} |'.format(best_acc_overall))
        print("-> Saving model={}".format(self.exp_dir))

        return best_model, best_acc

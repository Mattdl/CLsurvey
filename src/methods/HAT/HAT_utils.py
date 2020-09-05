import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm


########################################################################################################################

def print_model_report(model):
    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)
    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    print('-' * 100)
    return


########################################################################################################################

def get_model(model):
    return deepcopy(model)  # deepcopy(model.state_dict())


def get_model_state(model):
    return deepcopy(model.state_dict())


def set_model_state_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def gradivate(module):
    for param in module.parameters():
        param.requires_grad = True
    return


def set_lr_(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


########################################################################################################################

def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    if isinstance(kernel_size, tuple):
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean = 0
    std = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean += image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded = mean.view(mean.size(0), mean.size(1), 1, 1).expand_as(image)
    for image, _ in loader:
        std += (image - mean_expanded).pow(2).sum(3).sum(2)

    std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()

    return mean, std


########################################################################################################################

def fisher_matrix_diag(t, x, y, model, criterion, sbatch=20):
    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    for i in tqdm(range(0, x.size(0), sbatch), desc='Fisher diagonal', ncols=100, ascii=True):
        b = torch.LongTensor(np.arange(i, np.min([i + sbatch, x.size(0)]))).cuda()
        images = torch.autograd.Variable(x[b], volatile=False)
        target = torch.autograd.Variable(y[b], volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images)
        loss = criterion(t, outputs[t], target)
        loss.backward()
        # Get gradients
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += sbatch * p.grad.data.pow(2)
    # Mean
    for n, _ in model.named_parameters():
        fisher[n] = fisher[n] / x.size(0)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
    return fisher


########################################################################################################################

def cross_entropy(outputs, targets, exp=1, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs)
    tar = torch.nn.functional.softmax(targets)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


########################################################################################################################

def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad = req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad = req_grad
    return


########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


########################################################################################################################


class HAT_SGD(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, model, mask_back, t, s=None, thres_cosh=None, smax=None, clipgrad=None, finetune=False,
             closure=None):
        """Performs a single optimization step.

        Constraining joint objective based gradient and weight decay gradient.
        Momentum is disregarded, as cancelled out neurons don't build up momentum anyway.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, (modp_name, modp) in zip(group['params'], model.named_parameters()):
                assert modp is p
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0:
                    if 'embs' not in modp_name:  # Don't decay embedding params, doesn't train
                        d_p.add_(weight_decay, p.data)

                # Constrain grad
                if t > 0:  # Restrict layer gradients in backprop: a^{<t}
                    if modp_name in mask_back:
                        p.grad.data *= mask_back[modp_name]  # See before: stored as (1 - x) with prev task masks

                # Compensate embedding gradients
                if not finetune:
                    if 'embs' in modp_name:
                        num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                        den = torch.cosh(p.data) + 1
                        p.grad.data *= smax / s * num / den

                    # Clip
                    torch.nn.utils.clip_grad_norm(p, clipgrad)

                # Leave momentum as is
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)

        return loss

########################################################################################################################

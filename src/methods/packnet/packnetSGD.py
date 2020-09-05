import torch.optim as optim
import torch


class PacknetSGD(optim.SGD):
    r"""SGD adaptation for PackNet with weight decay.
    if data is set to exactly 0, also no weight decay or other regularization decays are applied.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, custom_L2=None):
        super(PacknetSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.custom_L2 = custom_L2

    def __setstate__(self, state):
        super(PacknetSGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for module_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # ADAPTED FOR PACKNET: only masked out weights are EXACTLY 0, also no other values should be added to these
                if weight_decay != 0:
                    mask = p.grad.data.ne(0).float()  # Create mask from weight grads: 0 stays zero, all other values become 1
                    weight_decay_term = weight_decay * p.data * mask # Decay actual weight values, applying the mask
                    d_p.add_(weight_decay_term)
                if momentum != 0: # No fix needed: Can't buildup momentum if weight is masked out for whole task
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

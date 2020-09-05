import methods.HAT.networks.vgg_hat as vgg_hat


class Net(vgg_hat.Net):
    """ Only difference with dynamic VGG construction is the dropout order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, uniform_init=True)  # Use std Pytorch init N(0,1)
        self.enable_warmup = False
        assert self.smid == 6  # See model summary

    def forward(self, t, x, s=1, masks=None, first_drop=True):
        return super().forward(t, x, s, masks=masks, first_drop=True)

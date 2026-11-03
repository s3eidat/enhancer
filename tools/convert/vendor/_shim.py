"""Minimal stand-ins for the BasicSR helpers the vendored archs import.

default_init_weights / make_layer / pixel_unshuffle follow BasicSR (Apache-2.0).
LayerNorm2d is NAFNet's (MIT), rewritten with plain tensor ops so it exports to ONNX.
"""
import torch
from torch import nn
from torch.nn import init


class _Registry:
    def register(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda obj: obj


ARCH_REGISTRY = _Registry()


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)

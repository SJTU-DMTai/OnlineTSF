import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from adapter.module import ssf, up
from adapter.module.up import LayerNorm, BatchNorm1d, RevIN, _RevIN


def add_down_up_(parent_module: nn.Module, module_name: str, freeze_weight: bool,
             merge_weights=False, load_weights=True, **kwargs):
    old_module = getattr(parent_module, module_name)
    if isinstance(old_module, nn.Linear):
        new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                            bias=old_module.bias is not None, freeze_weight=freeze_weight,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            merge_weights=merge_weights, **kwargs)
    elif isinstance(old_module, transformers.Conv1D):
        new_module = TFMConv1D(nx=old_module.weight.shape[0], nf=old_module.nf, merge_weights=merge_weights,
                               freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, _RevIN) and old_module.affine:
        new_module = RevIN(num_features=old_module.num_features, eps=old_module.eps, affine=True,
                           subtract_last=old_module.subtract_last,
                           merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, nn.BatchNorm1d) and old_module.affine:
        new_module = BatchNorm1d(num_features=old_module.num_features, eps=old_module.eps,
                                 momentum=old_module.momentum, affine=old_module.affine,
                                 track_running_stats=old_module.track_running_stats,
                                 device=old_module.weight.device, dtype=old_module.weight.dtype,
                                 merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, nn.LayerNorm) and len(old_module.normalized_shape) == 1 and old_module.elementwise_affine:
        new_module = LayerNorm(normalized_shape=old_module.normalized_shape[-1], eps=old_module.eps,
                               elementwise_affine=True, device=old_module.weight.device, dtype=old_module.weight.dtype,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)
    elif isinstance(old_module, nn.Conv1d):
        new_module = Conv1d(old_module.in_channels, old_module.out_channels,
                            kernel_size=old_module.kernel_size, stride=old_module.stride, padding=old_module.padding,
                            dilation=old_module.dilation, groups=old_module.groups, bias=old_module.bias is not None,
                            padding_mode=old_module.padding_mode,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            freeze_weight=freeze_weight, merge_weights=merge_weights, **kwargs)
    else:
        raise NotImplementedError
    if load_weights:
        new_module.load_state_dict(old_module.state_dict(), strict=False)
    setattr(parent_module, module_name, new_module)


class Down_Up(up.Adaptation_Up):
    def __init__(self, in_features: int, *args, **kwargs):
        up.Adaptation_Up.__init__(self, *args, **kwargs)
        self.in_features = in_features
        self.register_buffer('scale2', None, persistent=False)

    def assign_adaptation(self, adaptation):
        if adaptation is None:
            self.scale, self.scale2, self.shift = None, None, None
        else:
            self.scale = adaptation[..., :self.out_features] + 1
            self.scale2 = adaptation[..., self.out_features:self.out_features + self.in_features] + 1
            if self.flag_adapt_bias:
                self.shift = adaptation[..., -self.out_features:]
            if self.scale.dim() == 2:
                self.scale = self.scale.unsqueeze(1)
                self.scale2 = self.scale2.unsqueeze(1)
                self.shift = self.shift.unsqueeze(1)

    def _merge(self, weight, bias):
        weight, bias = super()._merge(weight, bias)
        if weight is not None and self.flag_adapt_weight:
            scale2 = self.scale2.squeeze()
            if self.fan_in_fan_out:
                weight = weight * scale2.reshape(scale2.shape[-1:] + (1,) * (weight.dim() - 1))
            else:
                weight = weight * scale2.reshape((1, scale2.shape[-1]) + (1,) * (weight.dim() - 2))
        return weight, bias

    def _ssf_input(self, x: torch.Tensor):
        batch_size = x.size()[:-1]
        x = x.reshape(self.scale2.shape[0], -1, x.shape[-1])
        return (x * self.scale2).view(*batch_size, x.shape[-1])

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-1]
        res = res.view(self.scale.shape[0], -1, res.shape[-1])
        if self.bias is not None:
            res = res * self.scale + (self.shift + self.bias)
        else:
            res = res * self.scale
        return res.view(*batch_size, res.shape[-1])


class Linear(Down_Up, ssf.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None, dtype=None,
            merge_weights: bool = True, freeze_weight: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        Down_Up.__init__(self, in_features=in_features, out_features=out_features, flag_adapt_bias=bias,
                         merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if not hasattr(self, 'scale') or self.scale is None:
            return nn.Linear.forward(self, x)
        if self.merged:
            return F.linear(x, self.weight, bias=self.bias)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.linear(x, weight, bias=bias)
        else:
            return self._ssf(F.linear(self._ssf_input(x), self.weight, bias=None))


class TFMConv1D(Down_Up, ssf.AttnConv1D):
    def __init__(
            self,
            nx: int,
            nf: int,
            merge_weights: bool = True, freeze_weight: bool = True,
            **kwargs
    ):
        transformers.Conv1D.__init__(self, nx=nx, nf=nf)
        Down_Up.__init__(self, in_features=nx, out_features=nf, flag_adapt_bias=True, fan_in_fan_out=True,
                         merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if not hasattr(self, 'scale') or self.scale is None:
            return transformers.Conv1D.forward(self, x)
        if self.merged:
            size_out = x.size()[:-1] + (self.nf,)
            return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(size_out)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.linear(x, weight.transpose(-1, -2), bias=bias)
        else:
            return self._ssf(F.linear(self._ssf_input(x), self.weight.transpose(-1, -2), bias=None))


class Conv1d(Down_Up, ssf.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None,
                 merge_weights: bool = True, freeze_weight: bool = True, **kwargs):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        Down_Up.__init__(self, in_features=in_channels, out_features=out_channels, flag_adapt_bias=bias,
                         merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x):
        if not hasattr(self, 'scale') or self.scale is None:
            return nn.Conv1d.forward(self, x)
        if self.merged:
            return self._conv_forward(x, self.weight, bias=self.bias)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return self._conv_forward(x, weight, bias=bias)
        else:
            return self._ssf(self._conv_forward(self._ssf_input(x), self.weight, bias=None))

    def _ssf_input(self, x: torch.Tensor):
        batch_size = x.size()[:-2]
        x = x.view(self.scale2.shape[0], -1, *x.shape[-2:]) * self.scale2.unsqueeze(-1)
        return x.view(*batch_size, *x.shape[-2:])

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-2]
        res = res.view(self.scale.shape[0], -1, *res.shape[-2:])
        if self.bias is not None:
            res = res * self.scale.unsqueeze(-1) + (self.shift + self.bias).unsqueeze(-1)
        else:
            res = res * self.scale.unsqueeze(-1)
        return res.view(*batch_size, *res.shape[-2:])

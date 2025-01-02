import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from layers.RevIN import RevIN as _RevIN
from adapter.module import ssf


def add_adaptation_up_(parent_module: nn.Module, module_name: str, freeze_weight: bool,
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


class Adaptation_Up(ssf.SSF):
    def _merge(self, weight, bias):
        if weight is not None and self.flag_adapt_weight:
            scale = self.scale.squeeze()
            if self.fan_in_fan_out:
                weight = weight * scale.reshape((1, scale.shape[-1]) + (1,) * (weight.dim() - 2))
            else:
                weight = weight * scale.reshape(scale.shape[-1:] + (1,) * (weight.dim() - 1))
        if bias is not None:
            if self.flag_adapt_bias:
                bias = bias + self.shift.squeeze()
        return weight, bias

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-1]
        res = res.view(self.scale.shape[0], -1, res.shape[-1])
        if self.bias is not None:
            res = res * self.scale + (self.shift + self.bias)
        else:
            res = res * self.scale
        return res.view(*batch_size, res.shape[-1])


class Linear(Adaptation_Up, ssf.Linear):
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
        Adaptation_Up.__init__(self, out_features=out_features, flag_adapt_bias=bias,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.scale is None:
            return super().forward(x)
        if self.merged:
            return F.linear(x, self.weight, bias=self.bias)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.linear(x, weight, bias=bias)
        else:
            return self._ssf(F.linear(x, self.weight, bias=None))


class TFMConv1D(Adaptation_Up, ssf.AttnConv1D):
    def __init__(
            self,
            nx: int,
            nf: int,
            merge_weights: bool = True, freeze_weight: bool = True,
            **kwargs
    ):
        transformers.Conv1D.__init__(self, nx=nx, nf=nf)
        Adaptation_Up.__init__(self, out_features=nf, flag_adapt_bias=True, fan_in_fan_out=True,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.scale is None:
            return super().forward(x)
        if self.merged:
            size_out = x.size()[:-1] + (self.nf,)
            return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(size_out)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.linear(x, weight.transpose(-1, -2), bias=bias)
        else:
            return self._ssf(F.linear(x, self.weight.transpose(-1, -2), bias=None))


class Conv1d(Adaptation_Up, ssf.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None,
                 merge_weights: bool = True, freeze_weight: bool = True, **kwargs):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        Adaptation_Up.__init__(self, out_features=out_channels, flag_adapt_bias=bias,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x):
        if self.scale is None:
            return super().forward(x)
        if self.merged:
            return self._conv_forward(x, self.weight, bias=self.bias)
        elif self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return self._conv_forward(x, weight, bias=bias)
        else:
            return self._ssf(self._conv_forward(x, self.weight, bias=None))

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-2]
        res = res.view(self.scale.shape[0], -1, *res.shape[-2:])
        if self.bias is not None:
            res = res * self.scale.unsqueeze(-1) + (self.shift + self.bias).unsqueeze(-1)
        else:
            res = res * self.scale.unsqueeze(-1)
        return res.view(*batch_size, *res.shape[-2:])


class LayerNorm(Adaptation_Up, nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None, merge_weights: bool = True, freeze_weight: bool = True, **kwargs):
        assert isinstance(normalized_shape, int)
        assert elementwise_affine
        nn.LayerNorm.__init__(self, normalized_shape, eps, elementwise_affine, device=device, dtype=dtype)
        Adaptation_Up.__init__(self, out_features=normalized_shape, flag_adapt_bias=True,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.scale is None:
            return super().forward(x)
        if self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        else:
            return self._ssf(F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps))


class BatchNorm1d(Adaptation_Up, nn.BatchNorm1d):
    def __init__(self, num_features, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None, dtype=None,
                 merge_weights: bool = True, freeze_weight: bool = True, **kwargs):
        assert isinstance(num_features, int)
        assert affine
        nn.BatchNorm1d.__init__(self, num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        Adaptation_Up.__init__(self, out_features=num_features, flag_adapt_bias=True,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.scale is None:
            return super().forward(x)

        self._check_input_dim(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if self.scale.shape[0] == 1 and self.merge_weights:
            weight, bias = self._merge(self.weight, self.bias)
            return F.batch_norm(
                x,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                weight,
                bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            return self._ssf(F.batch_norm(
                x,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                None,
                bn_training,
                exponential_average_factor,
                self.eps,
            ).contiguous())

    def _ssf(self, res: torch.Tensor):
        batch_size = res.size()[:-2]
        res = res.view(self.scale.shape[0], -1, res.shape[-2], res.shape[-1])
        if self.bias is not None:
            res = res * self.scale.unsqueeze(-1) + (self.shift + self.bias).unsqueeze(-1)
        else:
            res = res * self.scale.unsqueeze(-1)
        return res.view(*batch_size, res.shape[-2], res.shape[-1])

class RevIN(Adaptation_Up, _RevIN):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False,
                 merge_weights: bool = True, freeze_weight: bool = True, **kwargs):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        assert affine
        _RevIN.__init__(self, num_features, eps, affine, subtract_last, **kwargs)
        Adaptation_Up.__init__(self, out_features=num_features, flag_adapt_bias=True,
                               merge_weights=merge_weights, freeze_weight=freeze_weight, **kwargs)

    def _normalize(self, x):
        if self.scale is None:
            return _RevIN._normalize(self, x)
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            weight = self.scale * self.affine_weight
            bias = self.shift + self.affine_bias
            if self.scale.shape[0] == 1 and self.merge_weights:
                return x * weight.squeeze() + bias.squeeze()
            else:
                shape = (x.shape[0], ) + (1, ) * (x.dim() - 2) + (x.shape[-1], )
                return x * weight.reshape(shape) + bias.reshape(shape)
        return x

    def _denormalize(self, x):
        if self.scale is None:
            return _RevIN._denormalize(self, x)
        if self.affine:
            weight = self.scale * self.affine_weight + self.eps*self.eps
            bias = self.shift + self.affine_bias
            if self.scale.shape[0] == 1 and self.merge_weights:
                x = (x - bias.squeeze()) / weight.squeeze()
            else:
                shape = (x.shape[0], ) + (1, ) * (x.dim() - 2) + (x.shape[-1], )
                x = (x - bias.reshape(shape)) / weight.reshape(shape)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

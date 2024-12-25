import collections
import math

import torch
import torch.nn as nn
import transformers
import typing

from adapter.module import down_up
from adapter.module.base import Adaptation


def clip(x, max_norm=1):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / x_norm, max=1)
    return x * scale, x_norm


class Bottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int,
                 bottleneck_dim: typing.Union[typing.List[int], int], activation=nn.LeakyReLU, need_bias: bool = False,
                 shared=False, rand_init: dict = None):
        super().__init__()
        self.out_dim = out_dim
        self.activation = activation()
        self.shared = shared
        self.need_bias = need_bias
        self.biases = nn.ParameterList([nn.Parameter(torch.empty(n_layers, 1, bottleneck_dim))])
        self.weights = nn.ParameterList([nn.Linear(in_dim, bottleneck_dim, bias=False) if self.shared else
                                         nn.Parameter(torch.empty(n_layers, 1, bottleneck_dim, in_dim))])
        if self.need_bias:
            self.biases.append(nn.Parameter(torch.zeros(n_layers, 1, out_dim)))
        self.weights.append(nn.Parameter(torch.zeros(1 if self.shared else n_layers, 1, out_dim, bottleneck_dim)))
        for i in range(0, len(self.weights) - 1):
            for j in range(n_layers):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weights[i].weight if shared else self.weights[i][0, 0])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.biases[i][j], -bound, bound)
                if not self.shared:
                    nn.init.kaiming_uniform_(self.weights[i][j, 0], a=math.sqrt(5))
        if self.need_bias:
            nn.init.zeros_(self.biases[-1])
            if rand_init is not None:
                for pos, (r, in_fea) in rand_init.items():
                    nn.init.kaiming_uniform_(self.biases[-1][pos, :, :r * in_fea], a=math.sqrt(5))

        self.memories = None
        self.last_adaptation = None

    def forward(self, x, mask=None, training=False):
        if self.shared:
            x = self.activation(self.weights[0](x) + self.biases[0]).unsqueeze(-1)
        else:
            x = self.activation(self.weights[0] @ x.unsqueeze(-1) + self.biases[0].unsqueeze(-1))
        x = self.weights[-1] @ x
        x = x.squeeze(-1)
        if training and mask is not None:
            x = x * mask.unsqueeze(-1)
        if self.need_bias:
            x = x + self.biases[-1]
        return x


class AdaptGenerator(nn.Module):
    def __init__(self, backbone: nn.Module, concept_features: int, activation=nn.LeakyReLU,
                 shared: bool = True, need_bias: bool = True, adaptive_dim: bool = False, mid_dim: int = None):
        super().__init__()
        self.dim_name_dict = collections.defaultdict(list)
        self.bottlenecks = nn.ModuleDict()
        self.loras = nn.ModuleDict()
        self.memories = None
        self.memories_lora = None
        for name, module in backbone.named_modules():
            if isinstance(module, Adaptation):
                _weight = module.affine_weight if hasattr(module, 'affine_weight') else module.weight
                out_features = _weight.shape[1 if isinstance(module, transformers.Conv1D) else 0]
                if _weight.dim() == 1:
                    in_features = 1
                else:
                    in_features = _weight.shape[0 if isinstance(module, transformers.Conv1D) else 1]
                out_dim = out_features
                if module.bias is not None:
                    out_dim += out_features
                if isinstance(module, down_up.Down_Up):
                    out_dim += in_features
                self.dim_name_dict[name.split('.')[-1] + '_' + str(out_dim)].append(name)

        for key, names in self.dim_name_dict.items():
            out_dim = int(key.split('_')[-1])
            _hid_dims = min(mid_dim, out_dim // 4) if adaptive_dim else mid_dim
            self.bottlenecks[key] = Bottleneck(concept_features, out_dim, len(names), _hid_dims,
                                               activation, shared=shared, need_bias=need_bias)

    def forward(self, x, need_clip=False):
        if need_clip:
            x, x_norm = clip(x)
        coefs = {k: bottleneck(x) for k, bottleneck in self.bottlenecks.items()}
        return coefs


import torch
import torch.nn as nn
import transformers
from adapter.module.generator import AdaptGenerator
from adapter.module import down_up


def normalize(W, max_norm=1):
    W_norm = torch.norm(W, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / W_norm, max=1)
    return W * scale


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Proceed(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        if args.freeze:
            backbone.requires_grad_(False)
        self.backbone = add_adapters_(backbone, args)
        self.more_bias = not args.freeze
        self.generator = AdaptGenerator(backbone, args.concept_dim,
                                        activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
                                        adaptive_dim=False, need_bias=self.more_bias,
                                        shared=not args.individual_generator,
                                        mid_dim=args.bottleneck_dim)
        # print(self.adapters)
        self.register_buffer('recent_batch', torch.zeros(1, args.seq_len + args.pred_len, args.enc_in), persistent=False)
        if args.ema > 0:
            self.register_buffer('recent_concept', None, persistent=True)
        self.mlp1 = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seq_len, args.concept_dim), nn.GELU(),
                                  nn.Linear(args.concept_dim, args.concept_dim))
        self.mlp2 = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seq_len + args.pred_len, args.concept_dim), nn.GELU(),
                                  nn.Linear(args.concept_dim, args.concept_dim))
        self.ema = args.ema
        self.flag_online_learning = False
        self.flag_update = False
        self.flag_current = False
        self.flag_basic = False

    def generate_adaptation(self, x):
        concept = self.mlp1(x).mean(-2)
        recent_concept = self.mlp2(self.recent_batch).mean(-2).mean(list(range(0, self.recent_batch.dim() - 2)))
        if self.ema > 0:
            if self.recent_concept is not None:
                recent_concept = self.recent_concept * self.ema + recent_concept * (1 - self.ema)
            if self.flag_update or self.flag_online_learning and not self.flag_current:
                self.recent_concept = recent_concept.detach()
        drift = concept - recent_concept
        res = self.generator(drift, need_clip=not self.args.wo_clip)
        return res

    def forward(self, *x):
        if self.flag_basic:
            adaptations = {}
            for i, (k, adapter) in enumerate(self.generator.bottlenecks.items()):
                adaptations[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(self.generator.dim_name_dict[k])
        else:
            adaptations = self.generate_adaptation(x[0])
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.generator.dim_name_dict[out_dim][i]
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])
        if self.args.do_predict:
            return self.backbone(*x)
        else:
            return self.backbone(*x)

    def freeze_adapter(self, freeze=True):
        for module_name in ['mlp1', 'mlp2']:
            if hasattr(self, module_name):
                getattr(self, module_name).requires_grad_(not freeze)
                getattr(self, module_name).zero_grad(set_to_none=True)
        for adapter in self.generator.bottlenecks.values():
            adapter.weights.requires_grad_(not freeze)
            adapter.weights.zero_grad(set_to_none=True)
            adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
            adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)

    def freeze_bias(self, freeze=True):
        if self.more_bias:
            for adapter in self.generator.bottlenecks.values():
                adapter.biases[-1].requires_grad_(not freeze)
                adapter.biases[-1:].zero_grad(set_to_none=True)


def add_adapters_(parent_module: nn.Module, args, top_level=True):
    for name, module in parent_module.named_children():
        if args.tune_mode == 'all_down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D,
                                                                     nn.LayerNorm, nn.BatchNorm1d)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        else:
            add_adapters_(module, args, False)
    return parent_module

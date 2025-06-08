
import torch
import torch.nn as nn
import transformers
from adapter.module.generator import AdaptGenerator
from adapter.module import down_up
from adapter.proceed import Proceed

class ProceedPlus(Proceed):
    def __init__(self, backbone, args):
        super().__init__(backbone, args)

    def forward(self, *x):
        return super().forward(*x)

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


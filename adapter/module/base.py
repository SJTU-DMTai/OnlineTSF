
import torch.nn as nn


class Adaptation(object):
    def __init__(self, flag_adapt_bias: bool, flag_adapt_weight: bool = True,
                 merge_weights: bool = True, freeze_weight: bool = True, freeze_bias: bool = True):
        assert isinstance(self, nn.Module)
        self.flag_adapt_bias = flag_adapt_bias
        self.flag_adapt_weight = flag_adapt_weight
        self.weight.requires_grad = not freeze_weight
        if self.bias is not None:
            self.bias.requires_grad = not freeze_bias
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    def assign_adaptation(self, adaptation):
        raise NotImplementedError

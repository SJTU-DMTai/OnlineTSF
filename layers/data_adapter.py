import collections
import math

import torch

from torch import nn
from torch.nn import functional as F, init


def cosine(x1, x2, eps=1e-8):
    x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
    x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
    return x1 @ x2.transpose(0, 1)


class LabelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias

class LabelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = LabelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)


class FiLM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(in_dim))
        nn.init.uniform_(self.scale, 0.75, 1.25)

    def forward(self, x):
        return x * self.scale


class FeatureAdapter(nn.Module):
    def __init__(self, in_dim, num_head=4, temperature=4):
        super().__init__()
        self.num_head = num_head
        self.P = nn.Parameter(torch.empty(num_head, in_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.heads = nn.ModuleList([nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_head)])
        self.temperature = temperature

    def forward(self, x):
        s_hat = torch.cat(
            [torch.cosine_similarity(x, self.P[i], dim=-1).unsqueeze(-1) for i in range(self.num_head)], -1,
        )
        # s_hat = cosine(x, self.P)
        s = torch.softmax(s_hat / self.temperature, -1).unsqueeze(-1)
        return x + sum([s[..., i, :] * self.heads[i](x) for i in range(self.num_head)])

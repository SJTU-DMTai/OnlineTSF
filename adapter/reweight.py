import torch
from torch import nn
from torch.nn import functional as F

from adapter.proceed import Transpose


class Reweighter(nn.Module):
    def __init__(self, seq_len, seq_len2, concept_dim):
        super().__init__()
        self.mlp1 = nn.Sequential(Transpose(-1, -2),
                                  nn.Linear(seq_len, concept_dim),
                                  nn.GELU())
        self.mlp2 = nn.Sequential(Transpose(-1, -2),
                                  nn.Linear(seq_len2, concept_dim),
                                  nn.GELU())
        self.fc1 = nn.Linear(concept_dim, concept_dim, bias=False)
        self.fc2 = nn.Linear(concept_dim, concept_dim)
        self.w = nn.Linear(concept_dim, 1, bias=True)
        nn.init.zeros_(self.w.weight)
        nn.init.ones_(self.w.bias)

    def forward(self, labeled_data, test_data):
        drift = self.fc1(self.mlp1(labeled_data).mean(-2)) - self.fc2(self.mlp2(test_data).mean(-2))
        # return self.w(drift).squeeze(-1).exp()
        return F.leaky_relu(self.w(drift).squeeze(-1))

class LossReweighter(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(1, 64), nn.GELU(),
                               nn.Linear(64, 1), nn.LeakyReLU())
        # self.w2 = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.w[2].weight)
        nn.init.ones_(self.w[2].bias)

    def forward(self, loss):
        return self.w(loss.unsqueeze(-1)).squeeze(-1)

class ReweightWrapper(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        # self.reweighter = Reweighter(args.seq_len + args.pred_len, args.seq_len, args.concept_dim)
        self.reweighter = LossReweighter()

    @property
    def meta_learner(self):
        return self.reweighter

    @property
    def basic_model(self):
        return self.backbone

    def forward(self, *x, y=None, test_x=None):
        if test_x is not None:
            return self.backbone(*x), self.reweighter(torch.cat([x[0], y], dim=-2), test_x)
        else:
            return self.backbone(*x)
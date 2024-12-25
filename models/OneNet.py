import copy

import torch
from torch import nn

from models import normalization
from models.FSNet import Model as FSNet
from layers.RevIN import RevIN

class OneNet(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.decision = MLP(n_inputs=args.pred_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh())
        self.weight = nn.Parameter(torch.zeros(args.enc_in))
        # self.bias = nn.Parameter(torch.zeros(args.enc_in))

    def forward(self, *inp):
        flag = False
        if len(inp) == 1:
            inp = inp + (None, 1, 1)
            flag = True
        y1, y2 = self.backbone.forward_individual(*inp[:-2])
        if flag:
            b, t, d = y1.shape
            weight = self.weight.view(1, 1, -1).repeat(b, t, 1)
            loss1 = torch.sigmoid(weight + torch.zeros(b, 1, d, device=weight.device)).view(b, t, d)
            inp = inp[:-2] + (loss1, 1 - loss1)
        return y1.detach() * inp[-2] + y2.detach() * inp[-1], y1, y2

    def store_grad(self):
        self.backbone.store_grad()


class Model_Ensemble(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        _args = copy.deepcopy(args)
        _args.seq_len = 60
        self.seq_len = 60
        self.encoder = normalization.ForecastModel(FSNet(_args), num_features=args.enc_in, seq_len=60)
        # self.norm = False
        # if args.normalization.lower() == 'revin':
        if args.pretrain and hasattr(args, 'fsnet_path'):
            # if 'RevIN' in args.fsnet_path:
            print('Load FSNet from', args.fsnet_path)
            self.encoder.load_state_dict(torch.load(args.fsnet_path)['model'])
            # else:
            #     self.encoder.load_state_dict(torch.load(args.fsnet_path)['model'])
        self.encoder_time = backbone

    def forward_individual(self, x, x_mark):
        y1 = self.encoder_time(x, x_mark)
        y2 = self.encoder.forward(x[..., -self.seq_len:, :], x_mark[..., -self.seq_len:, :] if x_mark is not None else None)
        return y1, y2

    def forward(self, x, x_mark, w1=0.5, w2=0.5):
        y1, y2 = self.forward_individual(x, x_mark)
        return y1 * w1 + y2 * w2, y1, y2

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()

    def try_trigger_(self, flag=True):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.try_trigger = flag
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                # print('{} - {}'.format(name, type(layer).__name__))
                layer.try_trigger = flag

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return x

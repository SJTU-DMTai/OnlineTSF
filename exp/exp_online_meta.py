import copy
import os
from functools import partial, cache

import higher
import torch
from torch import optim, nn, Tensor
from torch.nn import functional as F
from tqdm import tqdm

from adapter import reweight
from data_provider.data_factory import get_dataset, get_dataloader
from exp.exp_online import Exp_Online, OnlineLoss
from data_provider.dataloader_online import Dataset_Recent_And_Replay
from util.tools import has_rnn, override_state, adjust_learning_rate


class OnlineBalanceLoss:
    def __init__(self, pred_len, criterion=nn.MSELoss(reduction='none')):
        self.H = pred_len
        self.criterion = criterion
        self.mask = None

    def get_mask(self, loss: Tensor):
        if self.mask is None or self.mask.size() != loss.size():
            mask = torch.ones_like(loss)
            mask[-self.H:] *= torch.tril(torch.ones(self.H, self.H, device=loss.device)).flip(0)
            self.mask = mask
        return self.mask

    def __call__(self, inputs: Tensor, target: Tensor, reweighter) -> Tensor:
        replay_num = len(inputs) - self.H
        loss = self.criterion(inputs, target).mean(-1) # Last dimension is the variate dimension
        mask = self.get_mask(loss)
        sample_weight = reweighter(loss.mean(-1))
        loss = loss * mask
        sample_weight = sample_weight * mask
        loss_weights = sample_weight.unsqueeze(-1) * torch.ones(self.H, device=inputs.device, dtype=torch.float32)
        loss_weights = torch.cat([loss_weights[:-self.H], loss_weights[-self.H:].tril().flip(0)])
        loss_weights_horizon = loss_weights[:-self.H].sum(dim=0) + loss_weights[-self.H:].tril().flip(0).sum(0)
        loss_weights = loss_weights / loss_weights_horizon

        loss_per_ele = loss_weights * loss
        if sample_weight is None:
            loss = loss_per_ele[:-self.H].sum() + loss_per_ele[-self.H:].flip(0).tril().sum()
            loss = loss / len(loss_per_ele)
        else:
            loss = loss_per_ele.sum() / self.H
        loss = loss.sum()
        return loss

class Exp_Online_Meta(Exp_Online):
    def __init__(self, args):
        args.batch_size = 1
        self.task_num = args.grad_accumulation
        self.replay_num = getattr(args, 'replay_num', 1)
        self.period = getattr(args, 'period', 1)
        self.meta_lr = args.learning_rate
        self.model_lr = args.online_learning_rate
        self.first_order = getattr(args, 'first_order', True)
        self.batch_update = getattr(args, 'batch_update', True)
        self.meta_optim = None
        super().__init__(args)
        self.args.wrap_data_class[-1] = partial(Dataset_Recent_And_Replay, replay_num=self.replay_num,
                                                period=self.period, gap=args.pred_len)
        self.has_rnn = has_rnn(self.model.backbone)
        self.model.meta_learner.requires_grad_(True)
        meta_params = self.model.meta_learner.parameters()
        self.meta_optim = getattr(optim, self.args.optim)(meta_params, lr=self.meta_lr)
        self.online_loss = OnlineLoss(self.args.pred_len)
        self.online_phases = ['val', 'test', 'online']

    def _build_model(self, model=None, framework_class=None):
        model = super()._build_model(model, framework_class=reweight.ReweightWrapper)
        model.meta_learner.requires_grad_(False)
        return model

    def state_dict(self, *args, **kwargs):
        destination = super().state_dict(*args, **kwargs)
        destination['meta_optim'] = self.meta_optim.state_dict()
        return destination

    def _update(self, batches, criterion, optimizer, scaler=None):
        self.iter_count += 1
        all_outputs = []
        support_set, query_set = batches
        with higher.innerloop_ctx(self._model, self.model_optim, copy_initial_weights=False,
                                   track_higher_grads=True) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.has_rnn):
                outputs = fmodel(*self._process_batch(support_set))
                # outputs = fmodel(*self._process_batch(support_set),
                #                  y=support_set[1].to(self.device), test_x=query_set[0].to(self.device))
            loss = self.online_loss(outputs, support_set[1].to(self.device), reweighter=self._model.reweighter)
            # loss = self.online_loss(outputs, support_set[1].to(self.device))
            diffopt.step(loss)
            outputs = fmodel(*self._process_batch(query_set))
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            all_outputs.append(outputs)
            loss = F.mse_loss(outputs, query_set[1].to(self.device))
            self._train_step(loss / self.args.grad_accumulation, self.meta_optim, self.iter_count)
        if self.iter_count % self.args.grad_accumulation == 0:
            self._model.basic_model.load_state_dict(fmodel.basic_model.state_dict())
            self.model_optim.state = override_state(self.model_optim.param_groups, diffopt)
        return loss, all_outputs

    def _update_online(self, batch, criterion, model_optim, scaler=None, current_data=None):
        model_optim.zero_grad()
        if not self.args.pin_gpu:
            batch = [batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) else batch[i] for i in range(len(batch))]
        inp = self._process_batch(batch)
        outputs = self.model(*inp)
        # outputs = self.model(*inp, y=batch[1], test_x=current_data[0].to(self.device))
        # loss = self.online_loss(outputs, batch[1])
        loss = self.online_loss(outputs, batch[1], reweighter=self._model.reweighter)
        if self.args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()
        return loss, outputs

    def update_valid(self, valid_data=None):
        self.args.grad_accumulation = 1
        self.phase = 'online'
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class,
                                     take_post=self.args.pred_len - 1)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        self.meta_optim.zero_grad()
        for i, batch in enumerate(tqdm(valid_loader, mininterval=10)):
            _, outputs = self._update(batch, criterion, model_optim, scaler)
            if self.args.do_predict:
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())
        self._model.meta_learner.requires_grad_(False)
        return predictions
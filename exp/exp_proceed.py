import copy
import numpy as np
import torch
from tqdm import tqdm

from adapter import proceed
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp import Exp_Online


class Exp_Proceed(Exp_Online):
    def __init__(self, args):
        args = copy.deepcopy(args)
        args.merge_weights = 1
        super(Exp_Proceed, self).__init__(args)
        self.online_phases = ['val', 'test', 'online']
        self.mean_dim = args.seq_len

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        if phase == 'val' and self.args.val_online_lr:
            lr = self.model_optim.param_groups[0]['lr']
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = self.args.online_learning_rate
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        ret = super().online(online_data, target_variate, phase, show_progress)
        self._model.freeze_adapter(False)
        if phase == 'val' and self.args.val_online_lr:
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = lr
        return ret

    def update_valid(self, valid_data=None, valid_dataloader=None):
        self.phase = 'online'
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                     **self.wrap_data_kwargs, take_post=self.args.pred_len - 1)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        if not self.args.joint_update_valid:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._model.freeze_bias(False)
                self._model.freeze_adapter(True)
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=False)
                self._model.freeze_bias(True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(False)
                self._model.freeze_adapter(False)
                _, outputs = self._update_online(current_batch, criterion, model_optim, scaler, flag_current=True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(True)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
            self._model.freeze_bias(False)
        else:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=True)
                if self.args.do_predict:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.forward(current_batch)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                    self.model.train()
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))
        return predictions

    def _build_model(self, model=None, framework_class=None):
        model = super()._build_model(model, framework_class= proceed.Proceed)
        print(model)
        return model

    def _update(self, batch, criterion, optimizer, scaler=None):
        self._model.flag_update = True
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        self._model.flag_update = False
        return loss, outputs

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False):
        self._model.flag_online_learning = True
        self._model.flag_current = flag_current
        loss, outputs = super()._update_online(batch, criterion, optimizer, scaler)
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        self._model.flag_online_learning = False
        self._model.flag_current = not flag_current
        return loss, outputs

    def analysis_online(self):
        self._model.freeze_adapter(True)
        return super().analysis_online()

    def predict(self, path, setting, load=False):
        self.update_valid()
        res = self.online()
        np.save('./results/' + setting + '_pred.npy', np.vstack(res[-1]))
        return res
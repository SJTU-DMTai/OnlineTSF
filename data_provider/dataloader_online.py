from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')


class Dataset_Recent_Full_Feedback(Dataset):
    def __init__(self, dataset, gap: Union[int, tuple, list], **kwargs):
        super().__init__()
        self.dataset = dataset
        self.gap = gap

    def __getitem__(self, index):
        return self.dataset[index], self.dataset[index + self.gap]

    def __len__(self):
        return len(self.dataset) - self.gap


class Dataset_Recent_Feedback(Dataset_Recent_Full_Feedback):
    def __init__(self, dataset, gap: Union[int, tuple, list], **kwargs):
        super().__init__(dataset, gap, **kwargs)
        self.dataset = dataset
        self.gap = gap

    def _stack(self, data):
        if isinstance(data[0], np.ndarray):
            return np.vstack(data)
        else:
            return torch.stack(data, 0)

    def __getitem__(self, index):
        if self.gap == 1:
            return self.dataset[index], self.dataset[index + self.gap]
        else:
            current_data = self.dataset[index + self.gap]
            if not isinstance(current_data, tuple):
                recent_data = tuple(self.dataset[index + n] for n in range(self.gap))
                recent_data = self._stack(recent_data)
                return current_data, recent_data
            else:
                recent_data = tuple([] for _ in range(len(current_data)))
                for past in range(self.gap):
                    for j, past_data in enumerate(self.dataset[index + past]):
                        recent_data[j].append(past_data)
                recent_data = tuple(self._stack(recent_d) for recent_d in recent_data)
            return recent_data, current_data

    def __len__(self):
        return len(self.dataset) - self.gap


class Dataset_Recent_And_Replay(Dataset_Recent_Feedback):
    def __init__(self, dataset, gap: Union[int, tuple, list], period: int = 24, replay_num: int = 1, **kwargs):
        self.period = max(1, period)
        self.replay_num = replay_num
        self.border1 = dataset.border[0] if hasattr(dataset, 'border') else 0
        border2 = dataset.border[1]
        super().__init__(dataset, gap, **kwargs)
        self.max_replay_offset = self.dataset.pred_len % self.period - self.period
        # 重新设置data_x和data_y为0到border2的数据
        self.dataset.data_x = self.dataset.data_y = torch.tensor(self.dataset.data[:border2],
                                                                 dtype=self.dataset.data_x.dtype,
                                                                 device=self.dataset.data_x.device)
        self.dataset.data_stamp = torch.tensor(self.dataset.all_data_stamp[:border2],
                                               dtype=self.dataset.data_stamp.dtype,
                                               device=self.dataset.data_stamp.device)

    def _sample_previous_period_indices(self, index: int) -> torch.Tensor | None:
        # 小于index的最大周期数
        end_index = index + self.max_replay_offset

        # 计算可用的历史周期数
        max_periods = end_index // self.period
        if max_periods <= 0:
            return None

        # 生成所有可能的周期偏移
        period_idx = np.random.choice(max_periods, size=min(max_periods, self.replay_num), replace=False)
        # 计算对应的索引
        return end_index - period_idx * self.period

    def __getitem__(self, index) -> Tuple[Tuple, ...]:
        # 获取基本数据（index到index+gap的序列）
        recent_data, current_data = super().__getitem__(index + self.border1)

        # 获取有效的历史周期索引
        sampled_indices = self._sample_previous_period_indices(index + self.border1)

        if sampled_indices is None:
            return recent_data, current_data

        # 获取历史样本并合并
        if isinstance(recent_data, tuple):
            # 多输出情况
            merged_recent = tuple(
                torch.cat([s[i].unsqueeze(0) for s in [self.dataset[idx] for idx in sampled_indices]] + [recent_data[i]], dim=0)
                for i in range(len(recent_data))
            )
        else:
            # 单输出情况
            merged_recent = torch.cat([torch.stack([self.dataset[idx] for idx in sampled_indices]),
                                       recent_data], dim=0)
        return merged_recent, current_data

    def __len__(self):
        return len(self.dataset) - self.border1 - self.gap
import os

import math
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from util.timefeatures import time_features

warnings.filterwarnings('ignore')

def get_alldata(filename='electricity.csv', root_path='./'):
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
        if filename.startswith('wind'):
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='H')
    else:
        if filename.startswith('nyc'):
            import h5py
            x = h5py.File(path, 'r')
            data = list()
            for key in x.keys():
                data.append(x[key][:])
            ts = np.stack(data, axis=1)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            df['date'] = pd.date_range(start='2007-04-01', periods=len(df), freq='30T')
        elif filename.endswith('.npz'):
            ts = np.load(path)['data'].astype(np.float32)
            ts = ts.reshape((ts.shape[0], ts.shape[1] * ts.shape[2]))
            df = pd.DataFrame(ts)
            if filename == 'PeMSD4':
                df['date'] = pd.date_range(start='2017-07-01', periods=len(df), freq='5T')
            else:
                df['date'] = pd.date_range(start='2012-03-01', periods=len(df), freq='5T')
        elif filename.endswith('.h5'):
            df = pd.read_hdf(path)
            df['date'] = df.index.values
        elif filename.endswith('.txt'):
            df = pd.read_csv(path, header=None)
            df['date'] = pd.date_range(start='1/1/2007', periods=len(df), freq='10T')
        df = df[[df.columns[-1]] + list(df.columns[:-1])]
    return df

class DatasetBenchmark(Dataset):
    def __init__(self, root_path, data_path, flag='train', seq_len=None, pred_len=None, label_len=None,
                 scale=True, timeenc=0, freq='h', stride=1, borders=None,
                 ratio=None, target='OT', border=None, **kwargs):
        self.ratio = ratio if ratio is not None else (0.7, 0.2)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.timeenc = timeenc
        self.scale = scale
        self.stride = stride
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.border = border
        self.borders = borders

        self.data_path = data_path
        self.root_path = root_path
        self.__confirm_data__()

    def __confirm_data__(self):
        data, data_stamp, border1s, border2s = self.__read_data__()
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.border = (border1, border2)
        self.data_x = torch.tensor(data[border1:border2])
        self.data_y = torch.tensor(data[border1:border2])
        self.data_stamp = torch.tensor(data_stamp[:border2])
        self.n_var = self.data_x.shape[-1]

    def __read_data__(self):
        df_raw = get_alldata(self.data_path, self.root_path)
        if 'ett' in self.data_path.lower():
            if self.borders:
                border1s, border2s = self.borders
            else:
                if 'etth' in self.data_path.lower():
                    border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
                    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
                else:
                    border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
                    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * self.ratio[0])
            num_test = int(data_len * self.ratio[1])
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        cols = list(df_raw.columns)
        cols.remove('date')
        data = df_raw[cols].values

        if self.scale:
            self.scaler = StandardScaler()
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
        data = data.astype(np.float32)
        self.data = data
        self.all_data_stamp = data_stamp
        return data, data_stamp, border1s, border2s

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[index:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[index:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class CIDatasetBenchmark(DatasetBenchmark):
    def _getid(self, index):
        c_begin = index // len(self)   # select variable
        s_begin = (index % len(self))  # select start time
        return s_begin, c_begin

    def _getitem(self, s_begin, c_begin ):
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __getitem__(self, index):
        return self._getitem(*self._getid(index))

    def __len__(self):
        return self.n_var * (len(self.data_x) - self.seq_len - self.pred_len + 1)


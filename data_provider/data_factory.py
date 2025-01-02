
from data_provider.data_loader import *
from torch.utils.data import DataLoader, DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ETTh1_CI': Dataset_ETT_hour_CI,
    'ETTh2_CI': Dataset_ETT_hour_CI,
    'ETTm1_CI': Dataset_ETT_minute_CI,
    'ETTm2_CI': Dataset_ETT_minute_CI,
    'custom_CI': Dataset_Custom_CI,
}

flag2num = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}


def get_dataset(args, flag, device='cpu', wrap_class=None, borders=None, take_post=0, take_pre=False, noise=0, **kwargs):
    if not hasattr(args, 'timeenc'):
        args.timeenc = 0 if not hasattr(args, 'embed') or args.embed != 'timeF' else 1
    if wrap_class is not None:
        if not isinstance(wrap_class, list):
            wrap_class = [wrap_class]
    border = None
    if borders is not None:
        if flag == 'pred' and len(borders[0]) == 3:
            flag = 'train'
            border = (borders[0][flag2num['train']], borders[1][flag2num['test']])
        else:
            border = (borders[0][flag2num[flag]], borders[1][flag2num[flag]])
        if flag != 'train':
            if Dataset_Recent in wrap_class or take_pre > 0:
                if take_pre > 1:
                    start = border[0] - take_pre
                else:
                    start = border[0] - args.pred_len
                border = (max(0, start), border[1])
        if take_post > 0:
            border = (border[0], border[1] + take_post)
    data_set = data_dict[args.data](
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=args.timeenc,
        freq=args.freq,
        train_only=args.train_only,
        border=border,
        borders=args.borders if hasattr(args, 'borders') else None,
        ratio=args.ratio if hasattr(args, 'ratio') else None
    )
    if args.pin_gpu and hasattr(data_set, 'data_x'):
        data_set.data_x = torch.tensor(data_set.data_x, dtype=torch.float32, device=device)
        data_set.data_y = torch.tensor(data_set.data_y, dtype=torch.float32, device=device)
        from settings import need_x_mark, need_x_y_mark
        if args.model in need_x_mark or args.model in need_x_y_mark or args.use_time or \
                hasattr(args, 'online_method') and args.online_method == 'OneNet':
            data_set.data_stamp = torch.tensor(data_set.data_stamp, dtype=torch.float32, device=device)

    if noise:
        print("Modify time series with strength =", noise)
        for i in range(3, len(data_set.data_y)):
            data_set.data_x[i] += 0.01 * (i // noise) * (data_set.data_x[i-1] - data_set.data_x[i-2])
            data_set.data_y[i] += 0.01 * (i // noise) * (data_set.data_y[i-1] - data_set.data_y[i-2])

    if wrap_class is not None:
        for cls in wrap_class:
            data_set = cls(data_set, **kwargs)
    print(flag, len(data_set))
    return data_set


def get_dataloader(data_set, args, flag, sampler=None):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'online':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag and args.local_rank == -1,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=False,
        sampler=sampler if args.local_rank == -1 or flag in ['online', 'test'] else DistributedSampler(data_set))
    return data_loader


def data_provider(args, flag, device='cpu', wrap_class=None, sampler=None, **kwargs):
    data_set = get_dataset(args, flag, device, wrap_class=wrap_class, **kwargs)
    data_loader = get_dataloader(data_set, args, flag, sampler)
    return data_set, data_loader

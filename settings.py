
need_x_y_mark = ['Autoformer', 'Transformer', 'Informer']
need_x_mark = ['TCN', 'FSNet', 'OneNet', 'iTransformer']
need_x_mark += [name + '_Ensemble' for name in need_x_mark]
no_extra_param = ['Online', 'ER', 'DERpp']
peft_methods = ['lora', 'adapter', 'ssf', 'mam_adapter', 'basic_tuning']

data_settings = {
    'wind_N2': {'data': 'wind_N2.csv', 'T':'FR51', 'M':[254, 254], 'prefetch_batch_size': 16},
    'wind': {'data': 'wind.csv', 'T':'UK', 'M':[28,28], 'prefetch_batch_size': 64},
    'ECL':{'data':'electricity.csv','T':'OT','M':[321,321],'S':[1,1],'MS':[321,1], 'prefetch_batch_size': 10, "period": 24},
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128, "period": 24},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128, "period": 24},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128, "period": 96},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128, "period": 96},
    'Solar':{'data':'solar_AL.txt','T': 136,'M':[137,137],'S':[1,1],'MS':[137,1], 'prefetch_batch_size': 32},
    'Weather':{'data':'weather.csv','T':'OT','M':[21,21],'S':[1,1],'MS':[21,1], 'prefetch_batch_size': 64, 'period': 24 * 6},
    'WTH':{'data':'WTH.csv','T':'OT','M':[12,12],'S':[1,1],'MS':[12,1], 'prefetch_batch_size': 64},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862], 'prefetch_batch_size': 2, 'period': 24},
    'METR_LA': {'data':'metr-la.csv','T': '773869','M':[207,207],'S':[1,1],'MS':[207,1], 'prefetch_batch_size': 16},
    'PEMS_BAY': {'data':'pems-bay.csv','T': 400001,'M':[325,325],'S':[1,1],'MS':[325,1], 'prefetch_batch_size': 10},
    'NYC_BIKE': {'data':'nyc-bike.h5','T': 0,'M':[500,500],'S':[1,1],'MS':[500,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'NYC_TAXI': {'data':'nyc-taxi.h5','T': 0,'M':[532,532],'S':[1,1],'MS':[532,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'PeMSD4': {'data':'PeMSD4/PeMSD4.npz','T': 0,'M':[921,921],'S':[1,1],'MS':[921,1], 'prefetch_batch_size': 2, 'feat_dim': 3},
    'PeMSD8': {'data':'PeMSD8/PeMSD8.npz','T': 0,'M':[510,510],'S':[1,1],'MS':[510,1], 'prefetch_batch_size': 6, 'feat_dim': 3},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Illness': {'data': 'illness.csv', 'T':'OT', 'M':[7,7], 'prefetch_batch_size': 128},
}

def get_borders(args):
    if args.border_type == 'online':
        if args.data.startswith('ETTh'):
            border1s = [0, 4*30*24 - args.seq_len, 5*30*24 - args.seq_len]
            border2s = [4*30*24, 5*30*24, 20*30*24]
            args.borders = (border1s, border2s)
        elif args.data.startswith('ETTm'):
            border1s = [0, 4*30*24*4 - args.seq_len, 5*30*24*4 - args.seq_len]
            border2s = [4*30*24*4, 5*30*24*4, 20*30*24*4]
            args.borders = (border1s, border2s)
        else:
            args.ratio = (0.2, 0.75)

hyperparams = {
    'PatchTST': {'e_layers': 3},
    'MTGNN': {},
    'LightCTS': {},
    'Crossformer': {'lradj': 'Crossformer', 'e_layers': 3, 'seg_len': 24, 'd_ff': 512, 'd_model': 256, 'n_heads': 4, 'dropout': 0.2},
    'DLinear': {},
    'GPT4TS': {'e_layers': 3, 'd_model': 768, 'n_heads': 4, 'd_ff': 768, 'dropout': 0.3},
    'iTransformer': {'e_layers': 3, 'd_model': 512, 'd_ff': 512, 'activation': 'gelu', 'timeenc': 1, 'patience': 3, },
    'Autoformer': {'timeenc': 1},
    'Informer': {'timeenc': 1},
}

def get_hyperparams(data, model, args, reduce_bs=True):
    hyperparam: dict = hyperparams[model]
    if model == 'iTransformer':
        if data == 'Traffic':
            hyperparam['e_layers'] = 4
        elif 'ETT' in data:
            hyperparam['e_layers'] = 2
            if data == 'ETTh1':
                hyperparam['d_model'] = 256
                hyperparam['d_ff'] = 256
            else:
                hyperparam['d_model'] = 128
                hyperparam['d_ff'] = 128

    if model == 'PatchTST':
        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'
        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})

    elif model == 'Crossformer':
        if data == 'ECL' or args.lradj == 'fixed':
            hyperparam['lradj'] = 'fixed'
        if reduce_bs:
            if data in ['PeMSD4']:
                hyperparam['batch_size'] = 4
            elif data in ['Traffic']:
                hyperparam['batch_size'] = 4
            elif data in ['NYC_BIKE', 'NYC_TAXI', 'PeMSD8']:
                hyperparam['batch_size'] = 8
        else:
            if data in ['Traffic', 'PeMSD4'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 24
            if data in ['PeMSD8'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 16

        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Illness', 'wind', 'Exchange']:
            hyperparam['d_model'] = 256
            hyperparam['n_heads'] = 4
        else:
            hyperparam['d_model'] = 64
            hyperparam['n_heads'] = 2

        if data in ['Traffic', 'ECL']:
            hyperparam['d_ff'] = 128

        if data in ['Illness']:
            hyperparam['e_layers'] = 2

    elif model == 'GPT4TS':
        if data == 'ETTh1':
            hyperparam['lradj'] = 'typy4'
            hyperparam['tmax'] = 20
            # hyperparam['label_len'] = 168
        elif data == 'ETTh2':
            hyperparam['dropout'] = 1
            hyperparam['tmax'] = 20
            # hyperparam['label_len'] = 168
        elif data == 'Traffic':
            hyperparam['dropout'] = 0.3
        elif data == 'ECL':
            hyperparam['tmax'] = 10
        elif data == 'Illness':
            hyperparam['patch_size'] = 24
            # hyperparam['label_len'] = 18
            hyperparam['batch_size'] = 16

        if data in ['ETTm1', 'ETTm2', 'ECL', 'Traffic', 'Weather', 'WTH']:
            hyperparam['seq_len'] = 512

        if data.startswith('ETTm'):
            hyperparam['stride'] = 16
        elif args.seq_len == 104:
            hyperparam['stride'] = 2

    return hyperparam


pretrain_lr_online_dict = {
     'TCN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.001, 'Traffic': 0.003},
     'TCN_RevIN': {'ECL': 0.003, 'ETTh2': 0.001, 'ETTm1': 0.0001, 'Weather': 0.001, 'Traffic': 0.003},
     'TCN_Ensemble': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.0003, 'Weather': 0.001, 'Traffic': 0.003},
     'FSNet_RevIN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.003, 'Traffic': 0.003},
    'GPT4TS': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0001},
    'PatchTST': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.00001, 'ECL': 0.0005},
    'NLinear': {'ETTh2': 0.05, 'ETTm1': 0.05, 'Traffic': 0.005, 'Weather': 0.01, 'ECL': 0.01},
    'Informer_RevIN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Autoformer_RevIN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Informer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Autoformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001}
}

pretrain_lr_dict = {
    'PatchTST': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0005},
}


def drop_last_PatchTST(args):
    bs = 128 if args.dataset in ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Weather'] else 32
    test_num = args.borders[1][2] - args.borders[0][2] - args.seq_len - args.pred_len + 1
    args.borders[1][2] -= test_num % bs

import argparse
import datetime
import gc

import settings
from data_provider import dataloader_online

cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(cur_sec)

from pprint import pprint

import random

import exp as exps
from exp import *

from settings import data_settings


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

# basic config
parser.add_argument('--train_only', action='store_true', default=False,
                    help='perform training on full input dataset without validation and testing')
parser.add_argument('--wo_test', action='store_true', default=False, help='only valid, not test')
parser.add_argument('--wo_valid', action='store_true', default=False, help='only test')
parser.add_argument('--model_id', type=str, default='', help='model id')
parser.add_argument('--only_test', action='store_true', default=False)
parser.add_argument('--do_valid', action='store_true', default=False)
parser.add_argument('--model', type=str, required=True, default='PatchTST')
parser.add_argument('--override_hyper', action='store_true', default=True, help='Override hyperparams by setting.py')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model by Pytorch 2.0')
parser.add_argument('--reduce_bs', type=str_to_bool, default=False,
                    help='Override batch_size in hyperparams by setting.py')
parser.add_argument('--normalization', type=str, default=None)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--tag', type=str, default='')

# online
parser.add_argument('--online_method', type=str, default=None)
parser.add_argument('--skip', type=str, default=None)
parser.add_argument('--online_learning_rate', type=float, default=None)
parser.add_argument('--val_online_lr', action='store_true', default=True)
parser.add_argument('--diff_online_lr', action='store_true', default=False)
parser.add_argument('--save_opt', action='store_true', default=True)
parser.add_argument('--leakage', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--replay_num', type=int, default=0, help='')

# Meta
parser.add_argument('--task_num', type=int, default=1)

# Proceed
parser.add_argument('--act', type=str, default='sigmoid', help='activation')
parser.add_argument('--tune_mode', type=str, default='down_up')
parser.add_argument('--ema', type=float, default=0, help='')
parser.add_argument('--concept_dim', type=int, default=200)
parser.add_argument('--bottleneck_dim', type=int, default=32, help='')
parser.add_argument('--individual_generator', action='store_true', default=False)
parser.add_argument('--share_encoder', action='store_true', default=False)
parser.add_argument('--use_mean', type=str_to_bool, default=True)
parser.add_argument('--joint_update_valid', action='store_true', default=False)
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--wo_clip', action='store_true', default=False)

# OneNet
parser.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')

# data loader
parser.add_argument('--border_type', type=str, default='online', help='set any other value for traditional data splits')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--dataset', type=str, default='ETTh1', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--wrap_data_class', type=list, default=[])
parser.add_argument('--pin_gpu', type=str_to_bool, default=True)

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DLinear
parser.add_argument('--individual', action='store_true', default=False,
                    help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--drop_last', action='store_true', default=False)

# Formers
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--output_enc', action='store_true', help='whether to output embedding from encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# Crossformer
parser.add_argument('--seg_len', type=int, default=24, help='segment length (L_seg)')
parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
parser.add_argument('--num_routers', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

# iTransformer
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')

# MTGNN
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--in_dim', type=int, default=1)

# GPT4TS
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--tmax', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=16)

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--begin_valid_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--macro_batch_size', type=int, default=0, help='')
parser.add_argument('--grad_accumulation', type=int, default=1)
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--warmup_epochs', type=int, default=5)

# GPU
parser.add_argument('--use_gpu', type=str_to_bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

# SOLID
parser.add_argument('--test_train_num', type=int, default=500)
parser.add_argument('--selected_data_num', type=int, default=5)
parser.add_argument('--lambda_period', type=float, default=0.1)
parser.add_argument('--whole_model', action='store_true')
parser.add_argument('--continual', action='store_true')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.model.endswith('_Ensemble') and 'TCN' not in args.model and 'FSNet' not in args.model:
    args.model = args.model[:-len('_Ensemble')]
    args.ensemble = True
else:
    args.ensemble = False

import platform

if platform.system() == 'Windows':
    torch.cuda.set_per_process_memory_fraction(48 / 61, 0)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.enc_in, args.c_out = data_settings[args.dataset][args.features]
args.period = data_settings[args.dataset].get('period', 1)
args.data_path = data_settings[args.dataset]['data']
args.dec_in = args.enc_in
if args.model.endswith('_leak'):
    args.model = args.model[:-len('_leak')]
    args.leakage = True
if args.online_method and args.online_method.endswith('_leak'):
    args.online_method = args.online_method[:-len('_leak')]
    args.leakage = True

if args.tag and args.tag[0] != '_':
    args.tag = '_' + args.tag

args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'
if args.model.startswith('GPT4TS'):
    if not args.online_method and not args.do_predict:
        args.data += '_CI'
    else:
        if args.dataset == 'ECL':
            args.batch_size = min(args.batch_size, 3)
        elif args.dataset == 'Traffic':
            args.batch_size = 1
if hasattr(args, 'border_type'):
    settings.get_borders(args)

Exp = Exp_Main

extra_model_id = args.model_id
args.model_id = f'{args.dataset}_{args.seq_len}_{args.pred_len}_{args.model}'
if args.normalization is not None:
    args.model_id += '_' + args.normalization

if args.border_type == 'online':
    args.patience = min(args.patience, 3)

if args.model in settings.need_x_mark:
    # args.optim = 'AdamW' if args.optim != 'AdamW' and args.online_method.lower() == 'Concept_Tune' else args.optim
    args.optim = 'AdamW'
    args.patience = 3

if args.online_method:
    args.train_epochs = min(args.train_epochs, 10)
    args.save_opt = True
    if 'FSNet' in args.model and args.online_method == 'Online':
        args.online_method = 'FSNet'
    if args.online_method == 'FSNet' and 'TCN' in args.model:
        args.model = args.model.replace('TCN', 'FSNet')

    if args.online_method == 'Online':
        args.pretrain = True
        args.only_test = True
    elif args.online_method == 'Online_Meta':
        args.pretrain = True
        args.optim = 'Adam'

    if 'FSNet' in args.model:
        args.pretrain = False
    elif args.online_method.lower() in settings.peft_methods:
        args.pretrain = True
        args.freeze = True

    Exp = getattr(exps, 'Exp_' + args.online_method)

    if args.online_method == 'SOLID':
        args.pretrain = True
        args.only_test = True
        args.online_method = 'Online'
        if not args.whole_model:
            args.freeze = True

args.timeenc = 2
if args.override_hyper and args.model in settings.hyperparams:
    for k, v in settings.get_hyperparams(args.dataset, args.model, args, args.reduce_bs).items():
        args.__setattr__(k, v)

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    args.gpu = args.local_rank
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.num_gpus = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.num_gpus

if args.model in ['MTGNN']:
    if 'feat_dim' in data_settings[args.dataset]:
        args.in_dim = data_settings[args.dataset]['feat_dim']
        args.enc_in = int(args.enc_in / args.in_dim)
        if args.features == 'M':
            args.c_out = int(args.c_out / args.in_dim)


args.find_unused_parameters = args.model in ['MTGNN']

data_name = args.data_path.split("/")[-1].split(".")[0]
if platform.system() != 'Windows':
    path = './'
else:
    path = 'D:/data/'
    if args.checkpoints:
        args.checkpoints = 'D:/checkpoints/'

if args.online_method:
    flag = args.online_method.lower()
    if not args.border_type:
        if args.online_method == 'Online':
            flag = args.data
            args.checkpoints = ""
        else:
            flag = args.data + '_' + flag

    if flag == 'fsnet':
        flag = 'online'

    if 'proceed' in flag:
        if not args.freeze:
            flag += "_fulltune"
        if not args.pretrain:
            flag += "_new"
        flag += f"_{args.lradj}"
        flag += f'_{args.tune_mode}_btl{args.bottleneck_dim}_ema{args.ema}'
        if args.concept_dim:
            flag += f'_mid{args.concept_dim}'
        if not args.individual_generator:
            flag += '_share'
        if args.share_encoder:
            flag += '_share_enc'
        if args.wo_clip:
            flag += '_noclip'
else:
    flag = args.border_type if args.border_type else args.data

print('Args in experiment:')
print(args)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # args = get_args()
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None

    all_results = {'mse': [], 'mae': []}
    for ii in range(args.itr):
        if ii == 0 and args.skip and os.path.exists(args.skip):
            if args.wo_test:
                continue
            with open(args.skip, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f.readlines():
                    if line.startswith('mse:'):
                        splits = line.split(',')
                        mse, mae = splits[0].split(':')[1], splits[1].split(':')[1]
                        all_results['mse'].append(float(mse))
                        all_results['mae'].append(float(mae))
                        break
            if len(all_results['mse']) > 0:
                continue
        if args.border_type:
            if args.model in ['PatchTST', 'iTransformer']:
                fix_seed = 2021 + ii
            else:
                fix_seed = 2023 + ii
        else:
            fix_seed = 2023 + ii if args.model == 'iTransformer' else 2021 + ii
        setup_seed(fix_seed)
        print('Seed:', fix_seed)

        if args.online_method == 'OneNet' and args.pretrain:
            fsnet_name = "FSNet_RevIN"
            args.fsnet_path = f'./checkpoints/{args.dataset}_60_{args.pred_len}_{fsnet_name}_' \
                              f'online_ftM_sl60_ll48_pl{args.pred_len}_lr{settings.pretrain_lr_online_dict[fsnet_name][args.dataset]}' \
                              f'_uniFalse_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_{ii}/checkpoint.pth'

        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            ((extra_model_id + "_") if extra_model_id else '') + args.model_id,
            flag,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        if args.pretrain:
            pretrain_lr = settings.pretrain_lr_online_dict[args.model + ("_RevIN" if args.normalization else "")][args.dataset] \
                if args.online_method else settings.pretrain_lr_dict[args.model][args.dataset]
            if not args.border_type and args.model == 'iTransformer' and args.dataset == 'Weather':
                pretrain_lr = 0.0001
            pretrain_setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_uniFalse_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.border_type if args.border_type else args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                pretrain_lr,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            args.pred_path = os.path.join('./results/', pretrain_setting, 'real_prediction.npy')
            if platform.system() == 'Windows':
                args.load_path = os.path.join('D://checkpoints/', pretrain_setting, 'checkpoint.pth')
            else:
                args.load_path = os.path.join('./checkpoints/', pretrain_setting, 'checkpoint.pth')
        exp = Exp(args)  # set experiments

        if train_data is None:
            train_data, train_loader = exp._get_data('train')
        if not hasattr(args, 'borders'):
            args.borders = train_data.borders
            if args.border_type != 'online' and args.model == 'PatchTST':
                settings.drop_last_PatchTST(args) # SOLID dropout the last when data split = 7:2:1
        if exp.args.wrap_data_class:
            exp.args.wrap_data_class[-1] = partial(exp.args.wrap_data_class[-1], borders=args.borders)

        path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        if args.online_method not in ['Online', 'SOLID', 'ER', 'DERpp']:
            print('Checkpoints in', path)
            if (args.only_test or args.do_valid) and os.path.exists(path):
                print('Loading', path)
                exp.load_checkpoint(path)
                print('Learning rate of model_optim is', exp.model_optim.param_groups[0]['lr'])
            else:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader,
                                                                                vali_data, vali_loader)
                torch.cuda.empty_cache()

        if args.online_learning_rate is not None and not isinstance(exp, Exp_SOLID):
            for j in range(len(exp.model_optim.param_groups)):
                exp.model_optim.param_groups[j]['lr'] = args.online_learning_rate
            print('Adjust learning rate of model_optim to', exp.model_optim.param_groups[0]['lr'])

        if args.do_valid and args.online_method and args.local_rank <= 0:
            assert isinstance(exp, Exp_Online)
            mse, mae = exp.online(online_data=vali_data if isinstance(vali_data, Dataset_Recent_Full_Feedback) else None,
                                  phase='val', show_progress=True)[:2]
            print('Best Valid MSE:', mse)
            all_results['mse'].append(mse)
            all_results['mae'].append(mae)
            continue

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            setup_seed(fix_seed)
            mse, mae = exp.predict(path, setting, True)[:2]
            all_results['mse'].append(mse)
            all_results['mae'].append(mae)
        elif not args.wo_test and not args.train_only and args.local_rank <= 0:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if isinstance(exp, Exp_Online):
                setup_seed(fix_seed)
                if not isinstance(exp, Exp_SOLID) and not args.wo_valid:
                    vali_data = None
                    torch.cuda.empty_cache()
                    gc.collect()
                    exp.update_valid()
                mse, mae, test_data = exp.online(test_data)
            else:
                mse, mae, test_data, test_loader = exp.test(setting, test_data, test_loader)
            all_results['mse'].append(mse)
            all_results['mae'].append(mae)
        torch.cuda.empty_cache()

    for k in all_results.keys():
        all_results[k] = np.array(all_results[k])
        all_results[k] = [all_results[k].mean(), all_results[k].std()]
    pprint(all_results)
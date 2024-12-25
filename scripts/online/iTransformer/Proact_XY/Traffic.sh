if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=Traffic
model_name=iTransformer
online_method=Proceed
ema=0
tune_mode=down_up
learning_rate=0.001
mid=150
btl=32
for pred_len in 24
do
for online_learning_rate in 0.00001
do
  suffix='_lr'$learning_rate'_onlinelr'$online_learning_rate
  filename=logs/online/$model_name'_'$online_method'_mid'$mid'_share_fulltune_'$tune_mode'_'$data'_'$pred_len'_btl'$btl$suffix.log
  python -u run.py \
    --dataset $data --border_type 'online' --batch_size 8 \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --tune_mode $tune_mode \
    --concept_dim $mid \
    --val_online_lr --diff_online_lr \
    --itr 3 --skip $filename --pretrain  \
    --save_opt --online_method $online_method \
    --bottleneck_dim $btl \
    --online_learning_rate $online_learning_rate --only_test \
    --learning_rate $learning_rate --lradj type3 >> $filename 2>&1
done
done
for pred_len in 48 96
do
for online_learning_rate in 0.000003
do
  suffix='_lr'$learning_rate'_onlinelr'$online_learning_rate
  filename=logs/online/$model_name'_'$online_method'_mid'$mid'_share_fulltune_'$tune_mode'_'$data'_'$pred_len'_btl'$btl$suffix.log
  python -u run.py \
    --dataset $data --border_type 'online' --batch_size 8 \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --tune_mode $tune_mode \
    --concept_dim $mid \
    --val_online_lr --diff_online_lr \
    --itr 3 --skip $filename --pretrain  \
    --save_opt --online_method $online_method \
    --bottleneck_dim $btl \
    --online_learning_rate $online_learning_rate --only_test \
    --learning_rate $learning_rate --lradj type3 >> $filename 2>&1
done
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=ETTm1
model_name=iTransformer
online_method=OneNet

for pred_len in 24
do
for learning_rate in 0.00003
do
for online_learning_rate in 0.000003
do
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --val_online_lr \
    --save_opt --only_test \
    --online_learning_rate $online_learning_rate --learning_rate $learning_rate >> $filename 2>&1
done
done
done
for pred_len in 48
do
for learning_rate in 0.00003
do
for online_learning_rate in 0.0000003
do
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --val_online_lr \
    --save_opt --only_test \
    --online_learning_rate $online_learning_rate --learning_rate $learning_rate >> $filename 2>&1
done
done
done
for pred_len in 96
do
for learning_rate in 0.00001
do
for online_learning_rate in 0.000003
do
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --val_online_lr \
    --save_opt --only_test \
    --online_learning_rate $online_learning_rate --learning_rate $learning_rate >> $filename 2>&1
done
done
done
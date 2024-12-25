if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=60
data=ETTm1
model_name=TCN
train_epochs=100

for pred_len in 24 48 96
do
for learning_rate in 0.0001
do
  filename=logs/online/$model_name'_RevIN_'$data'_'$pred_len'_lr'$learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --only_test \
    --pin_gpu True --reduce_bs False \
    --save_opt \
    --learning_rate $learning_rate >> $filename 2>&1
done
done
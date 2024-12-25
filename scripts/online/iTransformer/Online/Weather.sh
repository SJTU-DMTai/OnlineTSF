if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=Weather
model_name=iTransformer
online_method=Online

for pred_len in 24 48 96
do
for online_learning_rate in 0.0000003
do
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
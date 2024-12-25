if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=Traffic
model_name=PatchTST
online_method=Online

for pred_len in 24 48 96
do
for online_learning_rate in 0.0000003
do
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --online_method $online_method \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test \
    --online_learning_rate $online_learning_rate > logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_onlinelr'$online_learning_rate.log2 2>&1
done
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=60
data=Traffic
model_name=FSNet_Ensemble_leak
online_method=OneNet

for pred_len in 24 48 96
do
for learning_rate in 0.003
do
for online_learning_rate in 0.0001
do
  filename=logs/online/$model_name'_RevIN_'$data'_'$pred_len'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --online_method $online_method \
    --only_test --online_learning_rate $online_learning_rate \
    --save_opt \
    --skip $filename \
    --learning_rate $learning_rate >> $filename 2>&1
done
done
done
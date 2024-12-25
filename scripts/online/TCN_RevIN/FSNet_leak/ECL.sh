if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=60
data=ECL
model_name=FSNet_leak

for pred_len in 24
do
for learning_rate in 0.003
do
for online_learning_rate in 0.00003
do
  filename=logs/online/$model_name'_RevIN_'$data'_'$pred_len'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename \
    --only_test --online_learning_rate $online_learning_rate \
    --save_opt --online_method Online \
    --learning_rate $learning_rate >> $filename 2>&1
done
done
done
for pred_len in 48 96
do
for learning_rate in 0.003
do
for online_learning_rate in 0.00003
do
  filename=logs/online/$model_name'_RevIN_'$data'_'$pred_len'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 1 --skip $filename \
    --only_test --online_learning_rate $online_learning_rate \
    --save_opt --online_method Online \
    --learning_rate $learning_rate >> $filename 2>&1
done
done
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=60
data=Traffic
model_name=TCN
online_method=SOLID

for test_train_num in 500
do
for selected_data_num in 5
do
for pred_len in 24 48 96
do
for online_learning_rate in 0.00003
do
  filename=logs/online/$model_name'_RevIN_'$online_method'whole_cont_'$data'_'$pred_len'_N'$test_train_num'_n'$selected_data_num'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --normalization RevIN \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --skip $filename --checkpoints "" --online_method $online_method --whole_model --continual --pretrain \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test --test_train_num $test_train_num --selected_data_num $selected_data_num \
    --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
done
done
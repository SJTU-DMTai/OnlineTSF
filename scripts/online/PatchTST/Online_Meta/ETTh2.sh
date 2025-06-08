if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=336
data=ETTh2
model_name=PatchTST
online_method=Online_Meta

for pred_len in 24
do
for online_learning_rate in 1e-4 1e-5 1e-6
do
for learning_rate in 1e-3 1e-4 1e-5
do
for replay_num in 5 10 20
do
for task_num in 1 4 8
do
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_replay'$replay_num'_task'$task_num'_lr'$learning_rate'_onlinelr'$online_learning_rate.log
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name --model_id 'onlinelr'$online_learning_rate'_replay'$replay_num'_task'$task_num'_' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --replay_num $replay_num \
    --itr 3 --skip $filename --online_method $online_method \
    --pin_gpu True --reduce_bs False \
    --save_opt --only_test --batch_size $task_num \
    --learning_rate $learning_rate --online_learning_rate $online_learning_rate >> $filename 2>&1
done
done
done
done
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

seq_len=512
data=ECL
model_name=iTransformer

for pred_len in 24 48 96
do
for learning_rate in 0.0005
do
  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --itr 3 --only_test \
    --save_opt \
    --batch_size 16 \
    --learning_rate $learning_rate > logs/online/$model_name'_'$data'_'$pred_len'_lr'$learning_rate.log 2>&1
done
done
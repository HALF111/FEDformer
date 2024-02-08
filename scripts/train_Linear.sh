
gpu_num=0

dir_name=all_result

seq_len=336
# model=DLinear
model=Linear


for model in Linear DLinear NLinear
do

seq_len=336
for pred_len in 96 192 336 720
do

# 1. ETTh1
name=ETTh1
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

# 2.ETTh2
name=ETTh2
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --task_id ETTh2 \
  --model $model \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

# 3.ETTm1
name=ETTm1
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --task_id ETTm1 \
  --model $model \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

# 4.ETTm2
name=ETTm2
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

# 5.electricity
name=ECL
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --task_id ECL \
  --model $model \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

# 6.exchange
name=Exchange
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --task_id Exchange \
  --model $model \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0005 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log


# 7.traffic
name=Traffic
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --task_id traffic \
  --model $model \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log


# 8.weather
name=Weather
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --task_id weather \
  --model $model \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.0001 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

done


seq_len=104
for pred_len in 24 36 48 60
do

# 9.illness
name=Illness
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --task_id ili \
  --model $model \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.01 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log

done

done
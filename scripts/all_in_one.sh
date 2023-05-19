# gpu_num=0
gpu_num=1
model_name=FEDformer

# # 1.ETTh1数据集
# # 1.1 pred_len=24
# # for pred_len in 24 96 192 336 720
# for pred_len in 96 192 336 720
# # for pred_len in 336 720
# do
# name=ETTh1
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --task_id ETTh1 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# for test_train_num in 1 5 10 15 20
# do
# for adapted_lr_times in 2 5 10 20 50 100
# do
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --task_id ETTh1 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num \
#   --test_train_num $test_train_num \
#   --adapted_lr_times $adapted_lr_times \
#   > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# done
# done
# done



# # 4.ETTm2数据集
# # 4.1 pred_len=24
# # for pred_len in 24 96 192 336 720
# # for pred_len in 96 192 336 720
# for pred_len in 336 720
# # for pred_len in 336 720
# do
# name=ETTm2
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --task_id ETTm2 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# for test_train_num in 1 5 10 15 20
# do
# for adapted_lr_times in 2 5 10 20 50 100
# do
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --task_id ETTm2 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $pred_len \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu_num \
#   --test_train_num $test_train_num \
#   --adapted_lr_times $adapted_lr_times \
#   > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# done
# done
# done


# # 5.Electricty数据集
# # 5.1 pred_len=24
# for pred_len in 96 192 336 720
# # for pred_len in 336 720
# do
# name=ECL
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --task_id ECL \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# for test_train_num in 1 5 10 15 20
# do
# # for adapted_lr_times in 2 5 10 20 50 100
# for adapted_lr_times in 1000 2000 5000 10000 12000 15000 20000
# do
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log

# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/electricity/ \
#  --data_path electricity.csv \
#  --task_id ECL \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --test_train_num $test_train_num \
#   --adapted_lr_times $adapted_lr_times \
#   > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# done
# done
# done



# # 6.Exchange数据集
# # 6.1 pred_len=24
# # for pred_len in 24 96 192 336 720
# for pred_len in 96 192 336 720
# # for pred_len in 336 720
# do
# name=Exchange
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --task_id Exchange \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# for test_train_num in 1 5 10 15 20 30
# do
# for adapted_lr_times in 50 100 200 300 500 1000
# do
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --task_id Exchange \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 8 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --test_train_num $test_train_num \
#   --adapted_lr_times $adapted_lr_times \
#   > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# done
# done
# done



# # 7.Traffic数据集
# # 7.1 pred_len=24
# # for pred_len in 24 96 192 336 720
# for pred_len in 96 192 336 720
# # for pred_len in 336 720
# do
# name=Traffic
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --task_id traffic \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --itr 1 \
#  --train_epochs 3 \
#   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# for test_train_num in 1 5 10 15 20
# do
# for adapted_lr_times in 2000 5000 10000 12000 15000 20000
# do
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --task_id traffic \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --itr 1 \
#  --train_epochs 3 \
#   --gpu $gpu_num \
#   --test_train_num $test_train_num \
#   --adapted_lr_times $adapted_lr_times \
#   > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# done
# done
# done



# # 8.Weather数据集
# # 8.1 pred_len=24
# # for pred_len in 24 96 192 336 720
# for pred_len in 96 192 336 720
# # for pred_len in 336 720
# do
# name=Weather
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --task_id weather \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

# for test_train_num in 1 5 10 15 20
# do
# for adapted_lr_times in 5 10 20 50 100
# do
# # python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --task_id weather \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num \
#   --test_train_num $test_train_num \
#   --adapted_lr_times $adapted_lr_times \
#   > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
# done
# done
# done


# 9.Illness数据集
# 9.1 pred_len=24
# for pred_len in 24 96 192 336 720
# for pred_len in 96 192 336 720
for pred_len in 24 36 48 60
# for pred_len in 336 720
do
name=Illness
cur_path=./all_result/$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
# run train and test first
# python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model_name \
 --data custom \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 1 \
  --gpu $gpu_num --run_train --run_test > $cur_path'/'train_and_test_loss.log

for test_train_num in 1 5 10 15 20
do
for adapted_lr_times in 2 5 10 20 50 100
do
# python -u run.py   --is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_$pred_len   --model Autoformer   --data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len $pred_len   --e_layers 2   --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1   --gpu $gpu_num --test_train_num $test_train_num --adapted_lr_times $adapted_lr_times > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model_name \
 --data custom \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 1 \
  --gpu $gpu_num \
  --test_train_num $test_train_num \
  --adapted_lr_times $adapted_lr_times \
  > $cur_path'/'ttn$test_train_num'_'lr$adapted_lr_times'_'batch32.log
done
done
done

gpu_num=0
# gpu_num=1
model_name=FEDformer


# # 5.Electricty数据集
# # 5.1 pred_len=24
# # for pred_len in 96 192 336 720
# for pred_len in 720
# # for pred_len in 336 720
# do
# name=ECL
# cur_path=./all_result/$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # run train and test first

# # for test_train_num in 1 5 10 15 20
# for test_train_num in 30
# do
# # for adapted_lr_times in 2 5 10 20 50 100
# # for adapted_lr_times in 2000 5000 10000 12000 15000 20000
# for adapted_lr_times in 20000
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


# cur_path=./all_result/ECL_pl720
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
#  --pred_len 720 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 321 \
#  --dec_in 321 \
#  --c_out 321 \
#  --des 'Exp' \
#  --itr 1 \
#   --gpu $gpu_num --run_test > $cur_path'/'train_and_test_loss.log


# Traffic pred_len=96
cur_path=./all_result/Traffic_pl96
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model_name \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 96 \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 3 \
  --gpu $gpu_num --run_test > $cur_path'/'train_and_test_loss.log
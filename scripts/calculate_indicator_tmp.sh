gpu_num=0

dir=get_data_error_logs


# for model in FEDformer
# do
# for pred_len in 96
# do

# # 2.ETTh1
# name=ETTh1
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# # python -u run.py \
# #   --is_training 1 \
# #   --root_path ./dataset/ETT-small/ \
# #   --data_path ETTh1.csv \
# #   --task_id ETTh1 \
# #   --model $model \
# #   --data ETTh1 \
# #   --features M \
# #   --seq_len 96 \
# #   --label_len 48 \
# #   --pred_len $pred_len \
# #   --e_layers 2 \
# #   --d_layers 1 \
# #   --factor 3 \
# #   --enc_in 7 \
# #   --dec_in 7 \
# #   --c_out 7 \
# #   --des 'Exp' \
# #   --d_model 512 \
# #   --itr 1 \
# #   --gpu $gpu_num \
# #   --run_train --run_test \
# #   > $cur_path'/'train_and_test_loss.log
# # 计算残差
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --task_id ETTh1 \
#   --model $model \
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
#   --get_data_error --batch_size 1 \
# #   > $cur_path'/'get_data_error.log

# done

# done


model=FEDformer
pred_len=24

# 计算残差
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model \
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
  --get_data_error --batch_size 1 \
#   > $cur_path'/'get_data_error.log
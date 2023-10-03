gpu_num=0

dir_name=adapt_whole_model

model=Crossformer

# Traffic


# ETTh1
name=ETTh1
# 1.1 ETTh1 + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M \
  --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 \
  --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 \
  --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 200 \
  --gpu $gpu_num --adapt_whole_model > $dir_name/$name'_'$model'_'pl$pred_len.log

# 1.2 ETTh1 + 192
seq_len=720; pred_len=192
seg_len=24; lr=1e-5
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M \
  --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
  --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 \
  --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 200 \
  --gpu $gpu_num --adapt_whole_model > $dir_name/$name'_'$model'_'pl$pred_len.log


# # illness
# name=Illness
# for pred_len in 24 36
# do
# seq_len=48
# python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M \
#   --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 \
#   --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 \
#   --gpu $gpu_num --adapt_whole_model \
#   > $dir_name/$name'_'$model'_'pl$pred_len.log
# done

# for pred_len in 48 60
# do
# seq_len=60
# python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M \
#   --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 \
#   --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 \
#   --gpu $gpu_num --adapt_whole_model \
#   > $dir_name/$name'_'$model'_'pl$pred_len.log
# done
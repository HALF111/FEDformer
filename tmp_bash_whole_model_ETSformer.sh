gpu_num=1

dir_name=adapt_whole_model

model=ETSformer

# Traffic
name=Traffic
pred_len=96
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M \
  --seq_len 336 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' \
  --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 200 \
  --gpu $gpu_num --adapt_whole_model > $dir_name/$name'_'$model'_'pl$pred_len.log

pred_len=192
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M \
  --seq_len 336 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' \
  --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 200 \
  --gpu $gpu_num --adapt_whole_model > $dir_name/$name'_'$model'_'pl$pred_len.log


# ETTh1
name=ETTh1
for pred_len in 96 192
do
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model ETSformer --data ETTh1 --features M \
  --seq_len 96 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 \
  --d_model 512 --is_training 1 --task_id ETTh1 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' \
  --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 10 \
  --gpu $gpu_num --adapt_whole_model > $dir_name/$name'_'$model'_'pl$pred_len.log
done


# # illness
# name=Illness
# for pred_len in 24 36 48 60
# do
# python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model ETSformer --data custom --features M \
#   --seq_len 60 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' \
#   --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 0.5 \
#   --gpu $gpu_num --adapt_whole_model \
#   > $dir_name/$name'_'$model'_'pl$pred_len.log
# done
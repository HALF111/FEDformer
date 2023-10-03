gpu_num=0

dir_name=adapt_whole_model

for model in FEDformer Autoformer Informer
do

for pred_len in 96 192
# for pred_len in 192
do
# ETTh1
name=ETTh1
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 \
  --model $model --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
  --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1 \
  --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 1 \
  --gpu $gpu_num --adapt_whole_model \
  > $dir_name/$name'_'$model'_'pl$pred_len.log

# # electricity
# name=ECL
# python -u run.py --is_training 1 \--root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL \
#  --model $model --data custom --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
#  --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 \
#   --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 1000 \
#   --gpu $gpu_num --adapt_whole_model \
#   > $dir_name/$name'_'$model'_'pl$pred_len.log
done


for pred_len in 192
do
# traffic
name=Traffic
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic \
 --model $model --data custom --features M --seq_len 96 --label_len 48 --pred_len $pred_len \
 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 \
 --test_train_num 1000 --selected_data_num 10 --adapted_lr_times 200 \
  --gpu $gpu_num --adapt_whole_model \
  > $dir_name/$name'_'$model'_'pl$pred_len.log
done

# # for pred_len in 24 36 48 60
# for pred_len in 48 60
# do
# # illness
# name=Illness
# python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili \
#  --model $model --data custom --features M --seq_len 36 --label_len 18 --pred_len $pred_len \
#  --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 \
#  --test_train_num 200 --selected_data_num 3 --adapted_lr_times 50 \
#   --gpu $gpu_num --adapt_whole_model \
#   > $dir_name/$name'_'$model'_'pl$pred_len.log
# done

done
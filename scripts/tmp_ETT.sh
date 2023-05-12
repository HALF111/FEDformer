

model_name=FEDformer

test_train_num=10

gpu=0

for preLen in 96
do

# ETTh1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --task_id ETTh1 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1 \
  --gpu $gpu \
  --test_train_num $test_train_num # \
#   > logs/$model_name'_'M_ETTh1_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

# # ETTh2
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --task_id ETTh2 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu \
#   --test_train_num $test_train_num \
#   > logs/$model_name'_'M_ETTh2_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

# # ETT m1
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm1.csv \
#   --task_id ETTm1 \
#   --model $model_name \
#   --data ETTm1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu \
#   --test_train_num $test_train_num \
#   > logs/$model_name'_'M_ETTm1_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

# # ETTm2
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
#   --pred_len $preLen \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --itr 1 \
#   --gpu $gpu \
#   --test_train_num $test_train_num \
#   > logs/$model_name'_'M_ETTm2_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

done
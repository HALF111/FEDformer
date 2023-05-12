

model_name=FEDformer

test_train_num=10

gpu=1

for preLen in 96
do

# Exchange
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model_name \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 1 \
  --gpu $gpu \
  --test_train_num $test_train_num \
  > logs/$model_name'_'M_Exchange_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

done


model_name=FEDformer

test_train_num=10

gpu=1

for preLen in 24
do

# illness
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model_name \
 --data custom \
 --features S \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 1 \
  --gpu $gpu \
  --test_train_num $test_train_num \
  > logs/$model_name'_'M_ILI_predlen_$preLen'_'seqlen_96_labellen_48_ttn_$test_train_num.log

done
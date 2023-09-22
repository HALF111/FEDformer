gpu_num=1

dir_name=all_result

for model in FEDformer Autoformer Informer
do

for pred_len in 96 192 336 720
do
# weather
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
 --seq_len 96 \
 --label_len 48 \
 --pred_len $pred_len \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 1 \
  --gpu $gpu_num \
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log
done

for pred_len in 24 36 48 60
do
# illness
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
  --run_train --run_test \
  > $cur_path'/'train_and_test_loss.log
done

done
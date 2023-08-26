# ETSformer
gpu_num=1

model=ETSformer

dir_name=all_result


for pred_len in 96 192 336 720
do

# 1.1 ETTh1
name=ETTh1
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model $model --data ETTh1 --features M --seq_len 96 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id ETTh1 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 2.1 ETTm1
name=ETTm1
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model $model --data ETTm1 --features M --seq_len 96 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id ETTm1 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 3.1 ETTh2
name=ETTh2
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model $model --data ETTh2 --features M --seq_len 96 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id ETTh2 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 4.1 ETTm2
name=ETTm2
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model $model --data ETTm2 --features M --seq_len 96 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id ETTm2 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 5.1 electricity
name=ECL
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model $model --data custom --features M --seq_len 336 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --K 3 --learning_rate 3e-4 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id ECL --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log


# 6.1 exchange
name=Exchange
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model $model --data custom --features M --seq_len 336 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --K 0 --learning_rate 1e-4 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id Exchange --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 7.1 traffic
name=Traffic
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model $model --data custom --features M --seq_len 336 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 8.1 weather
name=Weather
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/weather/ --data_path weather.csv --model $model --data custom --features M --seq_len 336 --pred_len $pred_len --e_layers 2 --d_layers 2 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --K 3 --learning_rate 3e-4 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id weather --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log
done


# 9.1 illness
for pred_len in 24 36 48 60
do
# illness
name=Illness
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model $model --data custom --features M --seq_len 336 --pred_len 60 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 --gpu $gpu_num \
    --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --run_train --run_test  > $cur_path'/'train_and_test_loss.log
done
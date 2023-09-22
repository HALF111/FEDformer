# Crossformer
gpu_num=0

dir_name=all_result
model=Crossformer

# 1.ETTh1
name=ETTh1

# 1.1 ETTh1 + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 1.2 ETTh1 + 192
seq_len=720; pred_len=192
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 1.3 ETTh1 + 336
seq_len=720; pred_len=336
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 1.4 ETTh1 + 720
seq_len=720; pred_len=720
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 2.ETTh2
name=ETTh2

# 2.1 ETTh2 + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model Crossformer --data ETTh2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 2.2 ETTh2 + 192
seq_len=720; pred_len=192
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model Crossformer --data ETTh2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 2.3 ETTh2 + 336
seq_len=720; pred_len=336
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model Crossformer --data ETTh2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 2.4 ETTh2 + 720
seq_len=720; pred_len=720
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model Crossformer --data ETTh2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 3.ETTm1
name=ETTm1

# 3.1 ETTm1 + 96
seq_len=672; pred_len=96
seg_len=12; lr=1e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model Crossformer --data ETTm1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 3.2 ETTm1 + 192
seq_len=672; pred_len=192
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model Crossformer --data ETTm1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 3.3 ETTm1 + 336
seq_len=672; pred_len=336
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model Crossformer --data ETTm1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 3.4 ETTm1 + 720
seq_len=672; pred_len=720
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model Crossformer --data ETTm1 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm1 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 4.ETTm2
name=ETTm2

# 4.1 ETTm2 + 96
seq_len=672; pred_len=96
seg_len=12; lr=1e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model Crossformer --data ETTm2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 4.2 ETTm2 + 192
seq_len=672; pred_len=192
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model Crossformer --data ETTm2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 4.3 ETTm2 + 336
seq_len=672; pred_len=336
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model Crossformer --data ETTm2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 4.4 ETTm2 + 720
seq_len=672; pred_len=720
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --model Crossformer --data ETTm2 --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm2 --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 5.ECL
name=ECL

# 5.1 ECL + 96
seq_len=336; pred_len=96
seg_len=12; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id ECL --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 5.2 ECL + 192
seq_len=720; pred_len=192
seg_len=24; lr=5e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id ECL --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 5.3 ECL + 336
seq_len=720; pred_len=336
seg_len=24; lr=5e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id ECL --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 5.4 ECL + 720
seq_len=720; pred_len=720
seg_len=24; lr=5e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id ECL --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 6.Traffic
name=Traffic

# 6.1 Traffic + 96
seq_len=336; pred_len=96
seg_len=12; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id traffic --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 6.2 Traffic + 192
seq_len=720; pred_len=192
seg_len=24; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id traffic --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 6.3 Traffic + 336
seq_len=720; pred_len=336
seg_len=24; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id traffic --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 6.4 Traffic + 720
seq_len=720; pred_len=720
seg_len=24; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id traffic --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 7.Exchange
name=Exchange

# 7.1 Exchange + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id Exchange --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 7.2 Exchange + 192
seq_len=720; pred_len=192
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id Exchange --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 7.3 Exchange + 336
seq_len=720; pred_len=336
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id Exchange --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 7.4 Exchange + 720
seq_len=720; pred_len=720
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id Exchange --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 8.Weather
name=weather

# 8.1 weather + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/weather/ --data_path weather.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id weather --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 8.2 weather + 192
seq_len=336; pred_len=192
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/weather/ --data_path weather.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id weather --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 8.3 weather + 336
seq_len=336; pred_len=336
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/weather/ --data_path weather.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id weather --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 8.4 weather + 720
seq_len=720; pred_len=720
seg_len=24; lr=1e-5
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/weather/ --data_path weather.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id weather --dropout 0.2 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log



# 9.Illness
name=Illness

# 9.1 Illness + 24
seq_len=48; pred_len=24
seg_len=6; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 9.2 Illness + 36
seq_len=48; pred_len=36
seg_len=6; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 9.3 Illness + 48
seq_len=60; pred_len=48
seg_len=6; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

# 9.4 Illness + 60
seq_len=60; pred_len=60
seg_len=6; lr=5e-4
cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
if [ ! -d $cur_path ]; then
    mkdir $cur_path
fi
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len $seq_len --pred_len $pred_len --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate $lr --itr 1 --gpu $gpu_num \
    --is_training 1 --seg_len $seg_len --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --test_train_num 10 --run_train --run_test  > $cur_path'/'train_and_test_loss.log

gpu_num=0

dir_name=all_result

# ablation study on designed index

# 1. traffic

# 1.1 FEDformer
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2000 --draw_adapt_figure
original: mse:0.5744055509567261, mae:0.35589635372161865
# 1. normal
lr*2000,ttn200,select10: mse:0.5126578211784363, mae:0.34407877922058105


# 1.2 Autoformer
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2000 --draw_adapt_figure
original: mse:0.620464563369751, mae:0.3909095227718353
# 1. normal
mse:0.5646195411682129, mae:0.37694627046585083


# 1.3 Informer
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2000 --draw_adapt_figure
original: mse:0.731, mae:0.406
# 1. normal
mse:0.628, mae:0.393


# 1.4 ETSformer
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 200 --draw_adapt_figure
original: mse:0.599, mae:0.386
# 1. normal
mse:0.487, mae:0.354


# 2. illness

# 2.1 FEDformer
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model FEDformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --draw_adapt_figure
original: mse:3.2410900592803955, mae:1.2523982524871826
# 1. normal
lr*50,ttn200,select3: mse:3.0759356021881104, mae:1.2209198474884033
lr*50,ttn200,select10: mse:3.0844264030456543, mae:1.2246235609054565


# 2.2 Autoformer
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Autoformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --draw_adapt_figure
original: mse:3.313739538192749, mae:1.2446492910385132
# 1. normal
lr*50,ttn200,select3, in-fact: mse:3.125, mae:1.213


# 2.3 Informer
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Informer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --draw_adapt_figure
original: mse:5.105955123901367, mae:1.5340543985366821
# 1. normal
lr*50,ttn200,select3: mse:2.874431848526001, mae:1.1496471166610718


# 2.4 ETSformer
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model ETSformer --data custom --features M --seq_len 60 --pred_len 24 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 0.5 --draw_adapt_figure
original: mse:2.396606206893921, mae:0.9926647543907166
# 1. normal
lr*50,ttn200,select3: mse:2.3527326583862305, mae:0.9860926270484924


# 2.5 Crossformer
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len 48 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --draw_adapt_figure
original: mse:3.328763008117676, mae:1.2749556303024292
# 1. normal
lr*50,ttn200,select3: mse:2.3527326583862305, mae:0.9860926270484924




# 3. Electricity
# 1. FEDformer
python -u run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 500 --draw_adapt_figure
original: mse:0.188, mae:0.304
# 1. normal
mse:0.172, mae:0.284

# 2. Autoformer
python -u run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 1000 --draw_adapt_figure
original: mse:0.207, mse:0.324
# 1. normal
mse:0.189, mae:0.304

# 3. Informer
python -u run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 500 --draw_adapt_figure
original: mse:0.321, mae:0.407
# 1. normal
mse:0.245, mae:0.355

# 4. ETSformer
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --K 3 --learning_rate 3e-4 --itr 1 --gpu 0 --d_model 512 --is_training 1 --task_id ECL --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 1000 --draw_adapt_figure
original: mse:0.187, mae:0.304
# 1. normal
mse:0.171, mae:0.285

# 5. Crossformer
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model Crossformer --data custom --features M --seq_len 336 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --learning_rate 5e-4 --itr 1 --gpu 0 --is_training 1 --seg_len 12 --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id ECL --dropout 0.2 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 20 --draw_adapt_figure
original: mse:0.184, mae:0.297
# 1. normal
mse:0.183, mae:0.295


# 4. ETTh1
# 4.1 Informer
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model Informer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 50 --gpu 0 --draw_adapt_figure
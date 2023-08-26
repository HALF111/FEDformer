# Crossformer

# 1.1 ETTh1 + pred_len = 48 & 168
python main_crossformer.py --data ETTh1 --in_len 168 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1
python main_crossformer.py --data ETTh1  --in_len 720 --out_len 168 --seg_len 24 --learning_rate 1e-5 --itr 1 

# 1.2 ETTh1 + pred_len = 96
python main_crossformer.py --data ETTh1 --in_len 336 --out_len 96 --seg_len 12 --learning_rate 3e-5 --itr 1
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 3e-5 --itr 1 --gpu 1 \
    --is_training 1 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --seg_len 12 --test_train_num 10
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 3e-5 --itr 1 --gpu 1 \
    --is_training 1 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --seg_len 12 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 6.1 Traffic + 96
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model Crossformer --data custom --features M --seq_len 336 --pred_len 96 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --learning_rate 5e-4 --itr 1 --gpu 1 \
    --is_training 1 --seg_len 12 --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id traffic --dropout 0.2 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 10
# Autoformer
# PS：Autoformer好像只能在cuda:0上训练，在cuda:1上训练会出问题

# 1.1 ETTh1 & pred_len=96
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model Autoformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --test_train_num 10
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model Autoformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle

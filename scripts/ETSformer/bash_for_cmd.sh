# ETSformer

# 1.1 ETTh1 & pred_len = 96
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model ETSformer --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 --gpu 1 \
    --d_model 512 --is_training 1 \
    --task_id ETTh1 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid'  --test_train_num 10
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model ETSformer --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 --gpu 1 \
    --d_model 512 --is_training 1 \
    --task_id ETTh1 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid'  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5  --adapt_cycle

# 5.1 Exchange + pred_len = 96
# --model_id Exchange 
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --K 0 --learning_rate 3e-5 --itr 1 --gpu 1 \
    --d_model 512 --is_training 1 \
    --task_id Exchange --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid'  --run_train --run_test
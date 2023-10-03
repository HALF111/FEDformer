
dir=param_sensitivity
# # 1. FEDformer
# # PART I:
# selected_data_num=3
# for test_train_num in 50 100 200 300 500
# do
# python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model FEDformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_FED_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART II:
# test_train_num=200
# for selected_data_num in 1 2 5 10
# do
# python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model FEDformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_FED_ttn$test_train_num'_'select$selected_data_num.txt
# done


# 2. Autoformer
# PART I:
selected_data_num=3
for test_train_num in 50 100 200 300 500
do
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Autoformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_Auto_ttn$test_train_num'_'select$selected_data_num.txt
done
# PART II:
test_train_num=200
for selected_data_num in 1 2 5 10
do
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Autoformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_Auto_ttn$test_train_num'_'select$selected_data_num.txt
done

# 3. Informer
# PART I:
selected_data_num=3
for test_train_num in 50 100 200 300 500
do
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Informer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 100 > $dir/ILI_In_ttn$test_train_num'_'select$selected_data_num.txt
done
# PART II:
test_train_num=200
for selected_data_num in 1 2 5 10
do
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Informer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_In_ttn$test_train_num'_'select$selected_data_num.txt
done


# # 4. ETSformer
# # PART I:
# selected_data_num=3
# for test_train_num in 50 100 200 300 500
# do
# python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model ETSformer --data custom --features M --seq_len 60 --pred_len 24 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 0.5 > $dir/ILI_ETS_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART II:
# test_train_num=200
# for selected_data_num in 1 2 5 10
# do
# python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model ETSformer --data custom --features M --seq_len 60 --pred_len 24 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 0.5 > $dir/ILI_ETS_ttn$test_train_num'_'select$selected_data_num.txt
# done


# 5. Crossformer
# PART I:
selected_data_num=3
for test_train_num in 50 100 200 300 500
do
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len 48 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_Cross_ttn$test_train_num'_'select$selected_data_num.txt
done
# PART II:
test_train_num=200
for selected_data_num in 1 2 5 10
do
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len 48 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 50 > $dir/ILI_Cross_ttn$test_train_num'_'select$selected_data_num.txt
done
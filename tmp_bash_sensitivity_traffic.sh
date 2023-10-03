
dir=param_sensitivity
# 1. FEDformer
# # PART I:
# selected_data_num=10
# for test_train_num in 500 1000 2000 5000 10000
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_FED_ttn$test_train_num'_'select$selected_data_num.txt
# done

# # PART II:
# test_train_num=1000
# for selected_data_num in 2 5 20 30
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_FED_ttn$test_train_num'_'select$selected_data_num.txt
# done

# PART III:
test_train_num=1000; selected_data_num=10
for lambda_period in 0.05 0.1 0.2 0.5
do
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 --lambda_period $lambda_period > $dir/Traffic_FED_ttn$test_train_num'_'select$selected_data_num'_'lambda$lambda_period.txt
done


# 2. Autoformer
# # PART I:
# selected_data_num=10
# for test_train_num in 500 1000 2000 5000 10000
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_Auto_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART II:
# test_train_num=1000
# for selected_data_num in 2 5 20 30
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_Auto_ttn$test_train_num'_'select$selected_data_num.txt
# done
# PART III:
test_train_num=1000; selected_data_num=10
for lambda_period in 0.05 0.1 0.2 0.5
do
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 --lambda_period $lambda_period > $dir/Traffic_Auto_ttn$test_train_num'_'select$selected_data_num'_'lambda$lambda_period.txt
done



# # 3. Informer
# # PART I:
# selected_data_num=10
# for test_train_num in 500 1000 2000 5000 10000
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_In_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART II:
# test_train_num=1000
# for selected_data_num in 2 5 20 30
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_In_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART III:
# test_train_num=1000;
# for selected_data_num in 2 5 20 30
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_In_ttn$test_train_num'_'select$selected_data_num.txt
# done


# # 4. ETSformer
# # PART I:
# selected_data_num=10
# for test_train_num in 500 1000 2000 5000 10000
# do
# python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 200 > $dir/traffic_ETS_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART II:
# test_train_num=1000
# for selected_data_num in 2 5 20 30
# do
# python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 200 > $dir/traffic_ETS_ttn$test_train_num'_'select$selected_data_num.txt
# done




# # 5. Crossformer
# # PART I:
# selected_data_num=10
# for test_train_num in 500 1000 2000 5000 10000
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_Auto_ttn$test_train_num'_'select$selected_data_num.txt
# done
# # PART II:
# test_train_num=1000
# for selected_data_num in 2 5 20 30
# do
# python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num $test_train_num --run_select_with_distance --selected_data_num $selected_data_num --adapted_lr_times 2000 > $dir/traffic_Auto_ttn$test_train_num'_'select$selected_data_num.txt
# done

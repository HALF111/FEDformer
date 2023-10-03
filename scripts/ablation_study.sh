
gpu_num=0

dir_name=all_result

# ablation study on designed index

# 1. traffic

# 1.1 FEDformer
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2000 --remove_distance --remove_cycle --remove_nearest
original: mse:0.5744055509567261, mae:0.35589635372161865
# 1. normal
lr*2000,ttn200,select10: mse:0.5126578211784363, mae:0.34407877922058105
# 2.remove_distance
lr*2000,ttn200,select10: mse:0.5201103091239929, mae:0.3430194556713104
# 3.remove_distance + remove_cycle
lr*2000,ttn200,select10: mse:0.5316569209098816, mae:0.34730303287506104
# 4.remove_distance + remove_cycle + remove_nearest
lr*2000,ttn200,select10: mse:0.5203375220298767, mae:0.34564638137817383


# 1.2 Autoformer
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2000 --remove_distance --remove_cycle --remove_nearest
original: mse:0.620464563369751, mae:0.3909095227718353
# 1. normal
mse:0.5646195411682129, mae:0.37694627046585083
# 2.remove_distance
mse:0.5767139792442322, mae:0.3795345723628998
# 3.remove_distance + remove_cycle
mse:0.5898356437683105, mae:0.3843180537223816
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.5747594237327576, mae:0.37943434715270996


# 1.3 Informer
python -u run.py --is_training 1 --root_path ./dataset/traffic/ --data_path traffic.csv --task_id traffic --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --itr 1 --train_epochs 10 --gpu 0  --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 2000 --remove_distance --remove_cycle --remove_nearest
original: mse:0.731, mae:0.406
# 1. normal
mse:0.628, mae:0.393
# 2.remove_distance
mse:0.6273484826087952, mae:0.3927730917930603
# 3.remove_distance + remove_cycle
mse:0.636137843132019, mae:0.3963523209095001
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.639647364616394, mae:0.3975795805454254


# 1.4 ETSformer
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 200 --remove_distance --remove_cycle --remove_nearest
original: mse:0.599, mae:0.386
# 1. normal
mse:0.487, mae:0.354
# 2.remove_distance
mse:0.4887968897819519, mae:0.35630592703819275
# 3.remove_distance + remove_cycle
mse:0.4994681477546692, mae:0.3610208034515381
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.49303141236305237, mae:0.35640862584114075


# 2. illness

# 2.1 FEDformer
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model FEDformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --remove_distance --remove_cycle --remove_nearest
original: mse:3.2410900592803955, mae:1.2523982524871826
# 1. normal
lr*50,ttn200,select3: mse:3.0759356021881104, mae:1.2209198474884033
lr*50,ttn200,select10: mse:3.0844264030456543, mae:1.2246235609054565

# 2.remove_distance
lr*50,ttn200,select3: mse:3.0891544818878174, mae:1.2231615781784058
lr*50,ttn200,select10: mse:3.144599199295044, mae:1.233025312423706

# 3.remove_distance + remove_cycle
lr*50,ttn200,select10: mse:6.023703575134277, mae:1.6392604112625122
lr*10,ttn200,select10: mse:3.4409191608428955, mae:1.2816754579544067
lr*2,ttn200,select10: mse:3.2805557250976562, mae:1.2613320350646973

lr*50,ttn200,select3: mse:3.819453239440918, mae:1.342251181602478
lr*10,ttn200,select3: mse:3.3155012130737305, mae:1.2662614583969116

# 4.remove_distance + remove_cycle + remove_nearest
lr*50,ttn200,select10: mse:3.3389692306518555, mae:1.2662568092346191
lr*10,ttn200,select10: mse:3.2569525241851807, mae:1.2567212581634521
lr*2,ttn200,select10: mse:3.2578811645507812, mae:1.2575926780700684

lr*50,ttn200,select3: mse:3.2343649864196777, mae:1.252769112586975
lr*10,ttn200,select3: mse:3.245779514312744, mae:1.2557610273361206


# 2.2 Autoformer
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Autoformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --remove_distance --remove_cycle --remove_nearest
original: mse:3.313739538192749, mae:1.2446492910385132
# 1. normal
lr*50,ttn200,select3: mse:3.2633934020996094, mae:1.2333115339279175
lr*50,ttn200,select3, in-fact: mse:3.125, mae:1.213

# 2.remove_distance
lr*50,ttn200,select3: mse:3.259821653366089, mae:1.2354258298873901

# 3.remove_distance + remove_cycle
lr*50,ttn200,select3: mse:3.682704448699951, mae:1.3244783878326416

# 4.remove_distance + remove_cycle + remove_nearest
lr*50,ttn200,select3: mse:3.3331308364868164, mae:1.2506484985351562


# 2.3 Informer
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model Informer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --remove_distance --remove_cycle --remove_nearest
original: mse:5.105955123901367, mae:1.5340543985366821
# 1. normal
lr*50,ttn200,select3: mse:2.874431848526001, mae:1.1496471166610718

# 2.remove_distance
lr*50,ttn200,select3: mse:2.608675241470337, mae:1.0861648321151733

# 3.remove_distance + remove_cycle
lr*50,ttn200,select3: mse:4.765204906463623, mae:1.4892686605453491

# 4.remove_distance + remove_cycle + remove_nearest
lr*50,ttn200,select3: mse:4.442682266235352, mae:1.4269663095474243


# 2.4 ETSformer
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model ETSformer --data custom --features M --seq_len 60 --pred_len 24 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 0.5 --remove_distance --remove_cycle --remove_nearest
original: mse:2.396606206893921, mae:0.9926647543907166
# 1. normal
lr*50,ttn200,select3: mse:2.3527326583862305, mae:0.9860926270484924

# 2.remove_distance
lr*50,ttn200,select3: mse:2.334181547164917, mae:0.9867528080940247

# 3.remove_distance + remove_cycle
lr*50,ttn200,select3: mse:2.4969911575317383, mae:1.0290600061416626

# 4.remove_distance + remove_cycle + remove_nearest
lr*50,ttn200,select3: mse:2.37680721282959, mae:0.9892348051071167


# 2.5 Crossformer
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len 48 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --gpu 0 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 50 --remove_distance --remove_cycle --remove_nearest
original: mse:3.328763008117676, mae:1.2749556303024292
# 1. normal
lr*50,ttn200,select3: mse:2.3527326583862305, mae:0.9860926270484924

# 2.remove_distance
lr*50,ttn200,select3: mse:2.0698330402374268, mae:0.9631286859512329

# 3.remove_distance + remove_cycle
lr*50,ttn200,select3: mse:2.7046263217926025, mae:1.0982064008712769

# 4.remove_distance + remove_cycle + remove_nearest
lr*50,ttn200,select3: mse:2.821445941925049, mae:1.1353179216384888




# 3. Electricity
# 1. FEDformer
python -u run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 500 --remove_distance --remove_cycle --remove_nearest
original: mse:0.188, mae:0.304
# 1. normal
mse:0.172, mae:0.284
# 2.remove_distance
mse:0.17177370190620422, mae:0.2827122211456299
# 3.remove_distance + remove_cycle
mse:0.17308975756168365, mae:0.28376251459121704
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.1745980679988861, mae:0.2871730327606201

# 2. Autoformer
python -u run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL --model Autoformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 1000 --remove_distance --remove_cycle --remove_nearest
original: mse:0.207, mse:0.324
# 1. normal
mse:0.189, mae:0.304
# 2.remove_distance
mse:0.1956806629896164, mae:0.3094704747200012
# 3.remove_distance + remove_cycle
mse:0.19936515390872955, mae:0.3125047981739044
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.19282908737659454, mae:0.3091682195663452

# 3. Informer
python -u run.py --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --task_id ECL --model Informer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 1 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 500 --remove_distance --remove_cycle --remove_nearest
original: mse:0.321, mae:0.407
# 1. normal
mse:0.245, mae:0.355
# 2.remove_distance
mse:0.24231937527656555, mae:0.35214951634407043
# 3.remove_distance + remove_cycle
mse:0.2421179860830307, mae:0.3518781363964081
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.2421179860830307, mae:0.3518781363964081

# 4. ETSformer
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --K 3 --learning_rate 3e-4 --itr 1 --gpu 0 --d_model 512 --is_training 1 --task_id ECL --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 1000 --remove_distance --remove_cycle --remove_nearest
original: mse:0.187, mae:0.304
# 1. normal
mse:0.171, mae:0.285
# 2.remove_distance
mse:0.17370165884494781, mae:0.287659227848053
# 3.remove_distance + remove_cycle
mse:0.17433278262615204, mae:0.288535475730896
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.17431332170963287, mae:0.2897290885448456

# 5. Crossformer
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model Crossformer --data custom --features M --seq_len 336 --pred_len 96 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --learning_rate 5e-4 --itr 1 --gpu 0 --is_training 1 --seg_len 12 --d_model 64 --d_ff 128 --e_layers 3 --n_heads 2 --task_id ECL --dropout 0.2 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 20 --remove_distance --remove_cycle --remove_nearest
original: mse:0.184, mae:0.297
# 1. normal
mse:0.183, mae:0.295
# 2.remove_distance
mse:0.18820136785507202, mae:0.30169248580932617
# 3.remove_distance + remove_cycle
mse:0.1884719729423523, mae:0.3022782504558563
# 4.remove_distance + remove_cycle + remove_nearest
mse:0.18406078219413757, mae:0.29652708768844604




# # traffic
# name=Traffic
# cur_path=./$dir_name/$model'_'$name'_'pl$pred_len
# if [ ! -d $cur_path ]; then
#     mkdir $cur_path
# fi
# python -u run.py --is_training 1 \
#  --root_path ./dataset/traffic/ \
#  --data_path traffic.csv \
#  --task_id traffic \
#  --model $model \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 862 \
#  --dec_in 862 \
#  --c_out 862 \
#  --des 'Exp' \
#  --itr 1 \
#  --train_epochs 10 \
#   --gpu $gpu_num \
#   --run_train --run_test \
#   > $cur_path'/'train_and_test_loss.log
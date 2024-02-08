# 1.1 ETTh1 & pred_len=96
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model FEDformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --test_train_num 10
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model FEDformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# add_revin
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model FEDformer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --add_revin --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# Linear
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --task_id ETTh1 --model Linear --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.005 --gpu 0 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# 2.1 ETTh1
# add_revin
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --task_id ETTh2 --model FEDformer --data ETTh2 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --add_revin --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# 3.1 ETTm1 & pred_len=96
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --task_id ETTm1 --model FEDformer --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --add_revin --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# 4.1 ETTm2 & pred_len=96
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --task_id ETTm2 --model FEDformer --data ETTm2 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --add_revin --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# 5.1 Exchange & pred_len=96
python -u run.py --is_training 1 --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --task_id Exchange --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --itr 1  --gpu 0 --test_train_num 10


# 6.1 Traffic & pred_len=96


# 8.1 Weather
python -u run.py --is_training 1 --root_path ./dataset/weather/ --data_path weather.csv --task_id weather --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --add_revin --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5

# 9.1 Illness & pred_len=24
python -u run.py --is_training 1 --root_path ./dataset/illness/ --data_path national_illness.csv --task_id ili --model FEDformer --data custom --features M --seq_len 36 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --add_revin --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 3


# !Crossformer
# 1. ETTh1 + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
original: mse:0.410861074924469, mae:0.4315704107284546
lr*5,ttn1000,select10: mse:0.41027018427848816, mae:0.43151891231536865
lr*5,ttn1000,select5: mse:0.41048166155815125, mae:0.4316507577896118
lr*20,ttn1000,select10: mse:0.40912559628486633, mae:0.43040332198143005
lr*100,ttn1000,select10: mse:0.4038582146167755, mae:0.4273868203163147
lr*200,ttn1000,select10: mse:0.4011910557746887, mae:0.4273570775985718
lr*300,ttn1000,select10: mse:0.4028758704662323, mae:0.43138808012008667

python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model Crossformer --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 3e-5 --itr 1 \
    --gpu 1 --is_training 1 --seg_len 12 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh1 --dropout 0.2 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 2.1 ETTh2 + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
original: mse:0.6409829258918762, mae:0.5548349618911743 ???
lr*5,ttn1000,select10: mse:0.6420011520385742, mae:0.5561877489089966
lr*5,ttn1000,select5: mse:0.644170343875885, mae:0.5569418668746948
lr*20,ttn1000,select10: mse:0.6243089437484741, mae:0.5491762161254883
lr*100,ttn1000,select10: mse:0.5710586905479431, mae:0.5351401567459106
lr*200,ttn1000,select10: mse:0.5331114530563354, mae:0.5329473614692688
lr*300,ttn1000,select10: mse:0.5269593596458435, mae:0.5442441701889038
lr*500,ttn1000,select10: mse:0.6100402474403381, mae:0.6073511838912964
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model Crossformer --data ETTh2 --features M --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 3e-5 --itr 1 \
    --gpu 1 --is_training 1 --seg_len 12 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTh2 --dropout 0.2 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 3.1 ETTm1 + 96
seq_len=672; pred_len=96
seg_len=12; lr=1e-4
original: mse:0.31194964051246643, mae:0.36705461144447327
lr*2,ttn1000,select10: mse:0.3116062879562378, mae:0.3669998049736023
lr*5,ttn1000,select10: mse:0.3113485276699066, mae:0.3671877086162567
lr*10,ttn1000,select10: mse:0.311583936214447, mae:0.36819639801979065
lr*20,ttn1000,select10: mse:0.3145460784435272, mae:0.3726714551448822
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --model Crossformer --data ETTm1 --features M --seq_len 672 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 1e-4 --itr 1 \
    --gpu 1 --is_training 1 --seg_len 12 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ETTm1 --dropout 0.2 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle




# 7.1 Exchange + 96
seq_len=336; pred_len=96
seg_len=12; lr=3e-5
original: mse:0.24559393525123596, mae:0.3742602467536926
lr*5,ttn1000,select10: mse:0.24128007888793945, mae:0.3721718490123749
lr*20,ttn1000,select10: mse:0.22941231727600098, mae:0.3665124177932739
lr*50,ttn1000,select10: mse:0.21024805307388306, mae:0.35746464133262634
lr*200,ttn1000,select10: mse:0.20585429668426514, mae:0.3585275113582611
lr*300,ttn1000,select10: mse:0.28757980465888977, mae:0.4143621325492859
lr*500,ttn1000,select10: mse:0.6542031764984131, mae:0.6345582008361816
python -u run.py --root_path ./dataset/exchange_rate/ --data_path exchange_rate.csv --model Crossformer --data custom --features M --seq_len 336 --pred_len 96 --enc_in 8 --dec_in 8 --c_out 8 --des 'Exp' --learning_rate 3e-5 --itr 1 \
    --gpu 1 --is_training 1 --seg_len 12 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id Exchange --dropout 0.2 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 9.1 Illness + 24
seq_len=48; pred_len=24
seg_len=6; lr=5e-4
original: mse:3.328763008117676, mae:1.2749556303024292
lr*5,ttn200,select3: mse:3.12184476852417, mae:1.2312310934066772
lr*20,ttn200,select3: mse:2.663424253463745, mae:1.1207226514816284
lr*50,ttn200,select3: mse:2.3968188762664795, mae:1.0521787405014038
lr*70,ttn200,select3: mse:2.700737953186035, mae:1.1233477592468262
lr*100,ttn200,select3: mse:3.879101514816284, mae:1.3495787382125854
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model Crossformer --data custom --features M --seq_len 48 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --learning_rate 5e-4 --itr 1 \
    --gpu 1 --is_training 1 --seg_len 6 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id ili --dropout 0.6 --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 5 --adapt_cycle


# !ETSformer
# 1. ETTh1 + 96
lr=1e-5
original: mse:0.4950839877128601, mae:0.4803565442562103
lr*5,ttn1000,select10: mse:0.49577897787094116, mae:0.4805293083190918
lr*20,ttn1000,select10: mse:0.49291524291038513, mae:0.4792264401912689
lr*50,ttn1000,select10: mse:0.4924919009208679, mae:0.47900334000587463
lr*100,ttn1000,select10: mse:0.49916866421699524, mae:0.48234590888023376
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model ETSformer --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 \
    --gpu 1 --d_model 512 --is_training 1 --task_id ETTh1 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle

# 2.ETTh2 + 96
lr=1e-5
original: mse:0.34645792841911316, mae:0.4012892246246338
lr*5,ttn1000,select10: mse:0.34893798828125, mae:0.4028205871582031
lr*5,ttn1000,select5: mse:0.3462805151939392, mae:0.4011647403240204
lr*20,ttn1000,select10: mse:0.3460102379322052, mae:0.4009450674057007
lr*50,ttn1000,select10: mse:0.345519483089447, mae:0.4005391299724579
lr*100,ttn1000,select10: mse:0.344849169254303, mae:0.39996275305747986
lr*200,ttn1000,select10: mse:0.34406331181526184, mae:0.39921897649765015
lr*300,ttn1000,select10: mse:0.3440164029598236, mae:0.39904001355171204
lr*500,ttn1000,select10: mse:0.3461402952671051, mae:0.4002092182636261
python -u run.py --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model ETSformer --data ETTh2 --features M --seq_len 96 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 3 --learning_rate 1e-5 --itr 1 \
    --gpu 1 --d_model 512 --is_training 1 --task_id ETTh2 --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 5.ECL + 96
lr=3e-4
original: mse:0.1871151328086853, mae:0.3040798306465149
lr*10,ttn1000,select10: mse:0.18677830696105957, mae:0.3037184476852417
lr*50,ttn1000,select10: mse:0.18543478846549988, mae:0.3022591769695282
lr*200,ttn1000,select10: mse:0.18092131614685059, mae:0.297238290309906
lr*500,ttn1000,select10: mse:0.17438288033008575, mae:0.2896188497543335
lr*1000,ttn1000,select10: mse:0.17085830867290497, mae:0.2854418456554413
lr*1200,ttn1000,select10: mse:0.17655029892921448, mae:0.292043536901474
python -u run.py --root_path ./dataset/electricity/ --data_path electricity.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --K 3 --learning_rate 3e-4 --itr 1 \
    --gpu 1 --d_model 512 --is_training 1 --task_id ECL --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 6.Traffic + 96
lr=1e-3
original: mse:0.5993558764457703, mae:0.38553813099861145
lr*5,ttn1000,select10: mse:0.5980913043022156, mae:0.3851860463619232
lr*100,ttn1000,select10: mse:0.5753857493400574, mae:0.3784221112728119
lr*200,ttn1000,select10: mse:0.5546223521232605, mae:0.3723624646663666
lr*500,ttn1000,select10: mse:0.5116651654243469, mae:0.36181291937828064
lr*1000,ttn1000,select10: mse:0.5044950246810913, mae:0.37126803398132324
lr*1500,ttn1000,select10: mse:0.5778584480285645, mae:0.40615594387054443
python -u run.py --root_path ./dataset/traffic/ --data_path traffic.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 862 --dec_in 862 --c_out 862 --des 'Exp' --K 3 --learning_rate 1e-3 --itr 1 \
    --gpu 1 --d_model 512 --is_training 1 --task_id traffic --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle


# 8.weather + 96
python -u run.py --root_path ./dataset/weather/ --data_path weather.csv --model ETSformer --data custom --features M --seq_len 336 --pred_len 96 --e_layers 2 --d_layers 2 --enc_in 21 --dec_in 21 --c_out 21 --des 'Exp' --K 3 --learning_rate 3e-4 --itr 1 \
    --gpu 1 --d_model 512 --is_training 1 --task_id weather --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid'

# 9.Illness + 24
lr=1e-3
original: mse:2.396606206893921, mae:0.9926647543907166
lr*0.2,ttn200,select3: mse:2.381488561630249, mae:0.9907649159431458
lr*0.5,ttn200,select3: mse:2.3527326583862305, mae:0.9860926270484924
lr*1,ttn200,select3: mse:2.363419532775879, mae:0.9928531050682068
lr*2,ttn200,select3: mse:2.6045937538146973, mae:1.0513383150100708
lr*5,ttn200,select3: mse:5.086517333984375, mae:1.4711779356002808
python -u run.py --root_path ./dataset/illness/ --data_path national_illness.csv --model ETSformer --data custom --features M --seq_len 60 --pred_len 24 --e_layers 2 --d_layers 2 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --K 1 --learning_rate 1e-3 --itr 1 \
    --gpu 1 --d_model 512 --is_training 1 --task_id ili --lradj exponential_with_warmup --dropout 0.2 --label_len 0 --activation 'sigmoid' --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 5 --adapt_cycle



# WTH_informer
# 1.Autoformer

# 2.FEDformer
original: mse:0.48461171984672546, mae:0.49114471673965454
lr*20,ttn10000,select10: mse:0.48604705929756165, mae:0.49172478914260864
lr*5,ttn10000,select10: mse:0.48433244228363037, mae:0.49088990688323975
lr*5,ttn20000,select5: mse:0.48446616530418396, mae:0.49101945757865906
lr*10,ttn25000,select5: mse:0.48460277915000916, mae:0.49108758568763733
python -u run.py --is_training 1 --root_path ./dataset/WTH_informer/ --data_path WTH_informer.csv --task_id WTH_informer --model FEDformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 12 --dec_in 12 --c_out 12 --des 'Exp' --d_model 512 --itr 1  --gpu 0 --test_train_num 10

# 3.Crossformer
original: mse:0.4319957196712494, mae:0.4606797397136688
lr*5,ttn1000,select10: mse:0.4318191707134247, mae:0.460480272769928
lr*20,ttn1000,select10: mse:0.43188780546188354, mae:0.46041521430015564
lr*10,ttn10000,select10: mse:0.4317159056663513, mae:0.46010661125183105
lr*20,ttn10000,select10: mse:0.4316633939743042, mae:0.45971766114234924
lr*50,ttn10000,select10: mse:0.4328922629356384, mae:0.45968472957611084
python -u run.py --root_path ./dataset/WTH_informer/ --data_path WTH_informer.csv --model Crossformer --data custom --features M --seq_len 336 --pred_len 96 --enc_in 12 --dec_in 12 --c_out 12 --des 'Exp' --learning_rate 3e-5 --itr 1 \
    --gpu 1 --is_training 1 --seg_len 12 --d_model 256 --d_ff 512 --e_layers 3 --n_heads 4 --task_id WTH_informer --dropout 0.2 --test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 5 --adapt_cycle
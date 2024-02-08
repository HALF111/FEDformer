# train.py
from tunetools import decorator, Parameter

# This is the main function, which will be executed during training. 
# Tunetools will recognize the parameters automatically, and construct 
# grid search with the given domains. 
# "num_sample=2" means each experiment will run for 2 times.
# @decorator.main(num_sample=2)
@decorator.main(num_sample=1)
def main(
        # Register the hyper-parameters manifest here. 
        # alpha: Parameter(default=0, domain=[0, 1, 2]),
        # lambda_reg: Parameter(default=10000, domain=[1000, 10000, 100000, 1000000]),

        # test_train_num: Parameter(default=1000, domain=[10, 20, 30]),
        adapt_lr_level: Parameter(default=0, domain=[0, 1, 2, 3, 4, 5, 6]),
        
        # model: Parameter(default="Autoformer", domain=["Autoformer", "FEDformer"]),
        # model: Parameter(default="Autoformer", domain=["Autoformer", "Informer", "FEDformer", "ETSformer", "Crossformer"]),
        model: Parameter(default="Autoformer", domain=["Autoformer", "Informer", "FEDformer"]),

        # dataset: Parameter(default="ETTh1", domain=["ETTh1", "ECL", "Traffic", "Exchange", "ETTh2", "ETTm1", "ETTm2", "Weather", "Illness"]),
        # dataset: Parameter(default="ETTh1", domain=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "Traffic", "Exchange", "Weather", "Illness"]),
        dataset: Parameter(default="ETTh1", domain=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "Traffic", "Weather", "Illness"]),
        # dataset: Parameter(default="ETTh1", domain=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Illness"]),

        # pred_len: Parameter(default=96, domain=[96, 192, 336, 720]),
        pred_len: Parameter(default=96, domain=[96, 192]),
        # pred_len: Parameter(default=96, domain=[96]),
        gpu: Parameter(default=0, domain=[0]),

):
    # Do training here, use all the parameters...
    import os
    import random

    # mapping = {
    #     "ETTh1": {"root_path": "./dataset/ETT-small", "data_path": "ETTh1", "data": "ETTh1", "task_id": "ETTh1", "variant_num": 7, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [100, 200, 300], "K": 3, "lr_ets": 1e-5, "lr_cross": [3e-5, 1e-5, 1e-5, 1e-5], "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    #     "ETTh2": {"root_path": "./dataset/ETT-small", "data_path": "ETTh2", "data": "ETTh2", "task_id": "ETTh2", "variant_num": 7, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [100, 200, 300], "K": 3, "lr_ets": 1e-5, "lr_cross": [3e-5, 1e-5, 1e-5, 1e-5], "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    #     "ETTm1": {"root_path": "./dataset/ETT-small", "data_path": "ETTm1", "data": "ETTm1", "task_id": "ETTm1", "variant_num": 7, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 1e-5, "lr_cross": [1e-4, 1e-5, 1e-5, 1e-5], "seq_len_cross": [672,672,672,672], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    #     "ETTm2": {"root_path": "./dataset/ETT-small", "data_path": "ETTm2", "data": "ETTm2", "task_id": "ETTm2", "variant_num": 7, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 1e-5, "lr_cross": [1e-4, 1e-5, 1e-5, 1e-5], "seq_len_cross": [672,672,672,672], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},

    #     "ECL": {"root_path": "./dataset/electricity", "data_path": "electricity", "data": "custom", "task_id": "ECL", "variant_num": 321, "lr_range": [100, 200, 500], "lr_range_ets": [2, 5, 10], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 3e-4, "lr_cross": [5e-4, 5e-5, 5e-5, 5e-5], "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 64, "d_ff":128, "e_layers": 3, "n_heads": 2},
    #     "Exchange": {"root_path": "./dataset/exchange_rate", "data_path": "exchange_rate", "data": "custom", "task_id": "Exchange", "variant_num": 8, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [50, 100, 200], "K": 0, "lr_ets": 1e-4, "lr_cross": [3e-5, 1e-5, 1e-5, 1e-5], "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 64, "d_ff":128, "e_layers": 3, "n_heads": 2},
    #     "Traffic": {"root_path": "./dataset/traffic", "data_path": "traffic", "data": "custom", "task_id": "traffic", "variant_num": 862, "lr_range": [100, 200, 500], "lr_range_ets": [2, 5, 10], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 1e-3, "lr_cross": [5e-4, 5e-4, 5e-4, 5e-4], "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    #     "Weather": {"root_path": "./dataset/weather", "data_path": "weather", "data": "custom", "task_id": "weather", "variant_num": 21, "lr_range": [5, 10, 20], "lr_range_ets": [2, 5, 10], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 3e-4, "lr_cross": [3e-5, 1e-5, 1e-5, 1e-5], "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    #     "Illness": {"root_path": "./dataset/illness", "data_path": "national_illness", "data": "custom", "task_id": "ili", "variant_num": 7, "lr_range": [5, 10, 20], "lr_range_ets": [2, 5, 10], "lr_range_cross": [2, 5, 10], "K": 1, "lr_ets": 1e-3, "lr_cross": [5e-4, 5e-4, 5e-4, 5e-4], "seq_len_cross": [48,48,60,60], "seg_len_cross": [6,6,6,6], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    # }
    
    mapping = {
        "ETTh1": {"root_path": "./dataset/ETT-small", "data_path": "ETTh1", "data": "ETTh1", "task_id": "ETTh1", "variant_num": 7, "lr_range": [0.5, 2, 5, 10, 20, 50], "lr_range_ets": [5, 10, 20, 50, 100], "lr_range_cross": [100, 200, 300, 500], "K": 3, "lr_ets": 1e-5, "lr_cross": [3e-5]*4, "seq_len_ets": 96, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "ETTh2": {"root_path": "./dataset/ETT-small", "data_path": "ETTh2", "data": "ETTh2", "task_id": "ETTh2", "variant_num": 7, "lr_range": [5, 10, 20, 50, 100, 200, 500], "lr_range_ets": [200, 300, 500, 1000, 1500], "lr_range_cross": [200, 300, 500, 1000], "K": 3, "lr_ets": 1e-5, "lr_cross": [3e-5]*4, "seq_len_ets": 96, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "ETTm1": {"root_path": "./dataset/ETT-small", "data_path": "ETTm1", "data": "ETTm1", "task_id": "ETTm1", "variant_num": 7, "lr_range": [1, 5, 10, 20, 50], "lr_range_ets": [2, 5, 10, 20, 50], "lr_range_cross": [2, 5, 10, 20, 50], "K": 3, "lr_ets": 1e-5, "lr_cross": [1e-4]*4, "seq_len_ets": 96, "seq_len_cross": [672,672,672,672], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "ETTm2": {"root_path": "./dataset/ETT-small", "data_path": "ETTm2", "data": "ETTm2", "task_id": "ETTm2", "variant_num": 7, "lr_range": [5, 10, 20, 50, 100], "lr_range_ets": [50, 100, 200, 500], "lr_range_cross": [10, 20, 50, 100, 200], "K": 3, "lr_ets": 1e-5, "lr_cross": [1e-4]*4, "seq_len_ets": 96, "seq_len_cross": [672,672,672,672], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},

        "ECL": {"root_path": "./dataset/electricity", "data_path": "electricity", "data": "custom", "task_id": "ECL", "variant_num": 321, "lr_range": [100, 200, 500, 1000, 1500], "lr_range_ets": [500, 1000, 1500], "lr_range_cross": [10, 20, 50, 100], "K": 3, "lr_ets": 3e-4, "lr_cross": [5e-4]*4, "seq_len_ets": 336, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 64, "d_ff":128, "e_layers": 3, "n_heads": 2},
        "Exchange": {"root_path": "./dataset/exchange_rate", "data_path": "exchange_rate", "data": "custom", "task_id": "Exchange", "variant_num": 8, "lr_range": [2, 5, 10], "lr_range_ets": [2, 5, 10], "lr_range_cross": [50, 100, 200], "K": 0, "lr_ets": 1e-4, "lr_cross": [3e-5]*4, "seq_len_ets": 336, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "Traffic": {"root_path": "./dataset/traffic", "data_path": "traffic", "data": "custom", "task_id": "traffic", "variant_num": 862, "lr_range": [5, 10, 20, 50], "lr_range_ets": [100, 150, 200, 300, 500], "lr_range_cross": [2, 5, 10], "K": 3, "lr_ets": 1e-3, "lr_cross": [5e-4]*4, "seq_len_ets": 336, "seq_len_cross": [336,720,720,720], "seg_len_cross": [12,24,24,24], "d_model": 64, "d_ff":128, "e_layers": 3, "n_heads": 2},
        "Weather": {"root_path": "./dataset/weather", "data_path": "weather", "data": "custom", "task_id": "weather", "variant_num": 21, "lr_range": [1, 5, 10, 20, 30], "lr_range_ets": [5, 10, 20, 50], "lr_range_cross": [5, 10, 20], "K": 3, "lr_ets": 3e-4, "lr_cross": [3e-5]*4, "seq_len_ets": 336, "seq_len_cross": [336,336,336,720], "seg_len_cross": [12,24,24,24], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
        "Illness": {"root_path": "./dataset/illness", "data_path": "national_illness", "data": "custom", "task_id": "ili", "variant_num": 7, "lr_range": [1, 2, 5, 10, 20, 50, 100], "lr_range_ets": [0.2, 0.5, 1, 2, 5], "lr_range_cross": [5, 10, 20, 50, 70, 100], "K": 1, "lr_ets": 1e-3, "lr_cross": [5e-4]*4, "seq_len_ets": 60, "seq_len_cross": [48,48,60,60], "seg_len_cross": [6,6,6,6], "d_model": 256, "d_ff":512, "e_layers": 3, "n_heads": 4},
    }
    
    root_path = mapping[dataset]["root_path"]
    data_path = mapping[dataset]["data_path"]
    data = mapping[dataset]["data"]
    task_id = mapping[dataset]["task_id"]
    variant_num = mapping[dataset]["variant_num"]
    
    if model == "Crossformer" and dataset == "Traffic" or \
        model == "Crossformer" and dataset == "ECL" and pred_len > 200:
        return {
            "mse": 100,
            "mae": 100,
        }
    
    # 注意需要对于Illness需要把seq_len/label_len/pred_len都改小一些
    seq_len, label_len = 96, 48
    if dataset == "Illness":
        seq_len, label_len = 36, 18
        if pred_len == 96: pred_len = 24
        elif pred_len == 192: pred_len = 36
        elif pred_len == 336: pred_len = 48
        elif pred_len == 720: pred_len = 60
    
    # test_train_num和selected_data_num是两个比较关键的参数
    test_train_num = 200 if dataset == "Illness" else 1000
    selected_data_num = 3 if dataset == "Illness" else 10
    
    # 获取adapted_lr_times
    if model == "ETSformer":
        label_len = 0
        seq_len = mapping[dataset]["seq_len_ets"]
        
        K = mapping[dataset]["K"]
        lr_ets = mapping[dataset]["lr_ets"]
        lr_range = mapping[dataset]["lr_range_ets"]
    elif model == "Crossformer":
        lr_range = mapping[dataset]["lr_range_cross"]
        
        d_model = mapping[dataset]["d_model"]
        d_ff = mapping[dataset]["d_ff"]
        e_layers = mapping[dataset]["e_layers"]
        n_heads = mapping[dataset]["n_heads"]
        
        if dataset == "Illness":
            dropout = 0.6
            if pred_len == 24: pred_len_level = 0
            elif pred_len == 36: pred_len_level = 1
            elif pred_len == 48: pred_len_level = 2
            elif pred_len == 60: pred_len_level = 3
        else:
            dropout = 0.2
            if pred_len == 96: pred_len_level = 0
            elif pred_len == 192: pred_len_level = 1
            elif pred_len == 336: pred_len_level = 2
            elif pred_len == 720: pred_len_level = 3
        
        seq_lens = mapping[dataset]["seq_len_cross"]
        seq_len = seq_lens[pred_len_level]
        
        label_len = 48
        
        seg_lens = mapping[dataset]["seg_len_cross"]
        seg_len = seg_lens[pred_len_level]
        
        lrs = mapping[dataset]["lr_cross"]
        lr_cross = lrs[pred_len_level]
    else:
        lr_range = mapping[dataset]["lr_range"]
    
    # 如果设置的学习率level超出了lr_range的范围，那么不做这个part，直接返回即可。
    if adapt_lr_level >= len(lr_range):
        return {
            "mse": 100,
            "mae": 100,
        }
    
    # 将学习率设置成对应的level
    adapted_lr_times = lr_range[adapt_lr_level]
    
    
    # 读取mse和mae结果
    result_dir = "./mse_and_mae_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # dataset_name = data_path.replace(".csv", "")
    dataset_name = data_path
    file_name = f"RevIN_{model}_{dataset_name}_pl{pred_len}_ttn{test_train_num}_select{selected_data_num}_lr{adapted_lr_times:.2f}.txt"
    file_path = os.path.join(result_dir, file_name)
    
    # 如果结果文件存在，就无需再跑一遍了，不存在的话则再跑一遍：
    if not os.path.exists(file_path):
        # 新建log日志目录
        log_path = f"./all_result/RevIN_{model}_{dataset}_pl{pred_len}"
        if not os.path.exists(log_path): 
            os.makedirs(log_path)
        
        if model == "ETSformer":
            os.system(f"python -u ./run.py   --is_training 1   --root_path {root_path}   \
                --data_path {data_path}.csv   --task_id {task_id}   \
                --model {model}   --data {data}   --features M   \
                --seq_len {seq_len}   --label_len {label_len}   --pred_len {pred_len}   \
                --e_layers 2   --d_layers 2 \
                --enc_in {variant_num}   --dec_in {variant_num}   \
                --c_out {variant_num}   --des 'Exp'   --itr 1  --d_model 512  --train_epochs 1   \
                --K {K}  --learning_rate {lr_ets}  \
                --lradj exponential_with_warmup  --dropout 0.2  --activation 'sigmoid' \
                --gpu {gpu}  --test_train_num {test_train_num}  --selected_data_num {selected_data_num} \
                --adapted_lr_times {adapted_lr_times} \
                --add_revin \
                --run_select_caching > {log_path}/ttn{test_train_num}_select{selected_data_num}_lr{adapted_lr_times:.2f}.log")
        elif model == "Crossformer":
            os.system(f"python -u ./run.py   --is_training 1   --root_path {root_path}   \
                --data_path {data_path}.csv   --task_id {task_id}   \
                --model {model}   --data {data}   --features M   \
                --seq_len {seq_len}   --label_len {label_len}  --pred_len {pred_len}   \
                --seg_len {seg_len}  \
                --e_layers {e_layers}  --n_heads {n_heads} \
                --enc_in {variant_num}   --dec_in {variant_num}   \
                --c_out {variant_num}   --des 'Exp'   --itr 1  --train_epochs 1   \
                --learning_rate {lr_cross}  --d_model {d_model} --d_ff {d_ff} \
                --dropout {dropout} \
                --gpu {gpu}  --test_train_num {test_train_num}  --selected_data_num {selected_data_num} \
                --adapted_lr_times {adapted_lr_times} \
                --add_revin \
                --run_select_caching > {log_path}/ttn{test_train_num}_select{selected_data_num}_lr{adapted_lr_times:.2f}.log")
        else:
            os.system(f"python -u ./run.py   --is_training 1   --root_path {root_path}   \
                --data_path {data_path}.csv   --task_id {task_id}   \
                --model {model}   --data {data}   --features M   \
                --seq_len {seq_len}   --label_len {label_len}   --pred_len {pred_len}   \
                --e_layers 2   --d_layers 1   --factor 3   \
                --enc_in {variant_num}   --dec_in {variant_num}   \
                --c_out {variant_num}   --des 'Exp'   --itr 1  --d_model 512  --train_epochs 1   \
                --gpu {gpu}  --test_train_num {test_train_num}  --selected_data_num {selected_data_num} \
                --adapted_lr_times {adapted_lr_times} \
                --add_revin \
                --run_select_with_distance > {log_path}/ttn{test_train_num}_select{selected_data_num}_lr{adapted_lr_times:.2f}.log")

    # 将结果读出来
    with open(file_path) as f:
        line = f.readline()
        line = line.split(",")
        mse, mae = float(line[0]), float(line[1])

    return {
        "mse": mse,
        "mae": mae,
    }

# @decorator.filtering
# def filter_func(alpha, test_train_num, lambda_reg, model, dataset, gpu):
#     # Filter some parameter combinations you don't want to use.
#     return dataset != 'd3'
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
        model: Parameter(default="Linear", domain=["Linear", "NLinear"]),

        # dataset: Parameter(default="ETTh1", domain=["ETTh1", "ECL", "Traffic", "Exchange", "ETTh2", "ETTm1", "ETTm2", "Weather", "Illness"]),
        dataset: Parameter(default="ETTh1", domain=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "Traffic", "Weather", "Illness"]),

        pred_len: Parameter(default=96, domain=[96, 192, 336, 720]),
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
        "ETTh1": {"root_path": "./dataset/ETT-small", "data_path": "ETTh1", "data": "ETTh1", "task_id": "ETTh1", "variant_num": 7, "lr_range": [0.1, 0.5, 2, 5, 10, 20], "base_lr": 0.005, "batch_size": 32},
        "ETTh2": {"root_path": "./dataset/ETT-small", "data_path": "ETTh2", "data": "ETTh2", "task_id": "ETTh2", "variant_num": 7, "lr_range": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5], "base_lr": 0.05, "batch_size": 32},
        "ETTm1": {"root_path": "./dataset/ETT-small", "data_path": "ETTm1", "data": "ETTm1", "task_id": "ETTm1", "variant_num": 7, "lr_range": [10, 20, 50, 100, 200, 500], "base_lr": 0.0001, "batch_size": 8},
        "ETTm2": {"root_path": "./dataset/ETT-small", "data_path": "ETTm2", "data": "ETTm2", "task_id": "ETTm2", "variant_num": 7, "lr_range": [0.01, 0.03, 0.05, 0.1, 0.5, 1], "base_lr": 0.001, "batch_size": 32},

        "ECL": {"root_path": "./dataset/electricity", "data_path": "electricity", "data": "custom", "task_id": "ECL", "variant_num": 321, "lr_range": [0.1, 0.5, 2, 5, 10, 20], "base_lr": 0.001, "batch_size": 16},
        "Exchange": {"root_path": "./dataset/exchange_rate", "data_path": "exchange_rate", "data": "custom", "task_id": "Exchange", "variant_num": 8, "lr_range": [0.1, 0.5, 2, 5, 10, 20], "base_lr": 0.0005, "batch_size": 8},
        "Traffic": {"root_path": "./dataset/traffic", "data_path": "traffic", "data": "custom", "task_id": "traffic", "variant_num": 862, "lr_range": [0.01, 0.03, 0.05, 0.1, 0.5, 1], "base_lr": 0.05, "batch_size": 16},
        "Weather": {"root_path": "./dataset/weather", "data_path": "weather", "data": "custom", "task_id": "weather", "variant_num": 21, "lr_range": [10, 20, 50, 100, 200, 500], "base_lr": 0.0001, "batch_size": 16},
        "Illness": {"root_path": "./dataset/illness", "data_path": "national_illness", "data": "custom", "task_id": "ili", "variant_num": 7, "lr_range": [0.01, 0.03, 0.1, 0.5, 2, 5], "base_lr": 0.01, "batch_size": 32},
    }
    
    root_path = mapping[dataset]["root_path"]
    data_path = mapping[dataset]["data_path"]
    data = mapping[dataset]["data"]
    task_id = mapping[dataset]["task_id"]
    variant_num = mapping[dataset]["variant_num"]
    base_lr = mapping[dataset]["base_lr"]
    batch_size = mapping[dataset]["batch_size"]
    
    
    # 注意需要对于Illness需要把seq_len/label_len/pred_len都改小一些
    seq_len = 336
    if dataset == "Illness":
        seq_len = 104
        if pred_len == 96: pred_len = 24
        elif pred_len == 192: pred_len = 36
        elif pred_len == 336: pred_len = 48
        elif pred_len == 720: pred_len = 60
    
    # test_train_num和selected_data_num是两个比较关键的参数
    test_train_num = 200 if dataset == "Illness" else 1000
    selected_data_num = 3 if dataset == "Illness" else 10
    
    # 获取adapted_lr_times
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
    file_name = f"{model}_{dataset_name}_pl{pred_len}_ttn{test_train_num}_select{selected_data_num}_lr{adapted_lr_times:.2f}.txt"
    file_path = os.path.join(result_dir, file_name)
    
    # 如果结果文件存在，就无需再跑一遍了，不存在的话则再跑一遍：
    if not os.path.exists(file_path):
        # 新建log日志目录
        log_path = f"./all_result/{model}_{dataset}_pl{pred_len}"
        if not os.path.exists(log_path): 
            os.makedirs(log_path)
        
        os.system(f"python -u run.py --is_training 1 --root_path {root_path} \
            --data_path {data_path}.csv --task_id {task_id} \
            --model {model} --data {data} --features M \
            --seq_len {seq_len} --pred_len {pred_len} \
            --enc_in {variant_num} --dec_in {variant_num} --c_out {variant_num} \
            --des 'Exp' --itr 1 \
            --batch_size {batch_size} --learning_rate {base_lr} \
            --gpu {gpu}  --test_train_num {test_train_num}  --selected_data_num {selected_data_num} \
            --adapted_lr_times {adapted_lr_times} \
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
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import torch
from tqdm import tqdm

# name = "ETTh1"
seq_len = 96
pred_len = 96
device = 'cuda:0'
# seq_len = 24
# pred_len = 24

scale = False
scaler = StandardScaler()

type_map = {'train': 0, 'val': 1, 'test': 2}

def read_data(name, flag):

    # PART I：获取error计算结果
    # PS: how they are calculated?
    # error = pred - true
    # err_mean = np.mean(error)
    # err_var = np.var(error)
    # err_abs_mean = np.mean(np.abs(error))
    # err_abs_var = np.var(np.abs(error))
    # pos_num, neg_num = 0, 0
    # for ei in range(error.shape[0]):
    #     for ej in range(error.shape[1]):
    #         if error[ei][ej] >= 0: pos_num += 1
    #         else: neg_num += 1
    
    # result = [mae, mse, rmse, mape, mspe, err_mean, err_var, err_abs_mean, err_abs_var, pos_num, neg_num]
    results = []
    with open(dir_path + f"pl{pred_len}_{flag}.txt") as f:
        lines = f.readlines()
        for result in lines:
            result = result.split(",")
            result = [float(item) for item in result]
            results.append(result)
    # print(results[:5])
    
    
    # PART2：获取X和Y原数据
    df_raw = pd.read_csv(f"./dataset/{name}.csv")
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    
    set_type = type_map[flag]
    
    # 获取train/val/test的划分：
    if "ETTh" in name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif "ETTm" in name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
    
    border1 = border1s[set_type]
    border2 = border2s[set_type]
    
    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values
        
    final_data = data[border1:border2]
    
    data_x, data_y = [], []
    for i in range(len(final_data)-seq_len-pred_len+1):
        cur_x = final_data[i:i+seq_len]
        cur_y = final_data[i+seq_len: i+seq_len+pred_len]
        data_x.append(cur_x)
        data_y.append(cur_y)
    
    # print(data_x[:5])
    # print(data_y[:5])
    
    
    # X数据为data_x, Y数据为data_y, 其他计算结果在results中

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    results = np.asarray(results)
    # print(data_x.shape, data_y.shape, results.shape)
    # plt.clf()
    # plt.hist(results[:, 5], bins=20)
    # plt.savefig(f"temp/{name}-{flag}.png")
    
    return data_x, data_y, results


for name in ["traffic", "weather", "exchange", "electricity", "ETTh1", "ETTm2"]:
# for name in ["ETTm2"]:
    # if name != 'exchange':
    #     continue
    dir_path = f"./{name}/"

    data_x_train, data_y_train, results_train = read_data(name, "train")
    data_x_val, data_y_val, results_val = read_data(name, "val")
    data_x_test, data_y_test, results_test = read_data(name, "test")

    # metric = 5 取出的是err_mean
    metric = 5

    error_train = results_train[:, metric]
    # mean = np.mean(error_train)
    # std = np.std(error_train)
    # print(f"mean {mean} std {std}")

    data_x = np.concatenate([data_x_train, data_x_val, data_x_test], axis=0)
    data_y = np.concatenate([data_y_train, data_y_val, data_y_test], axis=0)
    data_x_th = torch.FloatTensor(data_x).reshape((len(data_x), -1)).to(device)
    data_y_th = torch.FloatTensor(data_y).reshape((len(data_y), -1)).to(device)
    results = np.concatenate([results_train, results_val, results_test], axis=0)
    results_th = torch.FloatTensor(results).to(device)

    offset = len(data_x_train) + len(data_y_val)
    lookback_length = 1000
    finetune_count = 10
    ks_value_list = []

    for i in tqdm(range(len(results_test))):
        try:
            current_x = data_x_th[i + offset].reshape((1, -1))
            current_y = data_y_th[i + offset].reshape((1, -1))
            lookback_window_x = data_x_th[offset + i - lookback_length - 1 : offset + i - 1]
            distance = torch.norm(lookback_window_x - current_x, dim=1, keepdim=False) / (current_x.shape[1] ** 0.5)
            finetune_inner_index = torch.argsort(distance)[:finetune_count]
            finetune_idx = finetune_inner_index + offset + i - lookback_length - 1
            finetune_error = results_th[finetune_idx, metric].cpu().numpy()
            ks_value = stats.kstest(finetune_error, error_train)
            # ks_value_list.append(ks_value.pvalue)
        except:
            continue

        # print(ks_value, finetune_error)
    
    print(name, 'diff', np.mean(ks_value_list), np.std(ks_value_list), np.mean(np.asarray(ks_value_list) > 0.05))
        # print(len(results[0]), data_x[0].shape, data_y[0].shape)

        # breakpoint()
        
        
        
        
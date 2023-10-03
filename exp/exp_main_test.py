from data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider_at_test_time
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from models.etsformer import ETSformer
from models.crossformer import Crossformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

# ETSformer
from utils.Adam import Adam

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings


import copy
import math


warnings.filterwarnings('ignore')


class Exp_Main_Test(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_Test, self).__init__(args)

        # 这个可以作为超参数来设置
        self.test_train_num = self.args.test_train_num

        # 判断哪些channels是有周期性的
        data_path = self.args.data_path
        if "ETTh1" in data_path: selected_channels = [1,3]  # [1,3, 2,4,5,6]
        # if "ETTh1" in data_path: selected_channels = [7]
        # elif "ETTh2" in data_path: selected_channels = [1,3,7]
        elif "ETTh2" in data_path: selected_channels = [7]
        elif "ETTm1" in data_path: selected_channels = [1,3, 2,4,5]
        elif "ETTm2" in data_path: selected_channels = [1,7, 3]
        elif "illness" in data_path: selected_channels = [1,2, 3,4,5]
        # elif "weather" in data_path: selected_channels = [17,18,19, 5,8,6,13,20]  # [2,3,11]
        elif "weather" in data_path: selected_channels = [17,18,19]
        # elif "weather" in data_path: selected_channels = [5,8,6,13,20]
        # elif "weather" in data_path: selected_channels = [1,4,7,9,10]
        else: selected_channels = list(range(1, self.args.c_out))
        for channel in range(len(selected_channels)):
            selected_channels[channel] -= 1  # 注意这里要读每个item变成item-1，而非item
        
        self.selected_channels = selected_channels

        # 判断各个数据集的周期是多久
        if "ETTh1" in data_path: period = 24
        elif "ETTh2" in data_path: period = 24
        elif "ETTm1" in data_path: period = 96
        elif "ETTm2" in data_path: period = 96
        elif "electricity" in data_path: period = 24
        elif "traffic" in data_path: period = 24
        elif "illness" in data_path: period = 52.142857
        elif "weather" in data_path: period = 144
        elif "Exchange" in data_path: period = 1
        elif "WTH_informer" in data_path: period = 24
        else: period = 1
        self.period = period


    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'ETSformer': ETSformer,
            'Crossformer': Crossformer,
        }
        
        if self.args.model == 'Crossformer':
            model = Crossformer.Model(
                self.args.enc_in, 
                self.args.seq_len, 
                self.args.pred_len,
                self.args.seg_len,
                self.args.win_size,
                self.args.cross_factor,
                self.args.d_model, 
                self.args.d_ff,
                self.args.n_heads, 
                self.args.e_layers,
                self.args.dropout, 
                self.args.baseline,
                self.device
            ).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    # 别忘了这里要加一个用data_provider_at_test_time来提供的data
    def _get_data_at_test_time(self, flag):
        data_set, data_loader = data_provider_at_test_time(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        if self.args.model == 'ETSformer':
            if 'warmup' in self.args.lradj: lr = self.args.min_lr
            else: lr = self.args.learning_rate

            if self.args.smoothing_learning_rate > 0: smoothing_lr = self.args.smoothing_learning_rate
            else: smoothing_lr = 100 * self.args.learning_rate

            if self.args.damping_learning_rate > 0: damping_lr = self.args.damping_learning_rate
            else: damping_lr = 100 * self.args.learning_rate

            nn_params = []
            smoothing_params = []
            damping_params = []
            for k, v in self.model.named_parameters():
                if k[-len('_smoothing_weight'):] == '_smoothing_weight':
                    smoothing_params.append(v)
                elif k[-len('_damping_factor'):] == '_damping_factor':
                    damping_params.append(v)
                else:
                    nn_params.append(v)

            model_optim = Adam([
                {'params': nn_params, 'lr': lr, 'name': 'nn'},
                {'params': smoothing_params, 'lr': smoothing_lr, 'name': 'smoothing'},
                {'params': damping_params, 'lr': damping_lr, 'name': 'damping'},
            ])
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # ? 是否需要对outputs取出最后一段[-self.args.pred_len:]？
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # ? 是否需要对outputs取出最后一段？
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.args.model == 'ETSformer': torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        
        # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model

    def test(self, setting, test=0, flag='test'):
        # test_data, test_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []

        criterion = nn.MSELoss()  # 使用MSELoss
        loss_list = []

        test_time_start = time.time()

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # ? 是否需要对outputs取出最后一段？
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # selected_channels = self.selected_channels
                if self.args.adapt_part_channels:
                    outputs = outputs[:, :, self.selected_channels]
                    batch_y = batch_y[:, :, self.selected_channels]

                # 计算MSE loss
                loss = criterion(outputs, batch_y)
                loss_list.append(loss.item())
                # print(loss)


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        # file_name = f"batchsize_32_{setting}" if flag == 'test' else f"batchsize_1_{setting}"
        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{file_name}.txt", "w") as f:
        #     for loss in loss_list:
        #         f.write(f"{loss}\n")


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        # return
        return loss_list



    def select_with_distance(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        data_len = len(test_data)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        a1, a2, a3, a4 = [], [], [], []
        all_angels = []
        all_distances = []

        error_per_pred_index = [[] for i in range(self.args.pred_len)]

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)


        # 加载模型参数到self.model里
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")
        

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # if data_len - i < self.args.batch_size: break
            if data_len - i < data_len % self.args.batch_size: break
            
            # 从self.model拷贝下来cur_model，并设置为train模式
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                # print(cur_model)
                for n_m, m in cur_model.named_modules():
                    # print(n_m)
                    # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                    if self.args.model == 'ETSformer':
                        linear_layer_name = "decoder.pred"
                    elif self.args.model == 'Crossformer':
                        # 因为decoder一共有e_layers+1层，所以其最后一层是{self.args.e_layers}
                        linear_layer_name = f"decoder.decode_layers.{self.args.e_layers}.linear_pred"
                        # linear_layer_name = "decoder.decode_layers.3.linear_pred"
                    else:
                        linear_layer_name = "decoder.projection"
                    
                    if linear_layer_name in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")

                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
                
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # 使用普通的SGD优化器，而非Adam优化器
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
            

            # tmp loss
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if not self.args.use_nearest_data or self.args.use_further_data:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
            
            # 存储原来的pred和true
            pred_before_adapt = pred.detach().cpu().clone().numpy()
            true_before_adapt = true.detach().cpu().clone().numpy()
            

            # 先用原模型的预测值和标签值之间的error，做反向传播之后得到的梯度值gradient_0
            # 并将这个gradient_0作为标准答案
            # 然后，对测试样本做了adaptation之后，会得到一个gradient_1
            # 那么对gradient_1和gradient_0之间做对比，
            # 就可以得到二者之间的余弦值是多少（方向是否一致），以及长度上相差的距离有多少等等。
            # params_answer = self.get_answer_grad(is_training_part_params, use_adapted_model,
            #                                         lr, test_data, 
            #                                         batch_x, batch_y, batch_x_mark, batch_y_mark,
            #                                         setting)
            if use_adapted_model:
                seq_len = self.args.seq_len
                pred_len = self.args.pred_len
                adapt_start_pos = self.args.adapt_start_pos
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            
            if self.args.adapt_part_channels:
                pred_answer = pred_answer[:, :, self.selected_channels]
                true_answer = true_answer[:, :, self.selected_channels]
            # criterion = nn.MSELoss()  # 使用MSELoss
            # 计算MSE loss
            loss_ans_before = criterion(pred_answer, true_answer)
            loss_ans_before.backward()

            w_T = params[0].grad.T  # 先对weight参数做转置
            b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
            params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
            params_answer = params_answer.ravel()  # 最后再展开成一维的，就得到了标准答案对应的梯度方向

            model_optim.zero_grad()  # 清空梯度


            
            # 选择出合适的梯度
            # 注意：这里是减去梯度，而不是加上梯度！！！！！
            # selected_channels = self.selected_channels

            # 再获得未被选取的unselected_channels
            unselected_channels = list(range(self.args.c_out))
            for item in self.selected_channels:
                unselected_channels.remove(item)
            

            # 在这类我们需要先对adaptation样本的x和测试样本的x之间的距离做对比
            import torch.nn.functional as F
            
            if self.args.adapt_part_channels:  
                test_x = batch_x[:, -seq_len:, self.selected_channels].reshape(-1)
            else:
                test_x = batch_x[:, -seq_len:, :].reshape(-1)
            
            
            distance_pairs = []
            
            if self.args.remove_nearest:
                for ii in range(self.test_train_num):
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])
                # 从其中随机筛选出selected_data_num个样本
                import random
                selected_distance_pairs = random.sample(distance_pairs, self.args.selected_data_num)
            else:
                for ii in range(self.args.test_train_num):
                    # 只对周期性样本计算x之间的距离
                    # if self.args.adapt_cycle:
                    
                    # PS：这里注释掉了adapt_cycle，相当于默认是加的；
                    # 现在改用remove_cycle，如果加了才说明掉周期性；不加则保留
                    if not self.args.remove_cycle:
                        # 为了计算当前的样本和测试样本间时间差是否是周期的倍数
                        # 我们先计算时间差与周期相除的余数
                        if 'illness' in self.args.data_path:
                            import math
                            cycle_remainer = math.fmod(self.args.test_train_num-1 + self.args.pred_len - ii, self.period)
                        else:
                            cycle_remainer = (self.args.test_train_num-1 + self.args.pred_len - ii) % self.period
                        # 定义判定的阈值
                        threshold = self.period // 12
                        # 如果余数在[-threshold, threshold]之间，那么考虑使用其做fine-tune
                        # 否则的话不将其纳入计算距离的数据范围内
                        if cycle_remainer > threshold or cycle_remainer < -threshold:
                            continue
                        
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                        
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])

                # 如果考虑距离计算，那么选距离最小；否则的话就选最近的（也即坐标最大的）：
                if not self.args.remove_distance:
                    # 先按距离从小到大排序
                    cmp = lambda item: item[1]
                    distance_pairs.sort(key=cmp)
                else:
                    cmp = lambda item: item[0]
                    distance_pairs.sort(key=cmp, reverse=True)

                # 筛选出其中最小的selected_data_num个样本出来
                selected_distance_pairs = distance_pairs[:self.args.selected_data_num]
                
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            # print(f"selected_distance_pairs is: {selected_distance_pairs}")

            all_distances.append(selected_distances)


            # 这个数组的内容是否需要预设？
            # params_adapted = torch.zeros((1)).to(self.device)
            cur_grad_list = []
            
            # 开始训练
            for epoch in range(test_train_epochs):

                gradients = []
                accpted_samples_num = set()

                # num_of_loss_per_update = 1
                mean_loss = 0


                for ii in selected_indices:

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # batch_x.requires_grad = True
                    # batch_x_mark.requires_grad = True

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    # 这里当batch_size为1还是32时
                    # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                    # 但是结果的loss值均只包含1个值
                    # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值

                    if self.args.adapt_part_channels:
                        pred = pred[:, :, self.selected_channels]
                        true = true[:, :, self.selected_channels]
                    
                    # 判断是否使用最近的数据
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)
                        # loss = criterion(pred, true)

                    # loss = criterion(pred, true)
                    mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        # # pass
                        # loss.backward()
                        # model_optim.step()

                        loss.backward()
                        w_T = params[0].grad.T
                        b = params[1].grad.unsqueeze(0)
                        params_tmp = torch.cat((w_T, b), 0)
                        original_shape = params_tmp.shape
                        params_tmp = params_tmp.ravel()

                        # 将该梯度存入cur_grad_list中
                        cur_grad_list.append(params_tmp.detach().cpu().numpy())

                        model_optim.zero_grad()

                    # 记录逐样本做了adaptation之后的loss
                    # mean_loss += tmp_loss
                    # mean_loss += loss
            
            
            # 定义一个权重和梯度相乘函数
            def calc_weighted_params(params, weights):
                results = 0
                for i in range(len(params)):
                    results += params[i] * weights[i]
                return results
            
            # 权重分别乘到对应的梯度上
            if weights_given:
                weighted_params = calc_weighted_params(cur_grad_list, weights_given)
            else:
                weights_all_ones = [1 for i in range(self.test_train_num)]
                weighted_params = calc_weighted_params(cur_grad_list, weights_all_ones)
            
            # 将weighted_params从np.array转成tensor
            weighted_params = torch.tensor(weighted_params)
            weighted_params = weighted_params.to(self.device)


            # 计算标准答案的梯度params_answer和adaptation加权后的梯度weighted_params之间的角度
            import math
            product = torch.dot(weighted_params, params_answer)
            product = product / (torch.norm(weighted_params) * torch.norm(params_answer))
            angel = math.degrees(math.acos(product))
            all_angels.append(angel)
            

            # 还原回原来的梯度
            # 也即将weighted_params变回w_grad和b_grad
            weighted_params = weighted_params.reshape(original_shape)
            w_grad_T, b_grad = torch.split(weighted_params, [weighted_params.shape[0]-1, 1])
            w_grad = w_grad_T.T  # (7, 512)
            b_grad = b_grad.squeeze(0)  # (7)


            # 设置新参数为原来参数 + 梯度值
            from torch.nn.parameter import Parameter
            cur_lr = self.args.learning_rate * self.args.adapted_lr_times

            # 将未选择的channels上的梯度置为0
            if self.args.adapt_part_channels:
                w_grad[unselected_channels, :] = 0
                b_grad[unselected_channels] = 0

            # 注意：这里是减去梯度，而不是加上梯度！！！！！
            if self.args.model == 'ETSformer':
                cur_model.decoder.pred.weight = Parameter(cur_model.decoder.pred.weight - w_grad * cur_lr)
                cur_model.decoder.pred.bias = Parameter(cur_model.decoder.pred.bias - b_grad * cur_lr)
            elif self.args.model == 'Crossformer':
                # 因为decoder一共有e_layers+1层，所以其最后一层是{self.args.e_layers}
                adapt_layer = cur_model.decoder.decode_layers[self.args.e_layers].linear_pred
                adapt_layer.weight = Parameter(adapt_layer.weight - w_grad * cur_lr)
                adapt_layer.bias = Parameter(adapt_layer.bias - b_grad * cur_lr)
            else:
                cur_model.decoder.projection.weight = Parameter(cur_model.decoder.projection.weight - w_grad * cur_lr)
                cur_model.decoder.projection.bias = Parameter(cur_model.decoder.projection.bias - b_grad * cur_lr)



            # mean_loss = mean_loss / self.test_train_num
            mean_loss = mean_loss / self.args.selected_data_num
            a2.append(mean_loss.item())
            
            # mean_loss.backward()
            # model_optim.step()


            # seq_len = self.args.seq_len
            # label_len = self.args.label_len
            # pred_len = self.args.pred_len
            # tmp_loss = 0
            # for ii in selected_indices:
            #     pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #         batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
            #         batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
            #     if self.args.adapt_part_channels:
            #         pred = pred[:, :, self.selected_channels]
            #         true = true[:, :, self.selected_channels]
            #     tmp_loss += criterion(pred, true)
            # tmp_loss = tmp_loss / self.args.selected_data_num
            # a3.append(tmp_loss.item())
            a3.append(0)


            # 记录做完adaptation之后的pred/true以及loss信息
            cur_model.eval()
            
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            # else:
            #     # pred, true = self._process_one_batch_with_model(self.model, test_data,
            #     #     batch_x[:, -self.args.seq_len:, :], batch_y, 
            #     #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            #     if not self.args.use_nearest_data or self.args.use_further_data:
            #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #             batch_x[:, -seq_len:, :], batch_y, 
            #             batch_x_mark[:, -seq_len:, :], batch_y_mark)
            #     else:
            #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #             batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
            #             batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)

            # 如果需要筛选部分维度，那么做一次筛选：
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
                
            # 存储做完adaptation之后的pred/true
            pred_after_adapt = pred.detach().cpu().clone().numpy()
            true_after_adapt = true.detach().cpu().clone().numpy()

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
            pred_len = self.args.pred_len
            for index in range(pred_len):
                cur_pred = pred.detach().cpu().numpy()[0][index]
                cur_true = true.detach().cpu().numpy()[0][index]
                cur_error = np.mean((cur_pred - cur_true) ** 2)
                error_per_pred_index[index].append(cur_error)


            if (i+1) % 100 == 0 or (data_len - i) < 100 and (i+1) % 10 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                avg_angel = 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]; avg_angel += all_angels[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num; avg_angel /= num
                print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt, 5.angel_between_answer")
                print("average:", avg_1, avg_2, avg_3, avg_4, avg_angel)
                print("last one:", a1[-1], a2[-1], a3[-1], a4[-1], all_angels[-1])

                printed_selected_channels = [item+1 for item in self.selected_channels]
                print(f"adapt_part_channels: {self.args.adapt_part_channels}")
                print(f"remove_distance: {self.args.remove_distance}, remove_cycle: {self.args.remove_cycle}, remove_nearest: {self.args.remove_nearest}")
                print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
                print(f"selected_distance_pairs are: {selected_distance_pairs}")


            # print("bacth_x.shape:", batch_x.detach().cpu().numpy().shape)
            # print("pred_before_adapt.shape", pred_before_adapt.shape)
            # print("pred_after_adapt.shape", pred_after_adapt.shape)
            interval = 20 if 'illness' in self.args.data_path else 50
            if i % interval == 0:
                # 如果需要画图：
                if self.args.draw_adapt_figure:
                    # 创建画图的文件夹
                    folder_path = './test_results/' + setting + '/'
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    
                    # 由于实际的batch_x长度为ttn+pred_len+seq_len，所以这里一定别忘要截取[:, -seq_len:, :]部分出来！
                    # 除了illness由于实在太短，于是决定多保留一部分
                    if not 'illness' in self.args.data_path:
                        input = batch_x[:, -self.args.seq_len:, :]
                    else:
                        input = batch_x[:, -3*self.args.seq_len:, :]
                        # input = batch_x[:, :, :]
                    input = input.detach().cpu().numpy()
                    # print(batch_x.shape, input.shape, self.args.seq_len)
                    assert (true_before_adapt == true_after_adapt).all()
                    
                    gt = np.concatenate((input[0, :, -1], true_before_adapt[0, :, -1]), axis=0)
                    pd_before_adapt = np.concatenate((input[0, :, -1], pred_before_adapt[0, :, -1]), axis=0)
                    pd_after_adapt = np.concatenate((input[0, :, -1], pred_after_adapt[0, :, -1]), axis=0)
                    
                    name = os.path.join(folder_path, str(i) + '.pdf')
                    plt.figure()
                    # 给ground-truth更大的zorder，保证其在最上面
                    plt.plot(gt, label='GroundTruth', linewidth=2, zorder=3)
                    if pd_before_adapt is not None:
                        plt.plot(pd_before_adapt, label='Before adaptation', linewidth=2, zorder=1)
                    if pd_after_adapt is not None:
                        plt.plot(pd_after_adapt, label='After adaptation', linewidth=2, zorder=2)
                    plt.legend()
                    plt.savefig(name, bbox_inches='tight')

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
        #     for i in range(len(a1)):
        #         t1, t2, t3 = a1[i], a2[i], a3[i]
        #         f.write(f"{t1}, {t2}, {t3}\n")


        # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计的结果输出出来
        mean_error_per_pred_index = [[] for i in range(pred_len)]
        for index in range(pred_len):
            error_i = error_per_pred_index[index]
            total_err = 0
            total_num = 0
            for tmp_err in error_i:
                total_err += tmp_err
                total_num += 1
            mean_error_per_pred_index[index] = total_err / total_num
        # print(mean_error_per_pred_index)


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")


        # # 保存distance和loss信息
        # distance_dir = "./distances_and_loss_diff/" + setting
        # if not os.path.exists(distance_dir):
        #     os.makedirs(distance_dir)

        # if weights_given:
        #     distance_file = f"{distance_dir}/distances_{weights_from}_select{self.args.selected_data_num}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"
        # else:
        #     distance_file = f"{distance_dir}/distances_allones_select{self.args.selected_data_num}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"

        # with open(distance_file, "w") as f:
        #     for i in range(len(a1)):
        #         for ii in range(len(all_distances[i])):
        #             f.write(f"{all_distances[i][ii]}, ")
        #         f.write(f"{a1[i]}, {a3[i]}" + "\n")

        # return a1, a2, a3
        return mse, mae


    def run_KNN(self, setting, test=0):
        print('loading model from checkpoint !!!')
        # self.model.load_state_dict(torch.load(os.paths.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        
        assert self.args.batch_size == 1
        assert self.args.k_value > 0
        
        preds, trues = [], []
        batch_x_list, batch_y_list = [], []
        test_time_start = time.time()

        for flag in ["train_without_shuffle", "val_without_shuffle"]:
            cur_data, cur_loader = self._get_data(flag=flag)
            
            # self.model.eval()
            # with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):
                batch_x = batch_x[0, :, :].float().to(self.device)
                batch_y = batch_y[0, -self.args.pred_len:, :].float().to(self.device)
                batch_x_list.append(batch_x)
                batch_y_list.append(batch_y)
        
        
        for flag in ["test"]:
            cur_data, cur_loader = self._get_data(flag=flag)
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):
                batch_x = batch_x[0, :, :].float().to(self.device)
                batch_y = batch_y[0, -self.args.pred_len:, :].float().to(self.device)
                test_x, test_y = batch_x, batch_y

            
                import torch.nn.functional as F
                
                distance_pairs = []
                data_len = len(batch_x_list)
                # 由于我们实际上只能看到相隔为pred_len以上的数据，所以这里要将data_len减去self.args.pred_len
                if i < self.args.pred_len: 
                    data_len = data_len - i
                else:
                    data_len = data_len - self.args.pred_len
                
                for ii in range(data_len):
                    lookback_x = batch_x_list[ii]
                    dist = F.pairwise_distance(test_x.reshape(1, -1), lookback_x.reshape(1, -1), p=2).item()
                    distance_pairs.append([ii, dist])

                # 先按距离从小到大排序
                cmp = lambda item: item[1]
                distance_pairs.sort(key=cmp)

                # 筛选出其中最小的k_value个样本出来
                selected_distance_pairs = distance_pairs[:self.args.k_value]
                selected_indices = [item[0] for item in selected_distance_pairs]
                selected_distances = [item[1] for item in selected_distance_pairs]
                # print(f"selected_distance_pairs is: {selected_distance_pairs}")

                pred = 0
                for index in selected_indices:
                    pred = pred + batch_y_list[index]
                pred = pred / len(selected_indices)
                true = batch_y
                
                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
            
                mae, mse, rmse, mape, mspe = metric(pred, true)
                
                # 将新样本插入进去
                batch_x_list.append(batch_x)
                batch_y_list.append(batch_y)
                
                if i % 100 == 0:
                    print(f"data {i} have been calculated, cost time: {time.time() - test_time_start}s")
                    print(f"current sample: mse:{mse}, mae:{mae}")
                    tmp_preds = np.array(preds)
                    tmp_preds = tmp_preds.reshape(-1, tmp_preds.shape[-2], tmp_preds.shape[-1])
                    tmp_trues = np.array(trues)
                    tmp_trues = tmp_trues.reshape(-1, tmp_trues.shape[-2], tmp_trues.shape[-1])
                    tmp_mae, tmp_mse, *_ = metric(tmp_preds, tmp_trues)
                    print(f"overall mse:{tmp_mse}, mae:{tmp_mae}")
            
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        return



    def my_test(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 从self.model拷贝下来cur_model，并设置为train模式
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    # 注意：这里的名字应当根据FEDformer模型而修改为"decoder.projection"
                    if self.args.model == 'ETSformer':
                        linear_layer_name = "decoder.pred"
                    elif self.args.model == 'Crossformer':
                        # 因为decoder一共有e_layers+1层，所以其最后一层是{self.args.e_layers}
                        linear_layer_name = f"decoder.decode_layers.{self.args.e_layers}.linear_pred"
                        # linear_layer_name = "decoder.decode_layers.3.linear_pred"
                    else:
                        linear_layer_name = "decoder.projection"
                    
                    if linear_layer_name in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")
                    # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    #     m.requires_grad_(True)
                    #     for n_p, p in m.named_parameters():
                    #         if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                    #             params_norm.append(p)
                    #             names_norm.append(f"{n_m}.{n_p}")

                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim = optim.Adam(params, lr=self.args.learning_rate/30)  # 使用Adam优化器
                # model_optim_norm = optim.Adam(params_norm, lr=self.args.learning_rate*1000 / (2**self.test_train_num))  # 使用Adam优化器

                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                self.model.requires_grad_(True)
                model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.args.test_train_num))
            

            # 开始训练
            # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
            # cur_lr_norm = self.args.learning_rate*1000 / (2**self.test_train_num)
            for epoch in range(test_train_epochs):
                # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
                # cur_lr_norm = self.args.learning_rate*1000 / (2**self.test_train_num)

                gradients = []
                accpted_samples_num = set()

                # for pass_num in range(2):
                #     if pass_num == 1:
                #         cmp = lambda lst: torch.norm(lst[1], lst[2])
                #         gradients.sort(key=cmp)
                #         half_num = self.args.test_train_num // 2
                #         # just get half of samples with smaller gradients
                #         gradients = gradients[:-half_num or None]
                #         for grad in gradients:
                #             accpted_samples_num.add(grad[0])
                        
                
                import random
                is_random = False
                sample_order_list = list(range(self.test_train_num))
                # print("before random, sample_order_list is: ", sample_order_list)
                if is_random:
                    random.shuffle(sample_order_list)
                    # print("after random, sample_order_list is: ", sample_order_list)
                else:
                    # print("do not use random.")
                    pass
                
                mean_loss = 0
                for ii in sample_order_list:
                    # if pass_num == 1 and ii not in accpted_samples_num:
                    #     continue

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # batch_x.requires_grad = True
                    # batch_x_mark.requires_grad = True

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    # from functorch import vmap
                    # from functorch.experimental import replace_all_batch_norm_modules_
                    # replace_all_batch_norm_modules_(cur_model)
                    # tmp_batch_x = batch_x.unsqueeze(1)
                    # tmp_batch_x_mark = batch_x_mark.unsqueeze(1)
                    # vmap_func = vmap(self._process_one_batch_with_model, 
                    #                  in_dims=(None, None, 0, 0, 0, 0), out_dims=(0, 0), 
                    #                  randomness='different')
                    # pred, true = vmap_func(cur_model, test_data,
                    #     tmp_batch_x[:, :, ii : ii+seq_len, :], tmp_batch_x[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     tmp_batch_x_mark[:, :, ii : ii+seq_len, :], tmp_batch_x_mark[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    loss = criterion(pred, true)

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                        # if pass_num == 0:
                        #     gradients.append([ii, torch.norm(params[0].grad), torch.norm(params[1].grad)])
                        # elif pass_num == 1:
                        #     model_optim.step()
                        
                        # model_optim_norm.step()

                    
                    # cur_lr = cur_lr * 2
                    # for param_group in model_optim.param_groups:
                    #     param_group['lr'] = cur_lr
                    
                    # cur_lr_norm = cur_lr_norm * 2
                    # for param_group in model_optim_norm.param_groups:
                    #     param_group['lr'] = cur_lr_norm
        
            cur_model.eval()

            if use_adapted_model:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(self.model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)


            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            if (i+1) % 100 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))

            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return
    
    
    def adapt_whole_model(self, setting, test=0, is_training_part_params=False, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        data_len = len(test_data)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        a1, a2, a3, a4 = [], [], [], []
        all_distances = []

        error_per_pred_index = [[] for i in range(self.args.pred_len)]

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        # 加载模型参数到self.model里
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")
        

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # if data_len - i < self.args.batch_size: break
            if data_len - i < data_len % self.args.batch_size: break
            
            # 从self.model拷贝下来cur_model，并设置为train模式
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                # print(cur_model)
                for n_m, m in cur_model.named_modules():
                    # print(n_m)
                    # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                    if self.args.model == 'ETSformer':
                        linear_layer_name = "decoder.pred"
                    elif self.args.model == 'Crossformer':
                        # 因为decoder一共有e_layers+1层，所以其最后一层是{self.args.e_layers}
                        linear_layer_name = f"decoder.decode_layers.{self.args.e_layers}.linear_pred"
                        # linear_layer_name = "decoder.decode_layers.3.linear_pred"
                    else:
                        linear_layer_name = "decoder.projection"
                    
                    if linear_layer_name in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")

                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                lr = self.args.learning_rate * self.args.adapted_lr_times
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
                
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # 使用普通的SGD优化器，而非Adam优化器
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
            
            
            # tmp loss
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if not self.args.use_nearest_data or self.args.use_further_data:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
            
            
            # 选择channels
            # selected_channels = self.selected_channels

            # 再获得未被选取的unselected_channels
            unselected_channels = list(range(self.args.c_out))
            for item in self.selected_channels:
                unselected_channels.remove(item)
            

            # 在这类我们需要先对adaptation样本的x和测试样本的x之间的距离做对比
            import torch.nn.functional as F
            
            if self.args.adapt_part_channels:  
                test_x = batch_x[:, -seq_len:, self.selected_channels].reshape(-1)
            else:
                test_x = batch_x[:, -seq_len:, :].reshape(-1)
            
            
            distance_pairs = []
            
            if self.args.remove_nearest:
                for ii in range(self.test_train_num):
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])
                # 从其中随机筛选出selected_data_num个样本
                import random
                selected_distance_pairs = random.sample(distance_pairs, self.args.selected_data_num)
            else:
                for ii in range(self.args.test_train_num):
                    # 只对周期性样本计算x之间的距离
                    # if self.args.adapt_cycle:
                    
                    # PS：这里注释掉了adapt_cycle，相当于默认是加的；
                    # 现在改用remove_cycle，如果加了才说明掉周期性；不加则保留
                    if not self.args.remove_cycle:
                        # 为了计算当前的样本和测试样本间时间差是否是周期的倍数
                        # 我们先计算时间差与周期相除的余数
                        if 'illness' in self.args.data_path:
                            import math
                            cycle_remainer = math.fmod(self.args.test_train_num-1 + self.args.pred_len - ii, self.period)
                        else:
                            cycle_remainer = (self.args.test_train_num-1 + self.args.pred_len - ii) % self.period
                        # 定义判定的阈值
                        threshold = self.period // 12
                        # 如果余数在[-threshold, threshold]之间，那么考虑使用其做fine-tune
                        # 否则的话不将其纳入计算距离的数据范围内
                        if cycle_remainer > threshold or cycle_remainer < -threshold:
                            continue
                        
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                        
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])

                # 如果考虑距离计算，那么选距离最小；否则的话就选最近的（也即坐标最大的）：
                if not self.args.remove_distance:
                    # 先按距离从小到大排序
                    cmp = lambda item: item[1]
                    distance_pairs.sort(key=cmp)
                else:
                    cmp = lambda item: item[0]
                    distance_pairs.sort(key=cmp, reverse=True)

                # 筛选出其中最小的selected_data_num个样本出来
                selected_distance_pairs = distance_pairs[:self.args.selected_data_num]
                
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            # print(f"selected_distance_pairs is: {selected_distance_pairs}")

            all_distances.append(selected_distances)


            # 这个数组的内容是否需要预设？
            # params_adapted = torch.zeros((1)).to(self.device)
            cur_grad_list = []
            
            # 开始训练
            for epoch in range(test_train_epochs):
                
                accpted_samples_num = set()

                # num_of_loss_per_update = 1
                mean_loss = 0

                for ii in selected_indices:

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # batch_x.requires_grad = True
                    # batch_x_mark.requires_grad = True

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    # 这里当batch_size为1还是32时
                    # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                    # 但是结果的loss值均只包含1个值
                    # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值

                    if self.args.adapt_part_channels:
                        pred = pred[:, :, self.selected_channels]
                        true = true[:, :, self.selected_channels]
                    
                    # 判断是否使用最近的数据
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)
                        # loss = criterion(pred, true)

                    # loss = criterion(pred, true)
                    mean_loss += loss
                    
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                    # 记录逐样本做了adaptation之后的loss
                    # mean_loss += tmp_loss
                    # mean_loss += loss
            

            # mean_loss = mean_loss / self.test_train_num
            mean_loss = mean_loss / self.args.selected_data_num
            a2.append(mean_loss.item())
            
            # mean_loss.backward()
            # model_optim.step()


            # seq_len = self.args.seq_len
            # label_len = self.args.label_len
            # pred_len = self.args.pred_len
            # tmp_loss = 0
            # for ii in selected_indices:
            #     pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #         batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
            #         batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
            #     if self.args.adapt_part_channels:
            #         pred = pred[:, :, self.selected_channels]
            #         true = true[:, :, self.selected_channels]
            #     tmp_loss += criterion(pred, true)
            # tmp_loss = tmp_loss / self.args.selected_data_num
            # a3.append(tmp_loss.item())
            a3.append(0)



            cur_model.eval()

            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            # else:
            #     # pred, true = self._process_one_batch_with_model(self.model, test_data,
            #     #     batch_x[:, -self.args.seq_len:, :], batch_y, 
            #     #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            #     if not self.args.use_nearest_data or self.args.use_further_data:
            #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #             batch_x[:, -seq_len:, :], batch_y, 
            #             batch_x_mark[:, -seq_len:, :], batch_y_mark)
            #     else:
            #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #             batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
            #             batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)

            # 如果需要筛选部分维度，那么做一次筛选：
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
            pred_len = self.args.pred_len
            for index in range(pred_len):
                cur_pred = pred.detach().cpu().numpy()[0][index]
                cur_true = true.detach().cpu().numpy()[0][index]
                cur_error = np.mean((cur_pred - cur_true) ** 2)
                error_per_pred_index[index].append(cur_error)


            if (i+1) % 100 == 0 or (data_len - i) < 100 and (i+1) % 10 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num
                print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt")
                print("average:", avg_1, avg_2, avg_3, avg_4)
                print("last one:", a1[-1], a2[-1], a3[-1], a4[-1])

                printed_selected_channels = [item+1 for item in self.selected_channels]
                print(f"adapt_part_channels: {self.args.adapt_part_channels}")
                print(f"remove_distance: {self.args.remove_distance}, remove_cycle: {self.args.remove_cycle}, remove_nearest: {self.args.remove_nearest}")
                print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
                print(f"selected_distance_pairs are: {selected_distance_pairs}")


            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
        #     for i in range(len(a1)):
        #         t1, t2, t3 = a1[i], a2[i], a3[i]
        #         f.write(f"{t1}, {t2}, {t3}\n")


        # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计的结果输出出来
        mean_error_per_pred_index = [[] for i in range(pred_len)]
        for index in range(pred_len):
            error_i = error_per_pred_index[index]
            total_err = 0
            total_num = 0
            for tmp_err in error_i:
                total_err += tmp_err
                total_num += 1
            mean_error_per_pred_index[index] = total_err / total_num
        # print(mean_error_per_pred_index)


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")


        # # 保存distance和loss信息
        # distance_dir = "./distances_and_loss_diff/" + setting
        # if not os.path.exists(distance_dir):
        #     os.makedirs(distance_dir)

        # if weights_given:
        #     distance_file = f"{distance_dir}/distances_{weights_from}_select{self.args.selected_data_num}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"
        # else:
        #     distance_file = f"{distance_dir}/distances_allones_select{self.args.selected_data_num}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"

        # with open(distance_file, "w") as f:
        #     for i in range(len(a1)):
        #         for ii in range(len(all_distances[i])):
        #             f.write(f"{all_distances[i][ii]}, ")
        #         f.write(f"{a1[i]}, {a3[i]}" + "\n")

        # return a1, a2, a3
        return mse, mae
    

    def _process_one_batch_with_model(self, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.model == 'Crossformer':
            outputs = model(batch_x)
        else:
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
        
        # if self.args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
        return outputs, batch_y


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'Crossformer':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

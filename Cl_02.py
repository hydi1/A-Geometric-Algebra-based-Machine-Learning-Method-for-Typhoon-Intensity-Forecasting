#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary

import os
import random
import datetime
from complexPyTorch.complexLayers import *
from core_qnn.quaternion_layers import *
from utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm
import time
from thop import profile
start_time1 = time.perf_counter()

#  predition TI of leading time at 24 hours
pre_seq = 4
min_val_loss = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train = pd.read_csv('D:/project/Typhoon/SAF/data/CMA/CMA_train_'+str(pre_seq*6)+'h.csv', header=None)
test= pd.read_csv('D:/project/Typhoon/SAF/data/CMA/CMA_test_'+str(pre_seq*6)+'h.csv', header=None)

CLIPER_feature =  pd.concat((train, test), axis=0)
CLIPER_feature.reset_index(drop=True, inplace=True)

X_wide_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_wide = X_wide_scaler.fit_transform(CLIPER_feature.iloc[:, 5:])
X_wide_train = X_wide[0: train.shape[0], :]

y = y_scaler.fit_transform(CLIPER_feature.loc[:, 3].values.reshape(-1, 1))
y_train = y[0: train.shape[0], :]


# now 6 hours ago  12 hours ago  18 hour ago
ahead_times = [0,1,2,3]

pressures = [1000, 750, 500, 250]

sequential_reanalysis_u_list = []
reanalysis_u_test_dict = {}
X_deep_u_scaler_dict = {}

sequential_reanalysis_v_list = []
reanalysis_v_test_dict = {}
X_deep_v_scaler_dict = {}

reanalysis_type = 'u'
for ahead_time in ahead_times:
    reanalysis_list = []
    for pressure in pressures:
        folder = None
        if ahead_time == 0:
            folder = reanalysis_type
        else:
            folder = reanalysis_type + '_' + str(ahead_time*6)
            
        train_reanalysis_csv = pd.read_csv('D:/project/Typhoon/SAF/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_train_31_31.csv', header=None)
        test_reanalysis_csv = pd.read_csv('D:/project/Typhoon/SAF/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_test_31_31.csv', header=None)

        train_reanalysis = train_reanalysis_csv[train_reanalysis_csv[0].isin(train[0].unique())]
        test_reanalysis = test_reanalysis_csv[test_reanalysis_csv[0].isin(test[0].unique())]
        reanalysis_u_test_dict[reanalysis_type+str(pressure)+str(ahead_time)] = test_reanalysis # 保存test 用于后面测试
        
        reanalysis =  pd.concat((train_reanalysis, test_reanalysis), axis=0)
        reanalysis.reset_index(drop=True, inplace=True)

        scaler_name = reanalysis_type +str(pressure) + str(ahead_time)
        X_deep_u_scaler_dict[scaler_name] = MinMaxScaler()
        
        # 5:end is the 31*31 u component wind speed
        X_deep = X_deep_u_scaler_dict[scaler_name].fit_transform(reanalysis.loc[:, 5:])
        
        # (batch, type, channel, height, widht, time) here type is u
        X_deep_final = X_deep[0: train.shape[0], :].reshape(-1, 1, 1, 31, 31, 1)
        reanalysis_list.append(X_deep_final)

    X_deep_temp = np.concatenate(reanalysis_list[:], axis=2)
    # print("ahead_time:", ahead_time, X_deep_temp.shape)
    sequential_reanalysis_u_list.append(X_deep_temp)

X_deep_u_train = np.concatenate(sequential_reanalysis_u_list, axis=5)


reanalysis_type = 'v'
for ahead_time in ahead_times:
    reanalysis_list = []
    for pressure in pressures:
        folder = None
        if ahead_time == 0:
            folder = reanalysis_type
        else:
            folder = reanalysis_type + '_' + str(ahead_time*6)

        train_reanalysis_csv = pd.read_csv('D:/project/Typhoon/SAF/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_train_31_31.csv', header=None)
        test_reanalysis_csv = pd.read_csv('D:/project/Typhoon/SAF/data/ERA_Interim/'+folder+'/'+reanalysis_type+str(pressure)+'_test_31_31.csv', header=None)

        train_reanalysis = train_reanalysis_csv[train_reanalysis_csv[0].isin(train[0].unique())]
        test_reanalysis = test_reanalysis_csv[test_reanalysis_csv[0].isin(test[0].unique())]
        reanalysis_v_test_dict[reanalysis_type+str(pressure)+str(ahead_time)] = test_reanalysis # 保存test 用于后面测试

        reanalysis =  pd.concat((train_reanalysis, test_reanalysis), axis=0)
        reanalysis.reset_index(drop=True, inplace=True)

        scaler_name = reanalysis_type +str(pressure) + str(ahead_time)
        X_deep_v_scaler_dict[scaler_name] = MinMaxScaler()
        
        # 5:end is the 31*31 v component wind speed
        X_deep = X_deep_v_scaler_dict[scaler_name].fit_transform(reanalysis.loc[:, 5:])
        
        # (batch, type, channel, height, widht, time) here type is v
        X_deep_final = X_deep[0: train.shape[0], :].reshape(-1, 1, 1, 31, 31, 1)
        reanalysis_list.append(X_deep_final)
        
    X_deep_temp = np.concatenate(reanalysis_list[:], axis=2)
    # print("ahead_time:", ahead_time, X_deep_temp.shape)
    sequential_reanalysis_v_list.append(X_deep_temp)

X_deep_v_train = np.concatenate(sequential_reanalysis_v_list, axis=5)


X_deep_train = np.concatenate((X_deep_u_train, X_deep_v_train), axis=1)



class TrainLoader(Data.Dataset):
    def __init__(self, X_wide_train, X_deep_train, y_train):
        self.X_wide_train = X_wide_train
        self.X_deep_train = X_deep_train
        self.y_train = y_train
        
    def __getitem__(self, index):
        return [self.X_wide_train[index], self.X_deep_train[index]], self.y_train[index]
    
    def __len__(self):
        return len(self.X_wide_train)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # spatial attention
        self.att_block_1 = nn.Sequential(
            QuaternionConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            QBatchNorm(32),
            nn.ReLU(inplace=True),
            QuaternionConv(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
        )
        self.att_block_2 = nn.Sequential(
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            QBatchNorm(64),
            nn.ReLU(inplace=True),
            QuaternionConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
        )
        self.att_block_3 = nn.Sequential(
            QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            QBatchNorm(128),
            nn.ReLU(inplace=True),
            QuaternionConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )


        # cross
        self.cross_unit = nn.Parameter(data=torch.ones(len(ahead_times), 6))
        # fuse
        self.fuse_unit = nn.Parameter(data=torch.ones(len(ahead_times), 4))

        self.pool = nn.MaxPool2d(2, 2)
        # deep
        self.qn_conv1 = QuaternionConv(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.qn_conv2 = QuaternionConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.qn_conv3 = QuaternionConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.qn_conv2_fuse = QuaternionConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.qn_conv3_fuse = QuaternionConv(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc1 = QuaternionLinearAutograd(128 * 3 * 3, 128)
        self.fc2 = QuaternionLinearAutograd(96 + 128 * len(ahead_times), 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, wide, deep):
        seq_list = []
        for i in range(len(ahead_times)):
            deep_u = deep[:, 0, :, :, :, i]#维度的意义分别表示：batch_size,U方向的风速，压强，经度 维度，时间步长
            deep_v = deep[:, 1, :, :, :, i]#维度的意义分别表示：batch_size,V方向的风速，压强，经度 维度，时间步长
            #print(deep_u.shape)
            # split 1
            deep_u = self.pool(F.relu(self.qn_conv1(deep_u)))
            deep_u = self.att_block_1(deep_u) * deep_u

            deep_v = self.pool(F.relu(self.qn_conv1(deep_v)))
            deep_v = self.att_block_1(deep_v) * deep_v

            # fuse 1
            time_seq_1 = self.cross_unit[i][0] / (self.cross_unit[i][0] + self.cross_unit[i][1]) * deep_u + \
                         self.cross_unit[i][1] / (self.cross_unit[i][0] + self.cross_unit[i][1]) * deep_v
            time_seq_1 = self.pool(F.relu(self.qn_conv2_fuse(time_seq_1)))
            time_seq_1 = self.att_block_2(time_seq_1) * time_seq_1#和1

            # split 2
            deep_u = self.pool(F.relu(self.qn_conv2(deep_u)))
            deep_u = self.att_block_2(deep_u) * deep_u

            deep_v = self.pool(F.relu(self.qn_conv2(deep_v)))
            deep_v = self.att_block_2(deep_v) * deep_v

            # fuse 2
            time_seq_2 = self.cross_unit[i][2] / (self.cross_unit[i][2] + self.cross_unit[i][3]) * deep_u + \
                         self.cross_unit[i][3] / (self.cross_unit[i][2] + self.cross_unit[i][3]) * deep_v
            time_seq_2 = self.fuse_unit[i][0] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]) * time_seq_1 + \
                         self.fuse_unit[i][1] / (self.fuse_unit[i][0] + self.fuse_unit[i][1]) * time_seq_2
            time_seq_2 = self.pool(F.relu(self.qn_conv3_fuse(time_seq_2)))
            time_seq_2 = self.att_block_3(time_seq_2) * time_seq_2#和3

            # split 3
            deep_u = self.pool(F.relu(self.qn_conv3(deep_u)))
            deep_u = self.att_block_3(deep_u) * deep_u

            deep_v = self.pool(F.relu(self.qn_conv3(deep_v)))
            deep_v = self.att_block_3(deep_v) * deep_v

            # fuse 3
            time_seq = self.cross_unit[i][4] / (self.cross_unit[i][4] + self.cross_unit[i][5]) * deep_u + \
                       self.cross_unit[i][5] / (self.cross_unit[i][4] + self.cross_unit[i][5]) * deep_v#和4
            time_seq = self.fuse_unit[i][2] / (self.fuse_unit[i][2] + self.fuse_unit[i][3]) * time_seq_2 + \
                       self.fuse_unit[i][3] / (self.fuse_unit[i][2] + self.fuse_unit[i][3]) * time_seq
            time_seq = self.att_block_3(time_seq) * time_seq

            time_seq = time_seq.view(-1, 128 * 3 * 3)
            time_seq = self.fc1(time_seq)
            seq_list.append(time_seq)
        wide = wide.view(-1, 96)
        wide_n_deep = torch.cat((wide, seq_list[0]), 1)
        if len(ahead_times) > 1:
            for i in range(1, len(ahead_times)):
                wide_n_deep = torch.cat((wide_n_deep, seq_list[i]), 1)
        wide_n_deep = F.relu(self.fc2(wide_n_deep))
        wide_n_deep = F.relu(self.fc3(wide_n_deep))
        return wide_n_deep


net = Net()


#Xavier 初始化
import torch.nn.init as init
def xavier_init(m):
    if isinstance(m,(nn.Linear,nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias,0)

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
print("Total trainable parameters:", count_param(net))
from torchinfo import summary
summary(net)

#重新实例化模型
net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
net.apply(xavier_init)

batch_size = 128
epochs = 128
lr = 0.0008
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
model_name = os.path.join('24h', 'cl02_24h.pkl')
os.makedirs('24h', exist_ok=True)  # 确保 24h 文件夹存在


import logging
#清除已经存在的日志处理器
logging.getLogger().handlers = []
log_name = 'cl02_9月9'
logging.basicConfig(filename=log_name,level=logging.INFO)
logging.info(f"SAF_Net 原模型 数据集拿到外面分")
logging.info(f"epochs:{epochs};lr:{lr};batch_size={batch_size}")
logging.info(f"损失函数:L1Loss();优化器:Adam")

full_train_index = [*range(0, len(X_wide_train))]
train_index, val_index, _, _, = train_test_split(full_train_index,full_train_index,test_size=0.2)
train_dataset = torch.utils.data.DataLoader(
    TrainLoader(X_wide_train[train_index], X_deep_train[train_index], y_train[train_index]), 
                                                 batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.DataLoader(
    TrainLoader(X_wide_train[val_index], X_deep_train[val_index], y_train[val_index]), 
                                                 batch_size=batch_size, shuffle=True)

for epoch in range(epochs):  # loop over the dataset multiple times
    start_time2 = time.perf_counter()
    # training
    total_train_loss = 0
    train_count = 0
    net.train()
    for step, (batch_x, batch_y) in enumerate(train_dataset):
        if torch.cuda.is_available():
            net.cuda()
            X_wide_train_cuda = batch_x[0].float().cuda()
            X_deep_train_cuda = batch_x[1].float().cuda()
            y_train_cuda = batch_y.float().cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_y = net(X_wide_train_cuda, X_deep_train_cuda)
        loss = criterion(pred_y, y_train_cuda)
        total_train_loss += loss.item()
        train_count +=1
        loss.backward()
        optimizer.step()

    # validating
    total_val_loss = 0
    val_count = 0
    net.eval()
    with torch.no_grad():
        for _,(batch_val_x, batch_val_y) in enumerate(val_dataset):

            if torch.cuda.is_available():
                X_wide_val_cuda = batch_val_x[0].float().cuda()
                X_deep_val_cuda = batch_val_x[1].float().cuda()
                y_val_cuda = batch_val_y.cuda()

            pred_y = net(X_wide_val_cuda, X_deep_val_cuda)
            val_loss = criterion(pred_y, y_val_cuda)
            total_val_loss += val_loss.item()
            val_count += 1
            # print statistics
        if min_val_loss > total_val_loss/val_count:
            torch.save(net.state_dict(), model_name)
            min_val_loss = total_val_loss
        end_time2 = time.perf_counter()
        run_time2 = end_time2 - start_time2
        # print("迭代一次时间", run_time2, "秒")
        print('epochs [%d/%d] train_loss: %.5f val_loss: %.5f' % (epoch + 1, epochs, total_train_loss/train_count, total_val_loss/val_count))
        logging.info(f"Epoch {epoch+1}/{epochs}; train_loss:{total_train_loss/train_count:.5f};val_loss:{total_val_loss/val_count:.5f}")
print('Finished Training')


net.load_state_dict(torch.load(model_name,weights_only=True))

years = test[4].unique()

test_list = []

for year in years:
    temp = test[test[4]==year]
    temp = temp.reset_index(drop=True)
    test_list.append(temp)
    
len(test_list)


with torch.no_grad():
    # 添加一个列表来存储每年的 MAE
    all_mae_list = []
    for year, _test in zip(years, test_list):

        print(year, '年:')
        y_test = _test.loc[:,3]
        y_test_scaler = y_scaler.transform(_test.loc[:,3].values.reshape(-1, 1))
    
        X_wide_test = X_wide_scaler.transform(_test.loc[:,5:])

        final_test_u_list = []
        for ahead_time in ahead_times:
            year_test_list = []
            for pressure in pressures:
                scaler_name = 'u' +str(pressure) + str(ahead_time)
                X_deep = reanalysis_u_test_dict[scaler_name][reanalysis_u_test_dict[scaler_name][0].isin(_test[0].unique())].loc[:,5:]
                X_deep = X_deep_u_scaler_dict[scaler_name].transform(X_deep)
                X_deep_final = X_deep.reshape(-1, 1, 1, 31, 31, 1)
                year_test_list.append(X_deep_final)
            X_deep_temp = np.concatenate(year_test_list, axis=2)
            final_test_u_list.append(X_deep_temp)
        X_deep_u_test = np.concatenate(final_test_u_list, axis=5)
        
        final_test_v_list = []
        for ahead_time in ahead_times:
            year_test_list = []
            for pressure in pressures:
                scaler_name = 'v' +str(pressure) + str(ahead_time)
                X_deep = reanalysis_v_test_dict[scaler_name][reanalysis_v_test_dict[scaler_name][0].isin(_test[0].unique())].loc[:,5:]
                X_deep = X_deep_v_scaler_dict[scaler_name].transform(X_deep)
                X_deep_final = X_deep.reshape(-1, 1, 1, 31, 31, 1)
                year_test_list.append(X_deep_final)
            X_deep_temp = np.concatenate(year_test_list, axis=2)
            final_test_v_list.append(X_deep_temp)
        X_deep_v_test = np.concatenate(final_test_v_list, axis=5)

    
        X_deep_test = np.concatenate((X_deep_u_test, X_deep_v_test), axis=1)
        
        if torch.cuda.is_available():
            X_wide_test = torch.from_numpy(X_wide_test).float().cuda()
            X_deep_test = torch.from_numpy(X_deep_test).float().cuda()
            y_test_scaler = torch.from_numpy(y_test_scaler).float().cuda()
        
        #预测函数损失  
        pred = net(X_wide_test, X_deep_test)
        pred_scaler = pred
        loss_scaler = criterion(pred, y_test_scaler)
        print('loss_scaler:', loss_scaler.item())

        # 计算未归一化损失
        pred = y_scaler.inverse_transform(pred.cpu().detach().numpy().reshape(-1, 1))
        true = y_test.values.reshape(-1, 1)
        diff = np.abs(pred - true)

        mae = sum(diff) / len(diff)
        print('mae wind error:', mae)
        # 将当前年的 MAE 添加到列表中
        all_mae_list.append(mae[0])
        rmse = np.sqrt(sum(diff ** 2) / len(diff))
        print('rmse wind error:', rmse)
        logging.info(f" Year {year};loss_scaler:{loss_scaler.item():.5f}; mae wind error:{mae}; rmse wind error:{rmse}")
# 在循环结束后计算所有年的平均 MAE
    average_mae = np.mean(all_mae_list)
    print('\n所有年的平均 MAE:', average_mae)
logging.info(f"FINISH")
logging.shutdown()
#
# end_time1 = time.perf_counter()
# run_time1 = (end_time1 - start_time1)/60
# print("运行时间",run_time1,"分钟")





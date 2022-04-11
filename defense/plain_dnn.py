# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:59:02 2021

@author: Jacob
"""
from defense.net import Net
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import random
import numpy as np

def train(data, net, task = 'FC', max_epoch = 2000):
    x_train = torch.from_numpy(data['x_train']).float()
    y_train = torch.LongTensor(data['y_train'])

    train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)


    model = net
    model.weight_init()
    optimizer = optim.Adam([{'params':model.parameters(), 'lr':3e-4}], betas=(0.5, 0.999))
    valid_best = 0

    for e in range(max_epoch):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            logit = model(x)
            optimizer.zero_grad()
            if 'FD' in task:
                loss_f = F.mse_loss(logit,x)
            else:
                loss_f = F.cross_entropy(logit,y)
            cost = loss_f
            cost.backward()
            optimizer.step()

        if (e+1) % 10 == 0:
            print(e)
            model.eval()
            if 'FD' in task:
                q_norm = model.cal_q(x_train)
                q_limit =  model.cal_limit(q_norm)
                q_norm_test = model.cal_q(data['x_valid'][data['y_valid']==0,:])
                q_fault_test = model.cal_q(data['x_valid'][data['y_valid']!=0,:])
                norm_acc = (q_norm_test<q_limit).mean()
                fault_acc = (q_fault_test>q_limit).mean()
                net.q_limit = q_limit
                print("loss:{},norm_acc:{},fault_acc:{}".format(cost,norm_acc,fault_acc))                
                if fault_acc > valid_best and norm_acc > 0.95:
                    valid_best = fault_acc
                    torch.save(net,'models/best_temp.pkl')
            else:
                logit = model(torch.from_numpy(data['x_valid']).float())
                prediction = logit.max(1)[1]
                valid_correct = torch.eq(prediction, torch.LongTensor(data['y_valid'])).float().mean().numpy()
                print(valid_correct)
                logit = model(torch.from_numpy(data['x_test']).float())
                prediction = logit.max(1)[1]
                test_correct = torch.eq(prediction, torch.LongTensor(data['y_test'])).float().mean().numpy()
                print(test_correct)
                if valid_correct > valid_best:
                    valid_best = valid_correct
                    torch.save(model,'models/best_temp.pkl')
                
    best_model = torch.load('models/best_temp.pkl')
    return best_model
#%%
# from defense.net import Net,AutoEncoder
# dataset_name = 'TEP_FD'

# data_dir = 'data/' + dataset_name
# data = np.load(data_dir + '/data.npy',allow_pickle=True).item()
# net = AutoEncoder(data,50)
# train(data,net,'FD')
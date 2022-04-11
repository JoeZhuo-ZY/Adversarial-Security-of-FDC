# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 21:14:40 2021

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

def train(data, net, max_epoch = 2000):
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
            optimizer.zero_grad()
            
            h_rep = model.encoder(x)
            xhat = model.decoder(h_rep)
            loss_f = F.mse_loss(xhat,x)
            
            W = model.encoder[-2].weight
            dh = h_rep * (1 - h_rep)
            w_sum = torch.sum(Variable(W)**2, dim=1)
            w_sum = w_sum.unsqueeze(1)
            contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
            
            cost = loss_f + contractive_loss.mul_(1e-2)
            cost.backward()
            optimizer.step()

        print(e)
        model.eval()
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

                
    best_model = torch.load('models/best_temp.pkl')
    return best_model
#%%
# from defense.net import Net,AutoEncoder
# dataset_name = 'TEP_FD'

# data_dir = 'data/' + dataset_name
# data = np.load(data_dir + '/data.npy',allow_pickle=True).item()
# net = AutoEncoder(data,50)
# train(data,net,'FD')
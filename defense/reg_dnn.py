# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:37:23 2021

@author: Jacob
"""

from defense.net import Net
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable,grad
import os
import random
import numpy as np
def l1_regularization(model):
    L1_reg = torch.tensor(0., requires_grad=True)
    for name,param in model.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)
    return L1_reg
def zero_weight_percent(model):
    with torch.no_grad():
        zero_numel = 0
        numel = 0
        for name,param in model.named_parameters():
            if 'weight' in name:
                zero_numel += (torch.abs(param.data)<1e-4).sum()
                numel += param.data.numel()
    return zero_numel/numel


def train(data,net,task='FC',max_epoch=3000):
    x_train = torch.from_numpy(data['x_train']).float()
    y_train = torch.LongTensor(data['y_train'])

    train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)

    model = net
    model.weight_init()
    optimizer = optim.Adam([{'params':model.parameters(), 'lr':3e-4}], betas=(0.5, 0.999))
    global_iter = 0
    valid_best = 0
    if 'FC' in task:        
        train_criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        train_criterion = nn.MSELoss(reduction='none')
    argsnorm = 'Linf'
    fd_order = 'O1'
    h = 1e-2
    tik = 0.05
    for e in range(max_epoch):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            optimizer.zero_grad()
            x.requires_grad_(True)
            logit = model(x)
            if 'FC' in task: 
                lx = train_criterion(logit, y)
            else:
                lx = train_criterion(logit, x)
            loss = lx.mean()
            # l1_loss = l1_regularization(model)
            # cost = fit_loss

            if True:

                dx = grad(loss, x, retain_graph=True)[0]
                sh = dx.shape
                x.requires_grad_(False)

                # v is the finite difference direction.
                # For example, if norm=='L2', v is the gradient of the loss wrt inputs
                v = dx.view(sh[0],-1)
                Nb, Nd = v.shape


                if argsnorm=='L2':
                    nv = v.norm(2,dim=-1,keepdim=True)
                    nz = nv.view(-1)>0
                    v[nz] = v[nz].div(nv[nz])
                if argsnorm=='L1':
                    v = v.sign()
                    v = v/np.sqrt(Nd)
                elif argsnorm=='Linf':
                    vmax, Jmax = v.abs().max(dim=-1)
                    sg = v.sign()
                    I = torch.arange(Nb, device=v.device)
                    sg = sg[I,Jmax]

                    v = torch.zeros_like(v)
                    I = I*Nd
                    Ix = Jmax+I
                    v.put_(Ix, sg)

                v = v.view(sh)
                xf = x + h* v

                mf = model(xf)
                if 'FC' in task: 
                    lf = train_criterion(mf,y)
                else:
                    lf = train_criterion(mf,xf)
                if fd_order=='O2':
                    xb = x - h*v
                    mb = model(xb)
                    if 'FC' in task: 
                        lb = train_criterion(mb,y)
                    else:
                        lb = train_criterion(mb,xb)
                    H = 2*h
                else:
                    H = h
                    lb = lx
                dl = (lf-lb)/H # This is the finite difference approximation
                               # of the directional derivative of the loss


            tik_penalty = torch.tensor(np.nan)
            dlmean = torch.tensor(np.nan)
            dlmax = torch.tensor(np.nan)
            if tik>0:
                dl2 = dl.pow(2)
                tik_penalty = dl2.mean()/2
                loss = loss + tik*tik_penalty

            loss.backward()
            optimizer.step()
            
        print(e)
        model.eval()
        if 'FD' in task:
            q_norm = model.cal_q(x_train)
            q_limit =  model.cal_limit(q_norm)               
            
            q_norm_test = model.cal_q(data['x_valid'][data['y_valid']==0,:])
            q_fault_test = model.cal_q(data['x_valid'][data['y_valid']!=0,:])
            norm_acc = (q_norm_test<q_limit).mean()
            fault_acc = (q_fault_test>q_limit).mean()
            print("loss:{},norm_acc:{},fault_acc:{}".format(loss,norm_acc,fault_acc))
            model.q_limit = q_limit
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
from defense.net import Net,AutoEncoder
dataset_name = 'TEP_FD'

data_dir = 'data/' + dataset_name
data = np.load(data_dir + '/data.npy',allow_pickle=True).item()
net = AutoEncoder(data,50)
train(data,net,'FD')

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:14:49 2021

@author: win10
"""
import sys
sys.path.append("../")
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

output_dim = 16
def one_hot(x, K) :
    return np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
def generate_kappa_schedule():

    kappa_schedule = 15000*[1] # warm-up phase
    kappa_value = 1.0
    step = 0.5/140000

    for i in range(140000):
        kappa_value = kappa_value - step
        kappa_schedule.append(kappa_value)

    return kappa_schedule

def generate_epsilon_schedule(epsilon_train):

    epsilon_schedule = []
    step = epsilon_train/30000

    for i in range(30000):
        epsilon_schedule.append(i*step) #ramp-up phase

    for i in range(100000):
        epsilon_schedule.append(epsilon_train)

    return epsilon_schedule

def interval_based_bound(model, c, bounds, idx):
    model = next(model.children())
    # requires last layer to be linear
    cW = c.t() @ model[-1].weight
    cb = c.t() @ model[-1].bias

    l,u = bounds[-2]
    return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:,None]).t()
def bound_propagation(model, initial_bound):
    l, u = initial_bound
    bounds = []
    model = next(model.children())
    for layer in model:
        if isinstance(layer, nn.Linear):
            l_ = (layer.weight.clamp(min=0) @ l.t() + layer.weight.clamp(max=0) @ u.t()
                  + layer.bias[:,None]).t()
            u_ = (layer.weight.clamp(min=0) @ u.t() + layer.weight.clamp(max=0) @ l.t()
                  + layer.bias[:,None]).t()


        if isinstance(layer, nn.ReLU):
            l_ = l.clamp(min=0)
            u_ = u.clamp(min=0)

        bounds.append((l_, u_))
        l,u = l_, u_
    return bounds

def epoch_robust_bound(loader, model, epsilon_schedule, kappa_schedule, batch_counter, opt=None):
    robust_err = 0
    total_robust_loss = 0
    total_combined_loss = 0

    C = [-torch.eye(output_dim) for _ in range(output_dim)]
    for y0 in range(output_dim):
        C[y0][y0,:] += 1

    for (X,y) in loader:
        ###### fit loss calculation ######
        yp = model(X)
        fit_loss = nn.CrossEntropyLoss()(yp,y)

        ###### robust loss calculation ######
        initial_bound = (X - epsilon_schedule[batch_counter], X + epsilon_schedule[batch_counter])
        bounds = bound_propagation(model, initial_bound)
        robust_loss = 0
        for y0 in range(output_dim):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)
                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y==y0]) / X.shape[0]

                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning

        total_robust_loss += robust_loss.item() * X.shape[0]

        ###### combined losss ######
        combined_loss = kappa_schedule[batch_counter]*fit_loss + (1-kappa_schedule[batch_counter])*robust_loss
        total_combined_loss += combined_loss.item()

        batch_counter +=1


        if opt:
            opt.zero_grad()
            combined_loss.backward()
            opt.step()

    return robust_err / len(loader.dataset), total_combined_loss / len(loader.dataset)
def epoch_calculate_robust_err (loader, model, epsilon):
    robust_err = 0.0

    C = [-torch.eye(output_dim) for _ in range(output_dim)]
    for y0 in range(output_dim):
        C[y0][y0,:] += 1


    for X,y in loader:
        X,y = X, y

        initial_bound = (X - epsilon, X + epsilon)
        bounds = bound_propagation(model, initial_bound)

        for y0 in range(output_dim):
            if sum(y==y0) > 0:
                lower_bound = interval_based_bound(model, C[y0], bounds, y==y0)
                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item() #increment when true label is not winning

    return robust_err / len(loader.dataset)

def train(data,net):
    x_train = torch.from_numpy(data['x_train']).float()
    y_train = torch.LongTensor(data['y_train'])

    train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)

    x_valid = torch.from_numpy(data['x_valid']).float()
    y_valid = torch.LongTensor(data['y_valid'])

    valid_dataset = torch.utils.data.TensorDataset(x_valid,y_valid)
    valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)

    epoch=300
    print_=True

    net.weight_init(_type='kaiming')
    optimizer = optim.Adam([{'params':net.parameters(), 'lr':1e-3}],
                                betas=(0.5, 0.999))
    EPSILON = 0.03
    EPSILON_TRAIN = 0.03
    epsilon_schedule = generate_epsilon_schedule(EPSILON_TRAIN)
    kappa_schedule = generate_kappa_schedule()
    batch_counter = 0
    # global_iter = 0
    best_err = 1000

    for e in range(epoch):
        correct = 0.
        cost = 0.
        total = 0.
        #train
        net.train()
        err, loss = epoch_robust_bound(train_loader,net,epsilon_schedule,kappa_schedule,batch_counter,optimizer)
        print('robust_err:{:.4f},loss:{:.4f}'.format(err,loss))
        batch_counter += len(train_loader)

        if e == 50:  #decrease learning rate after 25 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = 5e-4

        if e == 150:  #decrease learning rate after 41 epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4
        #test
        net.eval()
        correct = 0.
        cost = 0.
        total = 0.
        test_err = epoch_calculate_robust_err(valid_loader,net,EPSILON)
        for batch_idx, (x, y) in enumerate(valid_loader):
            x = Variable(x)
            y = Variable(y)
            logit = net(x)
            prediction = logit.max(1)[1]
            correct += torch.eq(prediction, y).float().sum().numpy()
            cost += F.cross_entropy(logit, y, size_average=False).detach().numpy()
            total += x.size(0)
        accuracy = correct / total
        cost /= total
        if print_:
            print('[{:03d}]\nTEST RESULT'.format(e))
            print('ACC:{:.4f}, Robust_err:{:.4f}'.format(accuracy,test_err))

        if  best_err > test_err:
            best_err = test_err
            torch.save(net,'models/best_temp.pkl')
            print("=> saved checkpoint (iter {})".format(batch_counter))


    print(" [*] Training Finished!")
    return net
#%% train
# from defense.net import Net
# dataset_name = 'TEP'
# data_dir = 'data/' + dataset_name
# data = np.load(data_dir + '/data.npy',allow_pickle=True).item()
# net = Net('TEP',50,16)
# train(data,net)
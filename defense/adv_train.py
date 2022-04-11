import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from attack.FGSM import FGSM
from attack.PGD_Linf import PGD_Linf
from utils import FDLoss

def train(data,net,task = 'FC', method='FGSM'):
    x_train = torch.from_numpy(data['x_train']).float()
    y_train = torch.LongTensor(data['y_train'])

    train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)
    max_epoch = 1000

    model = net
    model.weight_init()
    optimizer = optim.Adam([{'params':model.parameters(), 'lr':3e-4}], betas=(0.5, 0.999))
    global_iter = 0
    valid_best = 0
    if 'FD' in task:
        loss_adv = FDLoss()
    else:
        loss_adv = nn.CrossEntropyLoss()
        
    for e in range(0, max_epoch):
        correct = 0.
        loss_f = 0.
        total = 0.
        correct_train = 0.
        cost_train = 0.
        total_train = 0.
        # train
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            # print(y)
            # print(list(x.size()))

            model_cp = copy.deepcopy(model)
            for p in model_cp.parameters():
                p.requires_grad = False
            model_cp.eval()

            if 'FGSM' in method:
                adv_generator = FGSM(model_cp,x,y)
                param = {'eps':0.03, 'loss_fn':loss_adv}
            elif 'PGD' in method:
                adv_generator = PGD_Linf(model_cp,x,y)
                param = {'eps':0.03,'eps_iter':0.002, 'nb_iter':20, 'loss_fn':loss_adv}
                
            adv_x,_,_ = adv_generator.generate(param)
            adv_logit = model(adv_x)
            adv_prediction = adv_logit.max(1)[1]

            logit = model(x)

            # correct = (torch.eq(prediction, y).float().mean().numpy() + torch.eq(adv_prediction, y).float().mean().numpy()) / 2

            if 'FD' in task:
                loss_f = (F.mse_loss(logit, x) + F.mse_loss(adv_logit, adv_x)) / 2 
            else:
                loss_f = (F.cross_entropy(logit, y) + F.cross_entropy(adv_logit, y)) / 2                

            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()

        # test

        print(e)
        model.eval()
        model_cp.eval()
        # param = {'eps':0.03, 'loss_fn':loss_adv}
        if 'FGSM' in method:
            adv_generator = FGSM(model_cp,data['x_valid'],data['y_valid'])
            qadv_generator = FGSM(model_cp,x_train,y_train)
        elif 'PGD' in method:
            adv_generator = PGD_Linf(model_cp,data['x_valid'],data['y_valid'])
            qadv_generator = PGD_Linf(model_cp,x_train,y_train)
            
        adv_x,_,_ = adv_generator.generate(param)
        qadv_x,_,_ = qadv_generator.generate(param)
        
        if 'FD' in task:
            q_norm = model.cal_q(torch.cat([x_train,qadv_x],0))
            q_limit =  model.cal_limit(q_norm)
            q_valid = model.cal_q(adv_x)
            valid_correct = ((q_valid > q_limit) & (data['y_valid']==1)) | ((q_valid < q_limit) & (data['y_valid']==0))
            valid_correct = np.mean(valid_correct)
            
            q_norm_test = model.cal_q(data['x_valid'][data['y_valid']==0,:])
            q_fault_test = model.cal_q(data['x_valid'][data['y_valid']!=0,:])
            norm_acc = (q_norm_test<q_limit).mean()
            fault_acc = (q_fault_test>q_limit).mean()
            print("loss:{},norm_acc:{},fault_acc:{}".format(loss_f,norm_acc,fault_acc))
            model.q_limit = q_limit
        else:
            adv_logit = model(adv_x)
            adv_prediction = adv_logit.max(1)[1]
            valid_correct = torch.eq(adv_prediction, torch.LongTensor(data['y_valid'])).float().mean().numpy()
        
        print('valid_adv_acc:%.4f'%valid_correct)
        if valid_correct > valid_best:
            valid_best = valid_correct            
            torch.save(model,'models/best_temp.pkl')

    best_model = torch.load('models/best_temp.pkl')
    print(" [*] Training Finished!")
    return best_model
#%%
# from defense.net import Net,AutoEncoder
# dataset_name = 'TEP_FD'

# data_dir = 'data/' + dataset_name
# data = np.load(data_dir + '/data.npy',allow_pickle=True).item()
# net = AutoEncoder(data,50)
# train(data,net,'FD', method='FGSM')



# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:10:20 2021

@author: win10
"""
import torch.nn as nn
# import tensorflow as tf
import numpy as np
import numpy.linalg as la
# import cvxpy as cp
import torch
from defense.net import Net,AutoEncoder

class BasicAdaptor():
    """
        The adaptor for our basic framework
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = 'data/' + dataset
        self.net_dir = 'models/' + dataset
        self.data = np.load(self.data_dir + '/data.npy',allow_pickle=True).item()
        self.input_dim = self.data['x_train'].shape[1]
        self.output_dim = len(np.unique(self.data['y_train']))
        if 'FD' in dataset:
            self.net = AutoEncoder(dataset,self.input_dim)
            self.task = 'FD'
        else:
            self.net = Net(dataset,self.input_dim,self.output_dim)
            self.task = 'FC'

class PlainDNNAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(PlainDNNAdaptor, self).__init__(dataset)

    def train(self):
        from defense.plain_dnn import train
        model = train(self.data,self.net, task = self.task)
        torch.save(model, self.net_dir + '/plain_dnn/model.pkl')

class FGSMTrainAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(FGSMTrainAdaptor, self).__init__(dataset)

    def train(self):
        from defense.adv_train import train
        model = train(self.data,self.net,'FGSM')
        torch.save(model, self.net_dir + '/FGSM_train/model.pkl')

class PGDTrainAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(PGDTrainAdaptor, self).__init__(dataset)

    def train(self):
        from defense.adv_train import train
        model = train(self.data,self.net,'PGD')
        torch.save(model, self.net_dir + '/PGD_train/model.pkl')

class DistillationAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(DistillationAdaptor, self).__init__(dataset)

    def train(self):
        from defense.distillation import Train_distillation
        # teacher = torch.load(self.net_dir + '/plain_dnn/model.pkl')
        if 'FD' in self.task: 
            raise ValueError('The Distillation is not adapted for dection')
        else:
            teacher, student = Train_distillation(self.data,self.net)
            torch.save(teacher, self.net_dir + '/distillation/model_t.pkl')
            torch.save(student, self.net_dir + '/distillation/model.pkl')

class QuantAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(QuantAdaptor, self).__init__(dataset)

    def train(self,source_name='plain_dnn'):
        from defense.net import Net_quant,AutoEncoder_quant

        source_net = torch.load(self.net_dir + '/' +source_name + '/model.pkl')
        if 'FD' in self.task:
            new_net = AutoEncoder_quant(self.dataset,self.input_dim,16)
            AutoEncoder_quant.encoder = source_net.encoder
            AutoEncoder_quant.decoder = source_net.decoder
            AutoEncoder_quant.q_limit = source_net.q_limit
        else:
            new_net = Net_quant(self.dataset,self.input_dim,self.output_dim,8)
            new_net.mlp = source_net.mlp
        torch.save(new_net, self.net_dir + '/quant/model.pkl')

class IBPDNNAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(IBPDNNAdaptor, self).__init__(dataset)

    def train(self):
        from defense.IBP_train import train
        if 'FD' in self.task: 
            raise ValueError('The IBP is not adapted for dection')
        else:
            model = train(self.data,self.net)
            torch.save(model, self.net_dir + '/IBP_train/model.pkl')

class RegDNNAdaptor(BasicAdaptor):
    def __init__(self, dataset):
        super(RegDNNAdaptor, self).__init__(dataset)

    def train(self):
        from defense.reg_dnn import train
        model = train(self.data,self.net)
        torch.save(model, self.net_dir + '/reg_dnn/model.pkl')
        
class CAEAdaptor(BasicAdaptor):    
    def __init__(self, dataset):
        super(CAEAdaptor, self).__init__(dataset)

    def train(self):    
        if 'FD' in self.task: 
            from defense.CAE import train
            model = train(self.data,self.net)
            torch.save(model, self.net_dir + '/CAE/model.pkl')
        else:
            raise ValueError('The CAE is not adapted for classification')
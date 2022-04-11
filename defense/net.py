# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:56:15 2020

@author: Jacob
"""


import torch.nn as nn
import torch
import numpy as np
from scipy import stats

class Net(nn.Module):
    def __init__(self, dataset, x_dim, y_dim):
        super(Net, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        if 'TEP' in dataset:
            self.mlp = nn.Sequential(
                nn.Linear(self.x_dim, 64),
                nn.ReLU(True),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Linear(64, self.y_dim)
                )

    def forward(self, X):
        return self.mlp(X)

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())

class Net_quant(nn.Module):
    def __init__(self, dataset, x_dim, y_dim,depth):
        super(Net_quant, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.depth = depth
        if 'TEP' in dataset:
            self.mlp = nn.Sequential(
                nn.Linear(self.x_dim, 64),
                nn.ReLU(True),
                nn.Linear(64, 64),
                nn.ReLU(True),
                nn.Linear(64, self.y_dim)
                )

    def _quantize(self, im):
        N = int(pow(2, self.depth))
        im = (im * N).floor()
        im = im / N
        return im

    def forward(self, X):
        X = self._quantize(X)
        return self.mlp(X)

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())


class AutoEncoder(nn.Module):
    def __init__(self, dataset, x_dim):
        super(AutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.q_limit = 0

        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU()
        )

        self.decoder =  nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,self.x_dim)
        )


    def forward(self, x):
        h1 = self.encoder(x)
        xhat = self.decoder(h1)
        return xhat

    def cal_q(self, x):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        x_hat = self.decoder(self.encoder(x))
        sqe=(x-x_hat)**2
        # sqe=sqe.detach().numpy()
        q = torch.sum(sqe,axis = 1)
        return q

    def cal_limit(self, q_normal):
        q_normal = q_normal.detach()
        mean_q = torch.mean(q_normal)
        std_q = torch.std(q_normal)**2
        freedom = (2*(mean_q**2))/std_q
        chi_lim = stats.chi2.ppf(0.99,freedom)
        q_limit = std_q/(2*mean_q)*chi_lim
        return q_limit

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())

class AutoEncoder_quant(nn.Module):
    def __init__(self, dataset, x_dim, depth):
        super(AutoEncoder_quant, self).__init__()
        self.x_dim = x_dim
        self.q_limit = 0
        self.depth = depth

        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU()
        )

        self.decoder =  nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,self.x_dim)
        )

    def forward(self, x):
        x = self._quantize(x)
        h1 = self.encoder(x)
        xhat = self.decoder(h1)
        return xhat

    def cal_q(self, x):
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        x_hat = self.decoder(self.encoder(x))
        sqe=(x-x_hat)**2
        # sqe=sqe.detach().numpy()
        q = torch.sum(sqe,axis = 1)
        return q

    def cal_limit(self, q_normal):
        q_normal = q_normal.detach()
        mean_q = torch.mean(q_normal)
        std_q = torch.std(q_normal)**2
        freedom = (2*(mean_q**2))/std_q
        chi_lim = stats.chi2.ppf(0.99,freedom)
        q_limit = std_q/(2*mean_q)*chi_lim
        return q_limit

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())

    def _quantize(self, im):
        N = int(pow(2, self.depth))
        im = (im * N).floor()
        im = im / N
        return im

def xavier_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:56:15 2020

@author: Jacob
"""


import torch.nn as nn

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

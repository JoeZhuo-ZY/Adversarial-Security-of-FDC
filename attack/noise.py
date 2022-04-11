# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:14:19 2021

@author: win10
"""
import numpy as np
import torch
import torch.nn as nn

class Noise():
    def __init__(self,model,x,y):
        super().__init__()
        self.model=model
        self.x = torch.from_numpy(x).float()
        self.y = torch.LongTensor(y)

    def generate(self, eps, max_iter = 20):
        advx_list = []
        for i,(xi,yi,ei) in enumerate(zip(self.x,self.y,eps)):
            itr = 0
            correct = True
            while itr < 20 and correct:
                ori_x = xi.clone()
                eta = torch.FloatTensor(*xi.shape).uniform_(-ei, ei)
                adv_x = ori_x + eta
                adv_x = torch.clamp(adv_x, min=0, max=1)
                prediction = self.model(adv_x).max(0)[1]
                correct = torch.eq(prediction, yi).numpy()
                itr += 1
            advx_list.append(adv_x)
            # print(i)
        adv_x = torch.stack(advx_list)
        return adv_x
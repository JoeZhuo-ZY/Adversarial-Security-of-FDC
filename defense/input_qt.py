# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:50:44 2021

@author: Jacob
"""


def add_transform(source_net,new_net):
    new_net.mlp = source_net.mlp
    return new_net
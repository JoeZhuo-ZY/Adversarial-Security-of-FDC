# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:54:04 2021

@author: win10
"""
import torch
import torch.nn as nn
import numpy as np

class FDLoss(nn.Module):
    def __init__(self):
        super(FDLoss,self).__init__()

    def forward(self, x, xhat, y):
        mask = -2 * y + 1 * torch.ones_like(y) # label 0 for 1, label 1 for -1
        return (nn.functional.mse_loss(x,xhat,reduction='none').mean(1) * mask).mean()

def _quantize(im, depth=8):
    if torch.is_tensor(im):
        N = int(pow(2, depth))
        im = (im * N).floor()
        im = im / N
    else:
        N = int(pow(2, depth))
        im = np.floor((im * N))
        im = im / N
    return im

class BasicAdaptor():
    """
        The adaptor for our basic framework
    """

    def __init__(self, dataset, model):
        self.model_name = model
        self.dataset_name = dataset
        self.data_dir = 'data/' + dataset
        self.model_dir = 'models/' + dataset + '/' + model
        self.data = np.load(self.data_dir + '/data.npy',allow_pickle=True).item()
        self.model = torch.load(self.model_dir + '/model.pkl')
        self.x = self.data['x_test']
        self.y = self.data['y_test']
        if 'FD' in self.dataset_name:
            self.task = 'FD'
            # self.q_limit = self.model.cal_limit(self.model.cal_q(torch.from_numpy(self.data['x_train']).float()))
            self.q_limit = self.model.q_limit
            self.correct_func = self._fd_correct
            self.loss_fn = FDLoss()
        else:
            self.task = 'FC'
            self.correct_func = self._fc_correct
            self.loss_fn = nn.CrossEntropyLoss()

    def _fc_correct(self, test_x, test_y, offset = 0):
        if not torch.is_tensor(test_x):
            test_x = torch.FloatTensor(test_x)
        prediction = self.model(test_x).max(1)[1]
        test_correct = torch.eq(prediction, torch.LongTensor(test_y)).numpy()
        return test_correct

    def _fd_correct(self, test_x, test_y, offset = 0):
        q_test = self.model.cal_q(test_x)
        if not torch.is_tensor(test_y):
            test_y = torch.LongTensor(test_y)
        q_test[test_y==1] -= offset
        q_test[test_y==0] += offset
        test_correct = ((q_test > self.q_limit) & (test_y==1)) | ((q_test < self.q_limit) & (test_y==0))
        return test_correct.numpy()

    def get_advs(self, new_cal = False):
        res_file_dir = self.model_dir + self.res_dir +'.npy'
        adv_x = np.load(res_file_dir, allow_pickle=True)
        return adv_x

    def get_adv_info(self,eps, new_cal=False):
        adv_x = self.get_advs(new_cal)
        if not torch.is_tensor(adv_x):
            adv_x = torch.FloatTensor(adv_x)

        if self.model_name in 'quant':
            adv_x = _quantize(adv_x)

        if 'FC' in self.task:
            logits = self.model(adv_x)
            offset = 5e-3 * torch.ones_like(logits)
            for i in range(offset.shape[0]):
                offset[i,int(self.y[i])]=0
            logits += offset
            correct = torch.eq(logits.max(1)[1], torch.LongTensor(self.y)).numpy()

            if self.model_name not in 'quant' and 'UAP' not in self.res_dir:
                assert ~correct.any()
        else:
            correct = self.correct_func(adv_x,self.y,1e-4)

        epsilon = abs(adv_x - self.x).numpy()
        eps_inf = np.max(epsilon,1)
        eps_inf[correct] = 1
        if 'UAP' in self.res_dir:
            robust_index = correct
        else:
            robust_index = eps_inf > eps
        adv_acc = np.mean(robust_index)

        adv_y = np.copy(self.y)
        if 'FC' in self.task:
            adv_y[~robust_index] = logits[~robust_index].max(1)[1]
        else:
            adv_y[~robust_index] = 1 - adv_y[~robust_index]

        # radius on clean correct set
        clean_correct = self.correct_func(self.data['x_test'],self.y)
        epsilon_crr = eps_inf[clean_correct&~correct]
        radius = np.mean(epsilon_crr)
        return adv_acc, radius, eps_inf, epsilon_crr, adv_y



class CleanAdaptor(BasicAdaptor):
    """
        ** Not a real attack **
        Clean predictor
    """
    def __init__(self, dataset, model):
        super(CleanAdaptor, self).__init__(dataset, model)
        self.res_dir = '/clean' #dummy, for consistence

    def verify(self):
        if self.model_name not in 'quant':
            self.x = _quantize(self.x)
        test_correct = self.correct_func(self.x,self.data['y_test'])
        return np.mean(test_correct)



class AttackAdaptor(BasicAdaptor):
    # Supported attack_name:
        #noise, FGSM, PGD, CWinf, Milp, DeepFool, UAP0.xx, SPSA
    def __init__(self, dataset, model, attack_name):
        super(AttackAdaptor, self).__init__(dataset, model)
        self.res_dir = '/' + attack_name
        if 'UAP' in attack_name:
            attack_name = 'UAP'
        if self.model_name in 'quant':
            if attack_name in ['CWinf','DeepFool','UAP']:
                self.model_dir = 'models/' + self.dataset_name + '/' + 'plain_dnn'




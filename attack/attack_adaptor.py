# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:54:04 2021

@author: win10
"""
import torch
import torch.nn as nn
import numpy as np
import os.path
from attack.noise import Noise
from attack.FGSM import FGSM
from attack.PGD_Linf import PGD_Linf
from attack.SPSA import SPSA

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

    def get_advs(self, use_cache = True, eps = None):
        if 'UAP' in self.res_dir:
            rdir = self.res_dir + '_' + str(eps) + '.npy'
        else:
            rdir = self.res_dir + '.npy'
        res_file_dir = self.model_dir + rdir
        if os.path.isfile(res_file_dir) and use_cache:
            print('use cache')
            adv_x = np.load(res_file_dir, allow_pickle=True)
        else:
            adv_x = self.verify()
        return adv_x

    def get_adv_info(self,eps, use_cache = True):
        adv_x = self.get_advs(use_cache,eps)
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

    def get_y(self):
        if self.model_name not in 'quant':
                self.x = _quantize(self.x)
        test_x = self.x
        if 'FC' in self.task:
            if not torch.is_tensor(test_x):
                test_x = torch.FloatTensor(test_x)
            pred_y = self.model(test_x).max(1)[1]
            return pred_y
        else:
            test_y = self.y
            q_test = self.model.cal_q(test_x)
            if not torch.is_tensor(test_y):
                test_y = torch.LongTensor(test_y)
            q_test[q_test > self.q_limit] = 1
            q_test[q_test < self.q_limit] = 0
            return q_test.detach()

    def get_logit(self):
        if self.model_name not in 'quant':
                self.x = _quantize(self.x)
        test_x = self.x
        if 'FC' in self.task:
            if not torch.is_tensor(test_x):
                test_x = torch.FloatTensor(test_x)
            logit = self.model(test_x)
            return logit.detach()
        else:
            test_y = self.y
            logit = self.model.cal_q(test_x)
            return logit.detach()

    def fault_clean_acc(self):
        q_test = self.model.cal_q(self.x)
        test_y = torch.LongTensor(self.data['y_test'])
        correct = (q_test > self.q_limit) & (test_y==1)
        return correct.numpy().mean()

class NoiseAdaptor(BasicAdaptor):

    def __init__(self, dataset, model):
        super(NoiseAdaptor, self).__init__(dataset, model)
        self.res_dir = '/noise'

    def verify(self):
        l = 0.0 * np.ones_like(self.y).astype(np.float32)
        r = 0.5 * np.ones_like(self.y).astype(np.float32)
        error = 1e-3
        noise = Noise(self.model,self.x,self.y)
        adv_x_all =  np.copy(self.x)
        while max(r-l) > error:
            mid = (l + r) / 2.0
            adv_x = noise.generate(eps=mid)
            correct = self.correct_func(adv_x,self.y)
            r[~correct] = mid[~correct]
            l[correct] = mid[correct]
            adv_x_all[~correct] = adv_x[~correct]
        # assert adv_x_all.any(axis=1).all()
        np.save(self.model_dir + self.res_dir,adv_x_all)
        return adv_x_all

    def verify_ineps(self,eps):

        noise = Noise(self.model,self.x,self.y)
        adv_x = noise.generate(eps = eps * np.ones_like(self.y).astype(np.float32))
        prediction = self.model(adv_x).max(1)[1]
        correct = torch.eq(prediction, torch.LongTensor(self.y)).numpy()
        return np.mean(correct)


class FGSMAdaptor(BasicAdaptor):

    def __init__(self, dataset, model):
        super(FGSMAdaptor, self).__init__(dataset, model)
        self.res_dir = '/FGSM'

    def verify(self):
        if self.model_name in 'quant':
            self.model =  torch.load('models/' + self.dataset_name + '/plain_dnn/model.pkl')
        l = 0.0 * np.ones_like(self.y).reshape(-1,1).astype(np.float32)
        r = 0.41 * np.ones_like(self.y).reshape(-1,1).astype(np.float32)
        error = 1e-3
        fgsm = FGSM(self.model,self.x,self.y)
        adv_x_all = np.copy(self.x)
        while max(r-l) > error:
            mid = (l + r) / 2.0
            adv_x,_,_ = fgsm.generate({'eps':mid,'loss_fn':self.loss_fn})
            if self.model_name in 'quant':
                adv_x = _quantize(adv_x)
            correct = self.correct_func(adv_x,self.y)
            r[~correct] = mid[~correct]
            l[correct] = mid[correct]
            adv_x_all[~correct] = adv_x[~correct]
        # assert adv_x_all.any(axis=1).all()
        np.save(self.model_dir + self.res_dir,adv_x_all)
        return adv_x_all

    def verify_ineps(self,eps):
        if self.model_name in 'quant':
            self.model =  torch.load('models/' + self.dataset_name + '/plain_dnn/model.pkl')
        fgsm = FGSM(self.model,self.x,self.y)
        adv_x,_,_ = fgsm.generate({'eps':eps,'loss_fn':self.loss_fn})
        if self.model_name in 'quant':
            adv_x = _quantize(adv_x)
        correct = self.correct_func(adv_x,self.y)
        return np.mean(correct)

class PGDAdaptor(BasicAdaptor):

    def __init__(self, dataset, model):
        super(PGDAdaptor, self).__init__(dataset, model)
        self.res_dir = '/PGD'

    def verify(self):
        if self.model_name in 'quant':
            self.model =  torch.load('models/' + self.dataset_name + '/plain_dnn/model.pkl')
        params = {'eps': 0.03,'eps_iter':0.001, 'nb_iter':300,'loss_fn':self.loss_fn}
        l = 0.0 * np.ones_like(self.y).reshape(-1,1).astype(np.float32)
        r = 0.31 * np.ones_like(self.y).reshape(-1,1).astype(np.float32)
        error = 1e-3
        pgd = PGD_Linf(self.model,self.x,self.y)
        adv_x_all =  np.copy(self.x)
        while max(r-l) > error:
            mid = (l + r) / 2.0
            params['eps'] = mid
            adv_x,_,_ = pgd.generate(params)
            if self.model_name in 'quant':
                adv_x = _quantize(adv_x)
            correct = self.correct_func(adv_x,self.y)
            r[~correct] = mid[~correct]
            l[correct] = mid[correct]
            adv_x_all[~correct] = adv_x[~correct]
        assert adv_x_all.any(axis=1).all()
        np.save(self.model_dir + self.res_dir,adv_x_all)
        return adv_x_all

    def verify_ineps(self,eps):
        if self.model_name in 'quant':
            self.model =  torch.load('models/' + self.dataset_name + '/plain_dnn/model.pkl')
        pgd = PGD_Linf(self.model,self.x,self.y)
        adv_x,_,_ = pgd.generate({'eps':eps,'eps_iter':0.001, 'nb_iter':300,'loss_fn':self.loss_fn})
        if self.model_name in 'quant':
            adv_x = _quantize(adv_x)
        correct = self.correct_func(adv_x,self.y)
        return np.mean(correct)

class CWinfAdaptor(BasicAdaptor):

    def __init__(self, dataset, model):
        super(CWinfAdaptor, self).__init__(dataset, model)
        self.res_dir = '/CWinf'
        if self.model_name in 'quant':
            self.model_dir = 'models/' + self.dataset_name + '/' + 'plain_dnn'

    def verify(self):
        from attack.CWinf import CWinf
        if self.task == 'FD':
            params = {'steps': 1000, 'lr': 0.001, 'initial_const': 1e-3, 'binary_search_steps': 5, 'num_classes': 16, 'loss_fn':self.loss_fn, 'q_limit': self.q_limit}
        else:
            params = {'steps': 1000, 'lr': 0.001, 'initial_const': 1e-3, 'binary_search_steps': 5, 'num_classes': 16}
        cw = CWinf(self.model, self.x, self.y)
        adv_x, epsilon, rho = cw.generate(params)
        np.save(self.model_dir + self.res_dir,adv_x)
        return adv_x



class MilpAdaptor(BasicAdaptor):
    def __init__(self, dataset, model):
        super(MilpAdaptor, self).__init__(dataset, model)
        self.res_dir = '/Milp'

    def verify(self):
        import attack.MILPverifier as MILPverifier
        params =  { 'm_radius':0.5}
        min_radius_list = []
        adv_x = []
        status_list = []
        quant = False
        if self.model_name in 'quant':
            quant = 8
        if 'FD' in self.task:
            vfr = MILPverifier.AE(self.model,self.x.shape[1],self.q_limit, params['m_radius'],quant)
        else:
            vfr = MILPverifier.DNN(self.model,self.x.shape[1], params['m_radius'],quant)

        exist_l = 0
        res_file_dir = self.model_dir + self.res_dir + '_info.npy'
        if os.path.isfile(res_file_dir):
            info = np.load(res_file_dir, allow_pickle=True).item()
            min_radius_list, adv_x, status_list = info['radius'], info['adv_x'], info['status']
            exist_l = len(adv_x)

        for i,(x,y) in enumerate(zip(self.x,self.y.astype(np.int32))):

            if i >= exist_l:
                res = vfr.verify(x,y)
                min_radius_list.append(res)
                if 'infeasible' not in vfr.prob.status:
                    adv_x.append(vfr.cx.value)
                else:
                    adv_x.append(x)
                status_list.append(vfr.prob.status)

            print(i)
            np.save(self.model_dir + self.res_dir + '_info'
                ,{'radius':min_radius_list,'adv_x':adv_x,'status':status_list})
            np.save(self.model_dir + self.res_dir ,adv_x)

        return adv_x

class DeepFoolAdaptor(BasicAdaptor):

    def __init__(self, dataset, model):
        super(DeepFoolAdaptor, self).__init__(dataset, model)
        self.res_dir = '/DeepFool'
        if self.model_name in 'quant':
            self.model_dir = 'models/' + self.dataset_name + '/' + 'plain_dnn'

    def verify(self):
        from attack.DeepFoolLinf import DeepFoolLinf
        if self.task == 'FD':
            params = {'nb_candidate': 16, 'max_iter':50, 'overshoot':0.01, 'q_limit': self.q_limit}
        else:
            params = {'nb_candidate': 16, 'max_iter': 50, 'overshoot': 0.01}
        df = DeepFoolLinf(self.model, self.x, self.y)
        adv_x, epsilon, rho = df.generate(params)
        np.save(self.model_dir + self.res_dir,adv_x)
        return adv_x


class UAPAdaptor(BasicAdaptor):
    def __init__(self, dataset, model):
        super(UAPAdaptor, self).__init__(dataset, model)
        if self.model_name in 'quant':
            self.model_dir = 'models/' + self.dataset_name + '/' + 'plain_dnn'
        self.res_dir = '/UAP'
         
    def verify_ineps(self,eps,new_cal=False):
        from attack.UAP import UAP

        res_file_dir = self.model_dir + '/UAP' + '_{:.2}.npy'.format(eps)
        if os.path.isfile(res_file_dir) and not new_cal:
            adv_x = np.load(res_file_dir, allow_pickle=True)
            if not torch.is_tensor(adv_x):
                adv_x = torch.FloatTensor(adv_x)
        else:
            if self.task == 'FD':
                uap = UAP(self.model, self.data['x_train'], self.data['y_train'], self.q_limit)
            else:
                uap = UAP(self.model, self.data['x_train'], self.data['y_train'], None)
            adv_v = uap.generate({'eps': eps})
            adv_x = torch.clamp(torch.from_numpy(self.x).float()+ adv_v, 0 ,1 )
            np.save(res_file_dir,adv_x)
        correct = self.correct_func(adv_x,self.y)
        return np.mean(correct)


class SPSAAdaptor(BasicAdaptor):

    def __init__(self, dataset, model):
        super(SPSAAdaptor, self).__init__(dataset, model)
        self.res_dir = '/SPSA'

    def verify(self):
        if self.task == 'FD':
            raise ValueError('The SPSA is not adapted for dection')
        else:
            params = {'eps': 0.03, 'delta': 0.01, 'nb_iter': 50}
            l = 0.0 * np.ones_like(self.y).reshape(-1, 1).astype(np.float32)
            r = 0.51 * np.ones_like(self.y).reshape(-1, 1).astype(np.float32)
            error = 1e-3
            spsa = SPSA(self.model, self.x, self.y)
            adv_x_all = np.zeros_like(self.x)
            while max(r - l) > error:
                mid = (l + r) / 2.0
                params['eps'] = mid
                adv_x = spsa.generate(params)
                prediction = self.model(adv_x).max(1)[1]
                correct = torch.eq(prediction, torch.LongTensor(self.y)).numpy()
                r[~correct] = mid[~correct]
                l[correct] = mid[correct]
                adv_x_all[~correct] = adv_x[~correct]
    
            assert adv_x_all.any(axis=1).all()
            np.save(self.model_dir + self.res_dir, adv_x_all)
            return adv_x_all


    def verify_ineps(self, eps):
        spsa = SPSA(self.model, self.x, self.y)
        adv_x = spsa.generate({'eps': eps, 'delta': 0.01, 'nb_iter': 50})
        prediction = self.model(adv_x).max(1)[1]
        correct = torch.eq(prediction, torch.LongTensor(self.y)).numpy()
        return np.mean(correct)



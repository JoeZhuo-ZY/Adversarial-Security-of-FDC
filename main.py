import numpy as np
import torch
import attack.attack_adaptor as attack
import defense.defense_adaptor as defense
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import confusion_matrix


dataset_name = 'TEP' #  fault calssification
# dataset_name = 'TEP_FD' #  fault detection

defenders = {'plain_dnn':defense.PlainDNNAdaptor,'FGSM_train':defense.FGSMTrainAdaptor,
             'PGD_train':defense.PGDTrainAdaptor, 'distillation': defense.DistillationAdaptor, 
             'quant':defense.QuantAdaptor,'reg_dnn':defense.RegDNNAdaptor, 
             'IBP_train':defense.IBPDNNAdaptor,'CAE':defense.CAEAdaptor}

attackers = {'noise':attack.NoiseAdaptor, 'SPSA': attack.SPSAAdaptor,
             'UAP':attack.UAPAdaptor, 'FGSM': attack.FGSMAdaptor, 
             'PGD':attack.PGDAdaptor, 'CWinf':attack.CWinfAdaptor, 
             'DeepFool':attack.DeepFoolAdaptor, 'Milp': attack.MilpAdaptor}

use_cache_advs = True  # use the trained defensive models and calculated adversarial samples
eps = 0.03 # l_\infty epsilon bounds for adversarial samples


for d_name in defenders:
    print()
    # defense
    if dataset_name == 'TEP' and d_name == 'CAE':
        continue # CAE is not for classification
    if dataset_name == 'TEP_FD' and d_name in ['distillation','IBP_train']: 
        continue # 'distillation','IBP_train' is not for detection
    
    if not use_cache_advs:
        print('**** train defensive models %s *****'%d_name)
        dfr = defenders[d_name](dataset_name)
        dfr.train()   
    else:
       print('**** use trained defensive models %s *****'%d_name)
       
    # attack
    # clean acc without attacks
    names = {'dataset' : dataset_name, 'model' : d_name}
    clean = attack.CleanAdaptor(**names)
    cln_v = clean.verify()
    print('Clean acc {:.2%}'.format(cln_v))
    
    # iterate all attack methods
    for a_name in attackers:
        if dataset_name == 'TEP_FD' and a_name == 'SPSA':
            continue # SPSA is not for detection
            
        print('**** attack %s model by %s*****'%(d_name,a_name))
        atk = attackers[a_name](**names)
        acc, radius, eps_inf, epsilon_crr, _ = atk.get_adv_info(eps, use_cache_advs)
        print('acc:{:.2%} radius:{:.3}'.format(acc,radius)) # acc for AAcc, radius for MAP in paper


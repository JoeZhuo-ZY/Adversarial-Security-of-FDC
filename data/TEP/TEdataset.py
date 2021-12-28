import numpy as np
import torch
from sklearn import preprocessing
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import  Counter
import numpy.ma as ma
from sklearn import svm
# rootdir = "E:/OneDrive - zju.edu.cn/TE/"
rootdir = "C:/Users/win10/OneDrive - zju.edu.cn/TE/"

def get_data(seed,ind):
    data,labs=get_xy(ind)
    # data = preprocessing.StandardScaler().fit_transform(data)
    data = preprocessing.MinMaxScaler().fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(data, labs, test_size=0.1, random_state=seed,stratify=labs)
    # x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=seed,stratify=y_test)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=seed,stratify=y_test)
    nparray=dict()
    nparray['x_train']=x_train
    nparray['x_test']=x_test
    nparray['x_valid']=x_valid
    nparray['y_train']=y_train
    nparray['y_test']=y_test
    nparray['y_valid']=y_valid
    return nparray

def get_data_fd(seed,ind):
    data,labs=get_xy_fd(ind)
    data = preprocessing.StandardScaler().fit_transform(data)
    data = preprocessing.MinMaxScaler().fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(data, labs, test_size=0.1, random_state=seed,stratify=labs)
    # x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=seed,stratify=y_test)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=seed,stratify=y_test)
    y_test[y_test!=0]=1
    y_train[y_train!=0]=1
    y_valid[y_valid!=0]=1
    nparray=dict()
    nparray['x_train']=x_train[y_train==0,:]
    nparray['x_test']=x_test
    nparray['x_valid']=x_valid
    nparray['y_train']=y_train[y_train==0]
    nparray['y_test']=y_test
    nparray['y_valid']=y_valid
    return nparray

def get_fault(ind, num = 5):
    data=sio.loadmat(rootdir+'TE_wch_50_IDV_{:d}.mat'.format(ind))
    fl=data['TE_wch_IDV_{:d}'.format(ind)]
    fl=fl[::num,:]
    fl=np.delete(fl,[0,46,50,53],axis=1)
    return fl

def get_xy(ind):
    X=np.empty(shape=(0, 50))
    y=[]
    for n,i in enumerate(ind):
        fl=get_fault(i)
        X=np.append(X,fl,axis=0)
        y=np.append(y,(n)*np.ones(fl.shape[0]))
    return X,y

def get_xy_fd(ind):
    X=np.empty(shape=(0, 50))
    y=[]
    for n,i in enumerate(ind):
        if i == 0:
            fl=get_fault(i,2)
        else:
            fl=get_fault(i,30)
        X=np.append(X,fl,axis=0)
        y=np.append(y,(n)*np.ones(fl.shape[0]))
    return X,y

data = get_data_fd(seed=3691,ind=range(16))
# clf = svm.LinearSVC()
# clf.fit(data['x_train'],data['y_train'])
# clf.score(data['x_test'],data['y_test'])
# clf.score(data['x_valid'],data['y_valid'])
# np.save('TEdata_fd',data)
# fc = np.load('data.npy',allow_pickle=True).item()

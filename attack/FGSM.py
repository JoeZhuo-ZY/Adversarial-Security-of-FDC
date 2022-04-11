import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class FGSM():
    def __init__(self,model,x,y):
        super().__init__()
        self.model=model
        # self.device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x).float()
            self.y = torch.LongTensor(y)
        else:
            self.x = x
            self.y = y

    def generate(self,params):
        self.parse_params(**params)
        labels = torch.LongTensor(self.y)
        x_new = self.x
        if self.rand_init:
            x_new = self.x + torch.Tensor(np.random.uniform(-self.eps, self.eps, self.x.shape)).type_as(self.x)

        # get the gradient of x
        x_new=Variable(x_new,requires_grad=True)
        loss_fn = self.loss_fn

        if self.m_type=='dnn':
            preds = self.model(x_new)
            if 'FD' in str(loss_fn):
                loss = loss_fn(x_new, preds, labels)
            else:
                loss = loss_fn(preds, labels)  # 无目标攻击，labels为正确值
            self.model.zero_grad()
            
        loss.backward()
        grad = x_new.grad.detach()
        # get the pertubation of an iter_eps
        if self.ord==np.inf:
            grad = torch.sign(grad)
        else:
            tmp = grad.reshape(grad.shape[0], -1)
            norm = 1e-12 + np.linalg.norm(tmp, ord=self.ord, axis=1, keepdims=False).reshape(-1, 1, 1, 1)
            # 选择更小的扰动
            grad=grad/norm
        pertubation = grad*self.eps

        adv_x = self.x.detach() + pertubation
        adv_x=torch.clamp(adv_x,self.clip_min,self.clip_max)

        epsilon = (adv_x - self.x).numpy()
        epsilon = np.linalg.norm(epsilon, ord=np.Inf, axis=1)
        rho = epsilon / np.linalg.norm(self.x.numpy(), ord=np.Inf, axis=1)

        return adv_x, epsilon, rho

    def parse_params(self,eps=0.3,ord=np.inf,clip_min=0.0,clip_max=1.0,
                     rand_init=True,flag_target=False,m_type='dnn',loss_fn=nn.CrossEntropyLoss()):
        self.eps=eps
        self.ord=ord
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.rand_init=rand_init
        # self.model.to(self.device)
        self.flag_target=flag_target
        self.m_type=m_type
        self.loss_fn = loss_fn



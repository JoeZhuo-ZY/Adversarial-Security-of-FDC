import torch
import torch.nn as nn
import numpy as np

def perturb_iterative(xvar, yvar, model, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0, m_type='dnn'):

    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        if m_type == 'dnn':
            outputs = model(xvar + delta)
        if m_type == 'svm':
            weight = torch.tensor(model.coef_).float()
            bias = torch.tensor(model.intercept_)
            outputs = (xvar + delta) @ weight.T + bias
        
        if 'FD' in str(loss_fn):
            loss = loss_fn(xvar + delta, outputs, yvar)
        else:
            loss = loss_fn(outputs, yvar)  # 无目标攻击，labels为正确值
                
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + eps_iter * grad_sign
            delta.data = np.clip(delta.data, -eps, eps)
            delta.data = torch.clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data

        delta.grad.data.zero_()

    x_adv = torch.clamp(xvar + delta, clip_min, clip_max)
    return x_adv


class PGD_Linf():
    def __init__(self,model,x,y):
        super().__init__()
        self.model = model
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x).float()
            self.y = torch.LongTensor(y)
        else:
            self.x = x
            self.y = y


    def parse_params(self,loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, m_type='dnn'):
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.ord = ord
        self.targeted = targeted
        self.m_type = m_type
        self.loss_fn =loss_fn
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def generate(self,params):
        self.parse_params(**params)
        x = self.x.detach().clone()
        y = (self.y).detach().clone()

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            delta.data.uniform_(-1, 1)
            delta.data= delta.data * self.eps
            delta.data = torch.clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.model, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            m_type=self.m_type
        )

        epsilon = (rval.detach() - x).numpy()
        epsilon = np.linalg.norm(epsilon, ord=np.Inf, axis=1)
        rho = epsilon / np.linalg.norm(x.numpy(), ord=np.Inf, axis=1)

        return rval.data, epsilon, rho


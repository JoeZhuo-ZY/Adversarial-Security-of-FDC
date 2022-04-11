# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:14:31 2021

@author: win10
"""
import torch
from torch import nn
import numpy as np
import cvxpy as cp
import numpy as np
import torch
from torch import nn
import dccp

class BoundCalculator:

    def __init__(self, model, in_shape, in_min, in_max):
        super(BoundCalculator, self).__init__()

        in_numel = None
        shapes = list()

        Ws = list()
        bs = list()
        in_numel = in_shape
        shapes.append(in_numel)
        for m in model.children():
            for name,param in model.named_parameters():
                if 'weight' in name:
                    Ws.append(param.data.numpy())
                if 'bias' in name:
                    bs.append(param.data.numpy())
                    shapes.append(param.numel())

        self.in_min, self.in_max = in_min, in_max

        self.in_numel = in_numel
        self.shapes = shapes
        self.Ws = Ws
        self.bs = bs

        self.l = None
        self.u = None

    def verify(self, y_true, y_adv):
        """
            Assert if y_true >= y_adv holds for all
        :param y_true:
        :param y_adv:
        :return: True: y_true >= y_adv always holds, False: y_true >= y_adv MAY not hold
        """
        assert self.l is not None and self.u is not None
        assert len(self.l) == len(self.Ws)
        assert len(self.u) == len(self.bs)
        assert len(self.l) == len(self.u)
        assert len(self.Ws) == len(self.bs)

        l = self.l[-1]
        u = self.u[-1]
        l = np.maximum(l, 0)
        u = np.maximum(u, 0)
        W = self.Ws[-1]
        b = self.bs[-1]
        W_delta = W[y_true] - W[y_adv]
        b_delta = b[y_true] - b[y_adv]
        lb = np.dot(np.clip(W_delta, a_min=0., a_max=None), l) + np.dot(np.clip(W_delta, a_min=None, a_max=0.), u) + b_delta
        # print(l)
        # print(u)
        # print(u-l)
        # print(y_true, y_adv, lb)
        return lb >= 0.

    def calculate_bound(self, x0, eps):
        raise NotImplementedError("Haven't implemented yet.")


class IntervalBound(BoundCalculator):

    def calculate_bound(self, x0, eps):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.l = [np.clip(x0 - eps, a_min=self.in_min, a_max=self.in_max)]
        self.u = [np.clip(x0 + eps, a_min=self.in_min, a_max=self.in_max)]

        for i in range(len(self.Ws) - 1):
            now_l = self.l[-1]
            now_u = self.u[-1]
            if i > 0:
                now_l = np.clip(now_l, a_min=0., a_max=None)
                now_u = np.clip(now_u, a_min=0., a_max=None)
            W, b = self.Ws[i], self.bs[i]
            new_l = np.matmul(np.clip(W, a_min=0., a_max=None), now_l) + np.matmul(np.clip(W, a_min=None, a_max=0.), now_u) + b
            new_u = np.matmul(np.clip(W, a_min=None, a_max=0.), now_l) + np.matmul(np.clip(W, a_min=0., a_max=None), now_u) + b
            self.l.append(new_l)
            self.u.append(new_u)


class FastLinBound(BoundCalculator):

    def _form_diag(self, l, u):
        d = np.zeros(l.shape[0])
        for i in range(d.shape[0]):
            if u[i] >= 1e-6 and l[i] <= -1e-6:
                d[i] = u[i] / (u[i] - l[i])
            elif u[i] <= -1e-6:
                d[i] = 0.
            else:
                d[i] = 1.
        return np.diag(d)

    def calculate_bound(self, x0, eps):
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.l = [np.clip(x0 - eps, a_min=self.in_min, a_max=self.in_max)]
        self.u = [np.clip(x0 + eps, a_min=self.in_min, a_max=self.in_max)]

        A0 = self.Ws[0]
        A = list()

        for i in range(len(self.Ws) - 1):
            T = [None for _ in range(i)]
            H = [None for _ in range(i)]
            for k in range(i - 1, -1, -1):
                if k == i - 1:
                    D = self._form_diag(self.l[-1], self.u[-1])
                    A.append(np.matmul(self.Ws[i], D))
                else:
                    A[k] = np.matmul(A[-1], A[k])
                T[k] = np.zeros_like(A[k].T)
                H[k] = np.zeros_like(A[k].T)
                for r in range(self.l[k+1].shape[0]):
                    if self.u[k+1][r] >= 1e-6 and self.l[k+1][r] <= -1e-6:
                        for j in range(A[k].shape[0]):
                            if A[k][j, r] > 0.:
                                T[k][r, j] = self.l[k+1][r]
                            else:
                                H[k][r, j] = self.l[k+1][r]
            if i > 0:
                A0 = np.matmul(A[-1], A0)
            nowl = list()
            nowu = list()
            for j in range(self.Ws[i].shape[0]):
                nu_j = np.dot(A0[j], x0) + self.bs[i][j]
                mu_p_j = mu_n_j = 0.
                for k in range(0, i):
                    mu_p_j -= np.dot(A[k][j], (T[k].T)[j])
                    mu_n_j -= np.dot(A[k][j], (H[k].T)[j])
                    nu_j += np.dot(A[k][j], self.bs[k])
                nowl.append(mu_n_j + nu_j - eps * np.sum(np.abs(A0[j])))
                nowu.append(mu_p_j + nu_j + eps * np.sum(np.abs(A0[j])))
            self.l.append(np.array(nowl))
            self.u.append(np.array(nowu))


class IntervalFastLinBound(BoundCalculator):

    def __init__(self, model, in_shape, in_min, in_max):
        super(IntervalFastLinBound, self).__init__(model, in_shape, in_min, in_max)

        self.interval_calc = IntervalBound(model, in_shape, in_min, in_max)
        self.fastlin_calc = FastLinBound(model, in_shape, in_min, in_max)

    def calculate_bound(self, x0, eps):
        self.interval_calc.calculate_bound(x0, eps)
        self.fastlin_calc.calculate_bound(x0, eps)

        self.l = list()
        self.u = list()
        for i in range(len(self.interval_calc.l)):
            self.l.append(np.maximum(self.interval_calc.l[i], self.fastlin_calc.l[i]))
            self.u.append(np.minimum(self.interval_calc.u[i], self.fastlin_calc.u[i]))

class SVM:
    def __init__(self, model, data, data_percent, seed, in_min=0, in_max=1):

        Ws = list()
        bs = list()
        Ws.append(model.coef_)
        bs.append(model.intercept_)
        self.seed = seed
        self.in_min, self.in_max = in_min, in_max
        self.Ws = Ws
        self.bs = bs
        # x_sub, y_sub = select_data(data,data_percent,seed)
        # prediction = model.predict(x_sub)
        # correct = (prediction == y_sub)
        # self.clean_data = zip(x_sub[correct], y_sub[correct].astype(np.int32))
        self.clean_data = self.select_data_in_correct(model, data, data_percent, seed)

    def construct(self, x0, y0, eps=1):

        self.constraints = list()
        self.cx = cp.Variable(x0.shape[0])

        x_min = np.maximum(x0 - eps, self.in_min)
        x_max = np.minimum(x0 + eps, self.in_max)

        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))

        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        # last_w @ x0 +last_b
        mask = np.ones(last_w.shape[0], dtype=bool)
        mask[y0] = False
        last_w_masked = last_w[mask,:]
        last_b_masked = last_b[mask]

        # Big-M converts max function
        output = last_w_masked @ self.cx + last_b_masked
        self.z = cp.Variable(last_b_masked.shape[0],boolean=True)
        self.max = cp.Variable()
        self.constraints.append((cp.sum(self.z) == 1))
        self.constraints.append((self.max >= output))
        self.constraints.append((self.max <= output + cp.multiply((1 - self.z),10)))

        self.constraints.extend([self.max  >= last_w[y0] @ self.cx + last_b[y0] + 1e-5])
        self.prob = cp.Problem(self.obj, self.constraints)

    def verify(self):
        min_radius_list = []
        for i,(x,y) in enumerate(self.clean_data):
            self.construct(x,y)
            self.prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=60, Threads=12)
            if self.prob.status not in ['optimal','Converged']:
                print(self.prob.status)
                res = 0.3
            else:
                res = self.prob.value
            min_radius_list.append(res)
        return np.mean(min_radius_list), min_radius_list

    def select_data_in_correct(self, model, data,data_percent,seed):
        # prediction = model.predict(data['x_test'])
        # y_correct = data['y_test'][prediction == data['y_test']]
        # x_correct = data['x_test'][prediction == data['y_test'],:]
        # # random
        # random_state = np.random.RandomState(seed)
        # partial_indx = random_state.choice(np.arange(len(y_correct)),
        #                                    size=int(np.ceil(len(y_correct)*data_percent)), replace=False)
        # return zip(x_correct[partial_indx,:], y_correct[partial_indx].astype(np.int32))

        # random as class
        prediction = model.predict(data['x_test'])
        y_correct = data['y_test'][prediction == data['y_test']].astype(np.int32)
        x_correct = data['x_test'][prediction == data['y_test'],:]
        random_state = np.random.RandomState(seed)
        labels = np.unique(y_correct)
        label_nums = np.array([np.sum(y_correct==l) for l in labels])

        lnargs = np.argsort(label_nums)[::-1]
        labels, label_nums = labels[lnargs],label_nums[lnargs]
        data_num = int(np.ceil(len(y_correct)*data_percent))
        nums_select = 0
        yt = np.copy(y_correct)
        idx_list = []
        if data_num >= len(y_correct):
            return zip(x_correct, y_correct)
        else:
            while nums_select < data_num:
                for n,l in enumerate(labels):
                    if label_nums[n] > 0:
                    # while label_nums[n] > 0:
                        nums_select += 1
                        label_nums[n] -= 1
                        label_idx = np.argwhere(yt==l).flatten()
                        selected_idx = label_idx[random_state.choice(len(label_idx))]
                        yt[selected_idx] = -1
                        idx_list.append(selected_idx)
                        if nums_select == data_num:
                            idx_list = np.array(idx_list)
                            return zip(x_correct[idx_list], y_correct[idx_list])


        #stratify
        # prediction = model.predict(data['x_test'])
        # y_correct = data['y_test'][prediction == data['y_test']].astype(np.int32)
        # x_correct = data['x_test'][prediction == data['y_test'],:]
        # label_num = len(np.unique(y_correct))
        # min_resourse = label_num/len(y_correct)
        # real_resourse = np.max([data_percent, min_resourse])
        # if real_resourse == 1:
        #     x_sub, y_sub = x_correct, y_correct
        # else:
        #     x_sub, _, y_sub,_ = train_test_split(x_correct, y_correct, test_size=1-real_resourse,
        #                                       random_state=seed, stratify=y_correct)
        # return zip(x_sub, y_sub)
class DNN:
    def __init__(self, model, in_numel, m_radius= 1,quant=False ):
        super(DNN, self).__init__()

        model = next(model.children())
        num_layers = len([None for _ in model])
        shapes = list()

        Ws = list()
        bs = list()
        shapes.append(in_numel)
        for name,param in model.named_parameters():
            if 'weight' in name:
                Ws.append(param.data.numpy())
            if 'bias' in name:
                bs.append(param.data.numpy())
                shapes.append(param.numel())

        self.in_numel = in_numel
        self.num_layers = num_layers
        self.shapes = shapes
        self.Ws = Ws
        self.bs = bs
        self.m_radius = m_radius
        self.prebound = IntervalFastLinBound(model, in_numel, 0, 1)
        self.quant = quant

    def construct(self, l, u, x0, y0, eps=1, in_min=0, in_max=1):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()

        if self.quant:
            N = int(2**self.quant)
            self.cxN = cp.Variable(self.in_numel, integer=True)
            self.cx = self.cxN / N
        else:
            self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, in_min)
        x_max = np.minimum(x0 + eps, in_max)

        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))
        pre = self.cx


        for i in range(len(self.Ws) - 1):
            now_x = (self.Ws[i] @ pre) + self.bs[i]
            now_shape = self.shapes[i + 1]
            now_y = cp.Variable(now_shape)
            now_a = cp.Variable(now_shape, boolean=True)
            for j in range(now_shape):
                if l[i + 1][j] >= 0:
                    self.constraints.extend([now_y[j] == now_x[j]])
                elif u[i + 1][j] <= 0:
                    self.constraints.extend([now_y[j] == 0.])
                else:
                    self.constraints.extend([
                        (now_y[j] <= now_x[j] - (1 - now_a[j]) * l[i + 1][j]),
                        (now_y[j] >= now_x[j]),
                        (now_y[j] <= now_a[j] * u[i + 1][j]),
                        (now_y[j] >= 0.)
                    ])
            # self.constraints.extend([(now_y <= now_x - cp.multiply((1 - now_a), l[i + 1])), (now_y >= now_x), (now_y <= cp.multiply(now_a, u[i + 1])), (now_y >= 0)])
            pre = now_y
        self.last_x = pre

        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        # last_w @ x0 +last_b
        mask = np.ones(last_w.shape[0], dtype=bool)
        mask[y0] = False
        last_w_masked = last_w[mask,:]
        last_b_masked = last_b[mask]

        # Big-M converts max function
        output = last_w_masked @ self.last_x + last_b_masked
        self.zm = cp.Variable(last_b_masked.shape[0],boolean=True)
        self.max = cp.Variable()
        self.constraints.append((cp.sum(self.zm) == 1))
        self.constraints.append((self.max >= output))
        self.constraints.append((self.max <= output + cp.multiply((1 - self.zm),10000)))

        self.constraints.extend([self.max  >= last_w[y0] @ self.last_x + last_b[y0] + 1e-6])
        self.prob = cp.Problem(self.obj, self.constraints)

    def verify(self,x,y):
        res = None

        self.prebound.calculate_bound(x, self.m_radius)
        self.construct(self.prebound.l, self.prebound.u, x, y, self.m_radius)
        self.prob.solve(solver=cp.GUROBI, verbose=True, TimeLimit=100, Threads=12)
        if self.prob.status not in ['optimal','Converged','user_limit']:
            print(self.prob.status)
        else:
            res = self.prob.value
        print('radius:%f'%(res))
        print(self.prob.status)

        return res

class AE:
    def __init__(self, model, in_numel, q_lim, m_radius= 1, quant=False ):
        super(AE, self).__init__()

        shapes = list()
        Ws = list()
        bs = list()
        shapes.append(in_numel)
        for m in model.children():
            for name,param in m.named_parameters():
                if 'weight' in name:
                    Ws.append(param.data.numpy())
                if 'bias' in name:
                    bs.append(param.data.numpy())
                    shapes.append(param.numel())

        self.in_numel = in_numel
        self.shapes = shapes
        self.Ws = Ws
        self.bs = bs
        self.m_radius = m_radius
        self.prebound = IntervalFastLinBound(model, in_numel, 0, 1)
        self.q_lim = q_lim
        self.quant = quant

    def construct(self, l, u, x0, y0, eps=1, in_min=0, in_max=1):

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(eps, torch.Tensor):
            eps = eps.cpu().numpy()

        self.constraints = list()

        if self.quant:
            N = int(2**self.quant)
            self.cxN = cp.Variable(self.in_numel, integer=True)
            self.cx = self.cxN / N
        else:
            self.cx = cp.Variable(self.in_numel)

        x_min = np.maximum(x0 - eps, in_min)
        x_max = np.minimum(x0 + eps, in_max)

        self.constraints.append((self.cx >= x_min))
        self.constraints.append((self.cx <= x_max))

        self.obj = cp.Minimize(cp.norm_inf(self.cx - x0 ))
        pre = self.cx


        for i in range(len(self.Ws) - 1):
            now_x = (self.Ws[i] @ pre) + self.bs[i]
            now_shape = self.shapes[i + 1]
            now_y = cp.Variable(now_shape)
            now_a = cp.Variable(now_shape, boolean=True)
            for j in range(now_shape):
                if l[i + 1][j] >= 0:
                    self.constraints.extend([now_y[j] == now_x[j]])
                elif u[i + 1][j] <= 0:
                    self.constraints.extend([now_y[j] == 0.])
                else:
                    self.constraints.extend([
                        (now_y[j] <= now_x[j] - (1 - now_a[j]) * l[i + 1][j]),
                        (now_y[j] >= now_x[j]),
                        (now_y[j] <= now_a[j] * u[i + 1][j]),
                        (now_y[j] >= 0.)
                    ])
            # self.constraints.extend([(now_y <= now_x - cp.multiply((1 - now_a), l[i + 1])), (now_y >= now_x), (now_y <= cp.multiply(now_a, u[i + 1])), (now_y >= 0)])
            pre = now_y
        self.last_x = pre

        last_w = self.Ws[-1]
        last_b = self.bs[-1]
        self.x_hat = (last_w @ self.last_x) + last_b


    def verify(self,x,y):
        res = None

        for _ in range(10):
            try:
                self.prebound.calculate_bound(x, self.m_radius)
                self.construct(self.prebound.l, self.prebound.u, x, y, self.m_radius)

                if y == 1:
                    self.constraints.extend([cp.sum_squares(self.x_hat - self.cx) <= self.q_lim])
                    self.prob = cp.Problem(self.obj, self.constraints)
                    self.prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=100, Threads=12)
                    res = self.prob.value
                if y == 0:
                    self.constraints.extend([cp.sum_squares(self.x_hat - self.cx) >= self.q_lim])
                    self.prob = cp.Problem(self.obj, self.constraints)
                    result = self.prob.solve(method='dccp',solver=cp.GUROBI, verbose=False, TimeLimit=100, Threads=12)
                    res = result[0]

                if self.prob.status not in ['optimal','Converged','user_limit']:
                    print(self.prob.status)

                print('radius:%f'%(res))
                print(self.prob.status)

                return res
            except Exception as e:
                ...
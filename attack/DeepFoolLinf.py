import torch
import torch.nn as nn
# from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
import numpy as np


def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives


class DeepFoolLinf(nn.Module):
    def __init__(self,model,x,y):
        super().__init__()
        self.model=model
        self.x = torch.from_numpy(x).float()
        self.y = torch.LongTensor(y)

    def parse_params(self, nb_candidate=10, overshoot=0.02, max_iter=50, clip_min=0.0, clip_max=1.0, targeted=False, m_type='dnn', q_limit=None):
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        # self.model.to(self.device)
        self.targeted = targeted
        self.m_type = m_type
        self.q_limit = q_limit

    def generate(self, params):
        self.parse_params(**params)
        x = self.x
        y = self.y
        with torch.no_grad():
            if self.m_type == 'dnn':
                logits = self.model(x)
            if self.m_type == 'svm':
                weight = torch.tensor(self.model.coef_).float()
                bias = torch.tensor(self.model.intercept_)
                logits = x @ weight.T + bias

        self.nb_classes = logits.size(-1)
        assert self.nb_candidate <= self.nb_classes, 'nb_candidate should not be greater than nb_classes'

        # preds = logits.topk(self.nb_candidate)[0]
        # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
        # grads will be the shape [batch_size, nb_candidate, image_size]

        adv_x = x.clone().requires_grad_()

        iteration = 0
        if self.m_type == 'dnn':
            logits = self.model(adv_x)
        if self.m_type == 'svm':
            weight = torch.tensor(self.model.coef_).float()
            bias = torch.tensor(self.model.intercept_)
            logits = adv_x @ weight.T + bias


        w = torch.squeeze(torch.zeros(x.size()[1:]))
        r_tot = torch.zeros(x.size())

        if self.q_limit != None:
            q_test = self.model.cal_q(adv_x)
            current = ((q_test > self.q_limit) & (y == 1)) | ((q_test < self.q_limit) & (y == 0))

            while (current.any() and iteration < self.max_iter):
                predictions_val = q_test - self.q_limit
                # print('predictions_val:', predictions_val)
                gradients, = torch.autograd.grad(predictions_val, adv_x, grad_outputs=torch.ones_like(predictions_val), retain_graph=True)
                with torch.no_grad():
                    for idx in range(x.size(0)):
                        pert = float('inf')
                        if current[idx] == False:
                            continue
                        w = gradients[idx, ...]
                        f = predictions_val[idx]
                        pert = (f + 0.00001) / w.view(-1).norm(p=1)

                        r_i = - pert * w.sign()
                        r_tot[idx, ...] = r_tot[idx, ...] + r_i

                adv_x = torch.clamp(r_tot + x, self.clip_min, self.clip_max).requires_grad_()
                logits = self.model(adv_x)
                q_test = self.model.cal_q(adv_x)
                current = ((q_test > self.q_limit) & (y == 1)) | ((q_test < self.q_limit) & (y == 0))
                iteration = iteration + 1

        else:
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            original = self.y

            while ((current == original).any() and iteration < self.max_iter):
                predictions_val = logits.topk(self.nb_candidate)[0]
                print('predictions_val:', predictions_val)
                gradients = torch.stack(jacobian(predictions_val, adv_x, self.nb_candidate), dim=1)
                with torch.no_grad():
                    for idx in range(x.size(0)):
                        pert = float('inf')
                        if current[idx] != original[idx]:
                            continue
                        for k in range(1, self.nb_candidate):
                            w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                            f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                            pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm(p=1)
                            if pert_k < pert:
                                pert = pert_k
                                w = w_k

                        r_i = pert * w.sign()
                        r_tot[idx, ...] = r_tot[idx, ...] + r_i

                adv_x = torch.clamp(r_tot + x, self.clip_min, self.clip_max).requires_grad_()
                if self.m_type == 'dnn':
                    logits = self.model(adv_x)
                if self.m_type == 'svm':
                    weight = torch.tensor(self.model.coef_).float()
                    bias = torch.tensor(self.model.intercept_)
                    logits = adv_x @ weight.T + bias
                current = logits.argmax(dim=1)
                if current.size() == ():
                    current = torch.tensor([current])
                iteration = iteration + 1


        adv_x = torch.clamp((1 + self.overshoot) * r_tot + x, self.clip_min, self.clip_max)

        epsilon = (adv_x - x).numpy()
        epsilon = np.linalg.norm(epsilon, ord=np.Inf, axis=1)
        rho = epsilon / np.linalg.norm(x.numpy(), ord=np.Inf, axis=1)

        return adv_x, epsilon, rho


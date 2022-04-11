import torch
import torch.nn as nn
import numpy as np
import random

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives

def deepfool(x, model, y, num_classes = 16, overshoot = 0.02, max_iter = 50, nb_candidate = 10, q_limit=None):
    clip_min = 0.0
    clip_max = 1.0
    with torch.no_grad():
        logits = model(x)

    assert nb_candidate <= num_classes, 'nb_candidate should not be greater than nb_classes'

    # preds = logits.topk(self.nb_candidate)[0]
    # grads = torch.stack(jacobian(preds, x, self.nb_candidate), dim=1)
    # grads will be the shape [batch_size, nb_candidate, image_size]

    adv_x = x.clone().requires_grad_()

    iteration = 0
    logits = model(adv_x)

    w = torch.squeeze(torch.zeros(x.size()[1:]))
    r_tot = torch.zeros(x.size())

    if q_limit != None:
        q_test = model.cal_q(adv_x)
        current = ((q_test > q_limit) & (y == 1)) | ((q_test < q_limit) & (y == 0))
        while (current.any() and iteration < max_iter):
            predictions_val = q_test
            # print('predictions_val:', predictions_val)
            gradients, = torch.autograd.grad(predictions_val, adv_x, grad_outputs=torch.ones_like(predictions_val),
                                             retain_graph=True)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] == False:
                        continue
                    w = gradients[idx, ...]
                    f = predictions_val[idx]
                    pert = (f + 0.00001) / w.view(-1).norm(p=1)

                    r_i = pert * w.sign()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, clip_min, clip_max).requires_grad_()

            q_test = model.cal_q(adv_x)
            current = ((q_test > q_limit) & (y == 1)) | ((q_test < q_limit) & (y == 0))
            iteration = iteration + 1
    else:
        current = logits.argmax(dim=1)
        if current.size() == ():
            current = torch.tensor([current])
        original = y

        while ((current == original).any() and iteration < max_iter):
            predictions_val = logits.topk(nb_candidate)[0]
            gradients = torch.stack(jacobian(predictions_val, adv_x, nb_candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = float('inf')
                    if current[idx] != original[idx]:
                        continue
                    for k in range(1, nb_candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm(p=1)
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w.sign()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = torch.clamp(r_tot + x, clip_min, clip_max).requires_grad_()
            logits = model(adv_x)
            current = logits.argmax(dim=1)
            if current.size() == ():
                current = torch.tensor([current])
            iteration = iteration + 1

    r_tot = (1 + overshoot) * r_tot
    return r_tot, iteration


class UAP(nn.Module):
    def __init__(self,model,x_train,y_train,q_limit):
        super().__init__()
        self.model = model
        self.q_limit = q_limit
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.LongTensor(y_train)
        if q_limit != None:
            q_pred = self.model.cal_q(x_train)
            prediction = (q_pred > self.q_limit).long()
        else:
            prediction = self.model(x_train).max(1)[1]
        self.y_train = y_train[prediction == y_train]
        self.x_train = x_train[prediction == y_train, :]


    def parse_params(self, delta=0.20, max_iter_uni=6, eps=0.03, p=np.inf, num_classes=16, overshoot=0.02, max_iter_df=50):
        self.delta = delta
        self.max_iter_uni = max_iter_uni
        self.xi = eps
        self.p = p
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter_df = max_iter_df


    def generate(self, params):
        self.parse_params(**params)


        v = 0
        fooling_rate = 0.0
        best_fooling = 0.0
        best_v = 0
        itr = 0
        index = np.arange(len(self.x_train))
        print('start calculate...')
        while fooling_rate < 1 - self.delta and itr < self.max_iter_uni:
            # Go through the data set and compute the perturbation increments sequentially
            # random shuffle
            random.shuffle(index)
            self.x_train = self.x_train[index,:]
            self.y_train = self.y_train[index]
            # print('x:', self.x_train)
            # print('y:', self.y_train)
            for i in range(0, len(self.x_train)):
                x, y =  self.x_train[i:i + 1], self.y_train[i:i + 1]
                per = (x + torch.tensor(v)).clone().detach().requires_grad_(True)

                if self.q_limit != None:
                    q_test = self.model.cal_q(per)
                    judge = ((q_test > self.q_limit) & (y == 1)) | ((q_test < self.q_limit) & (y == 0))
                else:
                    judge = (int(self.model(x).argmax()) == int(self.model(per).argmax()))

                if judge:
                    # Compute adversarial perturbation
                    self.model.zero_grad()
                    dr, iter = deepfool(per,
                                        self.model,
                                        y,
                                        num_classes=self.num_classes,
                                        overshoot=self.overshoot,
                                        max_iter=self.max_iter_df,
                                        q_limit=self.q_limit)

                    # Make sure it converged...
                    if iter < self.max_iter_df - 1:
                        v = v + dr
                        v = proj_lp(v, self.xi, self.p)

            print('perturbation:', v)


            # Perturb the dataset with computed perturbation
            # dataset_perturbed = dataset + v

            # Compute the fooling rate

            with torch.no_grad():
                if self.q_limit != None:
                    q_test = self.model.cal_q(self.x_train + v)
                    correct = ((q_test > self.q_limit) & (self.y_train == 1)) | ((q_test < self.q_limit) & (self.y_train == 0))
                    fooling_rate = 1 - correct.float().mean().numpy()
                else:
                    prediction = self.model(self.x_train + v).max(1)[1]
                    fooling_rate = 1 - torch.eq(prediction, self.y_train).float().mean().numpy()

                print('FOOLING RATE = ', fooling_rate)
                if fooling_rate > best_fooling:
                    best_fooling = fooling_rate
                    best_v = v
                print('Best Fooling Rate = ', best_fooling)
            itr += 1

        return best_v

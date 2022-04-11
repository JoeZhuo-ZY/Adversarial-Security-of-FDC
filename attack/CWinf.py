import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utils import FDLoss
CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10
TAU = 1.0
DECREASE_FACTOR = 0.9

def to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    """
    y = y.detach().clone().view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot

def is_successful(y1, y2, targeted):
    if targeted is True:
        return y1 == y2
    else:
        return y1 != y2

def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5

def calc_linfdist(x, y):
    d = torch.max(torch.abs(x - y),1)[0]
    return d

def calc_linfdistfn(x, y, tau):
    d = torch.max(torch.abs(x - y) - tau, torch.zeros_like(x))
    return d.view(d.shape[0], -1).sum(dim=1)


def torch_arctanh(x, eps=1e-6):
    return (torch.log((1 + x) / (1 - x))) * 0.5

class CWinf(nn.Module):
    def __init__(self,model,x,y):
        super().__init__()
        self.model = model
        self.x = torch.from_numpy(x).float()
        self.y = torch.LongTensor(y)

    def parse_params(self, steps=100, lr=0.01, binary_search_steps=5,initial_const=1e-3, clip_min=0., clip_max=1.,
                     num_classes=10, targeted=False, m_type='dnn', loss_fn=None, q_limit=None):
        self.steps = steps
        self.binary_search_steps = binary_search_steps
        self.lr = lr
        self.initial_const = initial_const
        self.num_classes = num_classes
        self.clip_min = clip_min
        self.clip_max = clip_max
        # self.model.to(self.device)
        self.targeted = targeted
        self.m_type = m_type
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.tau = TAU
        self.DECREASE_FACTOR = DECREASE_FACTOR
        self.loss_fn = loss_fn
        self.q_limit = q_limit

    def _loss_fn(self, output, y_onehot, linfdist, const):
        real = (y_onehot * output).sum(dim=1)

        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = torch.clamp(other - real, min=0.)
        else:
            loss1 = torch.clamp(real - other, min=0.)
        loss2 = (linfdist).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss

    def loss_fd(self, adv_x, output, y, linfdist, const):
        loss_fn = self.loss_fn
        loss1 = -loss_fn(adv_x, output, y)
        loss2 = (linfdist).sum()
        loss1 = torch.sum(const * loss1)
        loss = loss1 + loss2
        return loss

    def _is_successful(self, output, label, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        #   label

        if is_logits:
            output = output.detach().clone()
            if self.q_limit != None:
                pred = (output > self.q_limit).long()
            else:
                pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()

        return is_successful(pred, label, self.targeted)

    def _forward_and_update_delta(
            self, optimizer, x_atanh, delta, y_onehot, loss_coeffs, y):

        optimizer.zero_grad()
        adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        if self.m_type == 'dnn':
            output = self.model(adv)
        if self.m_type == 'svm':
            weight = torch.tensor(self.model.coef_).float()
            bias = torch.tensor(self.model.intercept_)
            output = adv @ weight.T + bias

        linfdist = calc_linfdist(adv, transimgs_rescale)
        linfdistfn = calc_linfdistfn(adv, transimgs_rescale, self.tau)
        # print(linfdist)
        if self.loss_fn != None:
            loss = self.loss_fd(adv, output, y, linfdistfn, loss_coeffs)
            output = self.model.cal_q(adv)
        else:
            loss = self._loss_fn(output, y_onehot, linfdistfn, loss_coeffs)
        loss.backward()
        optimizer.step()


        return loss.item(), linfdist.data, output.data, adv.data

    def _get_arctanh_x(self, x):
        result = torch.clamp((x - self.clip_min) / (self.clip_max - self.clip_min), min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, linfdist, batch_size,
            cur_linfdists, cur_labels,
            final_linfdists, final_labels, final_advs):

        target_label = labs
        output_logits = output
        if self.q_limit != None:
            output_label = (output > self.q_limit).long()
        else:
            _, output_label = torch.max(output_logits, 1)
        mask = (linfdist < cur_linfdists) & self._is_successful(
            output_logits, target_label, True)

        cur_linfdists[mask] = linfdist[mask]  # redundant
        cur_labels[mask] = output_label[mask]

        mask = (linfdist < final_linfdists) & self._is_successful(
            output_logits, target_label, True)
        final_linfdists[mask] = linfdist[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]

    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound):

        # binary search step
        for ii in range(batch_size):
            cur_labels[ii] = int(cur_labels[ii])
            if self._is_successful(cur_labels[ii], labs[ii], False):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                                              coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                                              coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def generate(self, params):
        self.parse_params(**params)
        y = self.y
        x = self.x.detach().clone()
        x_orignal = x.clone()

        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_linfdists = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()


        final_linfdists = torch.FloatTensor(final_linfdists).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)


        while self.tau > 0.01:
            # Start binary search
            loss_coeffs = torch.ones_like(y).float() * self.initial_const
            for outer_step in range(self.binary_search_steps):
                delta = nn.Parameter(torch.zeros_like(x))
                optimizer = optim.Adam([delta], lr=self.lr)
                cur_linfdists = [CARLINI_L2DIST_UPPER] * batch_size
                cur_labels = [INVALID_LABEL] * batch_size
                cur_linfdists = torch.FloatTensor(cur_linfdists).to(x.device)
                cur_labels = torch.LongTensor(cur_labels).to(x.device)
                prevloss = PREV_LOSS_INIT


                if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                    loss_coeffs = coeff_upper_bound
                for ii in range(self.steps):
                    loss, linfdist, output, adv_img = \
                        self._forward_and_update_delta(
                            optimizer, x_atanh, delta, y_onehot, loss_coeffs, y)
                    if ii % (self.steps // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss

                    self._update_if_smaller_dist_succeed(
                        adv_img, y, output, linfdist, batch_size,
                        cur_linfdists, cur_labels,
                        final_linfdists, final_labels, final_advs)

                self._update_loss_coeffs(
                    y, cur_labels, batch_size,
                    loss_coeffs, coeff_upper_bound, coeff_lower_bound)

            actualtau = torch.max(torch.abs(final_advs - x_orignal))

            if actualtau < self.tau:
                self.tau = actualtau


            self.tau = torch.mul(self.tau, self.DECREASE_FACTOR)

            epsilon = (final_advs - x_orignal).numpy()
            epsilon = np.linalg.norm(epsilon, ord=np.Inf, axis=1)
            adv_acc = np.mean((epsilon>0.03))
            print([self.tau,np.mean(epsilon),adv_acc])

        epsilon = (final_advs - x_orignal).numpy()
        epsilon = np.linalg.norm(epsilon, ord=np.Inf, axis=1)
        rho = epsilon / np.linalg.norm(x_orignal.numpy(), ord=np.Inf, axis=1)

        return final_advs, epsilon, rho



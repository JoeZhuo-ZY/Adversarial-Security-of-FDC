# -*- coding: utf-8 -*-
# @Time : 2021/7/27 13:57
# @Author : Srina
import random

import numpy as np
import torch
from torch import optim

def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta

def _project_perturbation(perturbation, norm, epsilon, input_image, clip_min=-np.inf, clip_max=np.inf):
    """
    Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
    hypercube such that the resulting adversarial example is between clip_min and clip_max,
    if applicable. This is an in-place operation.
    """

    clipped_perturbation = clip_eta(perturbation, norm, epsilon)
    new_image = torch.clamp(input_image + clipped_perturbation, clip_min, clip_max)

    perturbation.add_((new_image - input_image) - perturbation)


def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):
    """
    Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
    given parameters. The gradient is approximated by evaluating `iters` batches
    of `samples` size each.
    """

    assert len(x) == 1
    num_dims = len(x.size())

    x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

    grad_list = []
    for i in range(iters):
        delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
        delta_x = torch.cat([delta_x, -delta_x])
        with torch.no_grad():
            loss_vals = loss_fn(x + delta_x)
        while len(loss_vals.size()) < num_dims:
            loss_vals = loss_vals.unsqueeze(-1)
        avg_grad = (
            torch.mean(loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta
        )

        grad_list.append(avg_grad)


    return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)


def _margin_logit_loss(logits, labels):
    """
    Computes difference between logits for `labels` and next highest logits.
    The loss is high when `label` is unlikely (targeted by default).
    """

    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

    logit_indices = torch.arange(
        logits.size()[1],
        dtype=labels.dtype,
        device=labels.device,
    )[None, :].expand(labels.size()[0], -1)
    incorrect_logits = torch.where(
        logit_indices == labels[:, None],
        torch.full_like(logits, float("-inf")),
        logits,
    )
    max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

    return max_incorrect_logits - correct_logits

class SPSA():
    def __init__(self,model,x,y):
        super().__init__()
        self.model = model
        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x).float()
            self.y = torch.LongTensor(y)
        else:
            self.x = x
            self.y = y

    def parse_params(self, eps=0.03, nb_iter=50, norm=np.inf, clip_min=0., clip_max=1.,
                     targeted=False, early_stop_loss_threshold=None, learning_rate=0.01,
                     delta=0.01, spsa_samples=128, spsa_iters=1, is_debug=False, sanity_checks=True):
        self.eps = eps
        self.nb_iter = nb_iter
        self.norm = norm
        self.clip_min =clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.early_stop_loss_threshold = early_stop_loss_threshold
        self.learning_rate = learning_rate
        self.delta = delta
        self.spsa_samples = spsa_samples
        self.spsa_iters = spsa_iters
        self.is_debug = is_debug
        self.sanity_checks = sanity_checks

    def singlebatch_generate(self, x, y, eps):
        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return x

        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min > self.clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        self.clip_min, self.clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        asserts.append(torch.all(x >= self.clip_min))
        asserts.append(torch.all(x <= self.clip_max))

        if self.is_debug:
            print("Starting SPSA attack with eps = {}".format(eps))

        perturbation = (torch.rand_like(x) * 2 - 1) * eps
        _project_perturbation(perturbation, self.norm, eps, x, self.clip_min, self.clip_max)
        optimizer = optim.Adam([perturbation], lr=self.learning_rate)

        for i in range(self.nb_iter):

            def loss_fn(pert):
                """
                Margin logit loss, with correct sign for targeted vs untargeted loss.
                """
                logits = self.model(x + pert)
                loss_multiplier = 1 if self.targeted else -1
                return loss_multiplier * _margin_logit_loss(logits, y.expand(len(pert)))

            spsa_grad = _compute_spsa_gradient(
                loss_fn, x, delta=self.delta, samples=self.spsa_samples, iters=self.spsa_iters
            )
            perturbation.grad = spsa_grad
            optimizer.step()

            _project_perturbation(perturbation, self.norm, eps, x, self.clip_min, self.clip_max)

            loss = loss_fn(perturbation).item()
            if self.is_debug:
                print("Iteration {}: loss = {}".format(i, loss))
            if self.early_stop_loss_threshold is not None and loss < self.early_stop_loss_threshold:
                break

        adv_x = torch.clamp((x + perturbation).detach(), self.clip_min, self.clip_max)

        if self.norm == np.inf:
            asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))
        else:
            asserts.append(
                torch.all(
                    torch.abs(
                        torch.renorm(adv_x - x, p=self.norm, dim=0, maxnorm=eps) - (adv_x - x)
                    )
                    < 1e-6
                )
            )
        asserts.append(torch.all(adv_x >= self.clip_min))
        asserts.append(torch.all(adv_x <= self.clip_max))

        if self.sanity_checks:
            assert np.all(asserts)

        return adv_x

    def generate(self, params):
        self.parse_params(**params)
        x = self.x
        y = self.y
        eps = self.eps
        # The rest of the function doesn't support batches of size greater than 1,
        # so if the batch is bigger we split it up.
        if len(x) != 1:
            adv_x = []
            if isinstance(eps, float):
                for x_single, y_single in zip(x, y):
                    adv_x_single = self.singlebatch_generate(x_single.unsqueeze(0), y_single.unsqueeze(0), eps)
                    adv_x.append(adv_x_single)
            else:
                for x_single, y_single, eps_single in zip(x, y, eps):
                    eps_single = float(eps_single)
                    adv_x_single = self.singlebatch_generate(x_single.unsqueeze(0), y_single.unsqueeze(0), eps_single)
                    adv_x.append(adv_x_single)
            return torch.cat(adv_x)


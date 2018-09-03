
import numpy as np
import random, sys, os, json, glob, math

import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Optimizer(nn.Module):
    def __init__(self, parameters):
        self.parameters = parameters
        pass

    def step(self, loss):
        loss.backward(create_graph=True, retain_graph=True)
        for param in self.parameters:
            update = self.forward(param.grad, param)
            yield (param + update)

    def forward(self, grad, param=None):
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, differentiable=False):

        super().__init__(parameters)

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.state = {param: {} for param in parameters}
        self.differentiable = differentiable

    def forward(self, grad, param=None):

        state = self.state[param]
        step = state["step"] = state.get("step", 0) + 1
        exp_avg = state["exp_avg"] = state.get("exp_avg", torch.zeros_like(grad.data))
        exp_avg_sq = state["exp_avg_sq"] = state.get("exp_avg_sq", torch.zeros_like(grad.data))
        beta1, beta2 = self.betas

        exp_avg = exp_avg * beta1 + (1 - beta1) * (grad)
        exp_avg_sq = exp_avg_sq * (beta2) + (1 - beta2) * (grad) * (grad)
        denom = exp_avg_sq.sqrt() + self.eps

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1
        update = -step_size * exp_avg / denom

        state["exp_avg"] = exp_avg.detach().data
        state["exp_avg_sq"] = exp_avg_sq.detach().data

        if not self.differentiable:
            update = update.data

        return update

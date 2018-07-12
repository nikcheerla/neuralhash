
from __future__ import print_function

import numpy as np
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *

import IPython


class GramMatrix(nn.Module):
	def forward(self, input):
		N, C, H, W = input.size()
		features = input.view(N, C, H*W)  # resise F_XL into \hat F_XL
		G = torch.bmm(features, features.permute(0,2,1))  # compute the gram product
		return G.div(N*C*H*W)



class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.classifiers = nn.ModuleList([nn.Linear(512*8, 512*8), nn.Linear(512*8, 512*8)])

    def forward(self, x):
        x_sum = x
        for classifier in self.classifiers:
            x = classifier(x_sum)
            x_sum = x_sum + x
        return x_sum

from __future__ import print_function

import numpy as np
import random, sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

from utils import *

import IPython

from scipy.ndimage import filters

def resize_rect(x, x_val_range=0.3, y_val_range=0.3):
    x_scale = random.uniform(1-x_val_range, 1+x_val_range)
    y_scale = random.uniform(1-y_val_range, 1+y_val_range)
    grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], size=x.unsqueeze(0).size())
    if USE_CUDA: grid = grid.cuda()
    grid = torch.cat([grid[:, :, :, 0].unsqueeze(3)*y_scale, grid[:, :, :, 1].unsqueeze(3)*x_scale], dim=3)
    img = F.grid_sample(x.unsqueeze(0), grid)[0]
    return img

def color_jitter(x, jitter=0.1):

    R, G, B = (random.uniform(1-jitter, 1+jitter) for i in range(0, 3))
    x = torch.cat([x[0].unsqueeze(0)*R, x[1].unsqueeze(0)*G, x[2].unsqueeze(0)*B], dim=0)
    return x

def resize(x, min_val=100, max_val=300, rand_val=True, resize_val=224):
    if rand_val: resize_val = random.uniform(min_val, max_val)
    grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], size=torch.Size((1, 2, int(resize_val), int(resize_val))))
    if USE_CUDA: grid = grid.cuda()
    img = F.grid_sample(x.unsqueeze(0), grid)[0]
    return img

def scale(x, min_val=0.3, max_val=1.7, rand_val=True, scale_val=1):
    if rand_val: scale_val = random.uniform(min_val, max_val)
    grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], size=x.unsqueeze(0).size())
    if USE_CUDA: grid = grid.cuda()
    img = F.grid_sample(x.unsqueeze(0), grid*scale_val)[0]
    return img

def rotate(x, max_angle=90, rand_val=True, theta=0):
    if rand_val: theta = np.radians(random.randint(-max_angle, max_angle))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c], [0, 0]]).T
    grid = F.affine_grid(torch.FloatTensor(R).unsqueeze(0), size=x.unsqueeze(0).size())
    if USE_CUDA: grid = grid.cuda()
    img = F.grid_sample(x.unsqueeze(0), grid)[0]
    return img

def translate(x, min_val=0, max_val=30, rand_val=True, shift_vals=(0, 0)):
    if rand_val: 
        shift_vals = tuple(random.uniform(min_val*1.0/x.size()[1], max_val*1.0/x.size()[2]) for i in range(0, 2))
    M = np.array([[1, 0], [0, 1], [shift_vals[0], shift_vals[1]]]).T
    grid = F.affine_grid(torch.FloatTensor(M).unsqueeze(0), size=x.unsqueeze(0).size())
    if USE_CUDA: grid = grid.cuda()
    img = F.grid_sample(x.unsqueeze(0), grid)[0]
    return img

def gauss(x, min_sigma=1.1, max_sigma=1.8, rand_val=True, sigma=1):

    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1
    if rand_val: sigma = random.uniform(min_sigma, max_sigma)
    kernel = filters.gaussian_filter(kernel, sigma=sigma)
    gaussian = torch.Tensor(kernel).view(1, 1, 5, 5)
    gaussian = gaussian.repeat(3, 3, 1, 1)
    if USE_CUDA: gaussian.cuda()
    for i in range(0, 3): 
        for j in range(i+1, 3): 
            gaussian[i, j] = 0
            gaussian[j, i] = 0

    img = F.conv2d(x.unsqueeze(0), weight=(Variable(gaussian)).cuda(), padding=2)[0]
    return img

def noise(x, max_noise_val=0.02):
    noise = (Variable(torch.rand(x.size()))*max_noise_val).cuda()
    grid = F.affine_grid(torch.eye(3).unsqueeze(0)[:, 0:2], size=x.unsqueeze(0).size())
    if USE_CUDA: grid = grid.cuda()
    img = F.grid_sample((x+noise).unsqueeze(0), grid)[0]
    return img

def flip(x):
    randnum = random.randint(0, 1)
    if (randnum == 0):
        M = np.array([[-1, 0], [0, 1], [0, 0]]).T
    else:
        M = np.array([[1, 0], [0, -1], [0, 0]]).T
    grid = F.affine_grid(torch.FloatTensor(M).unsqueeze(0), size=x.unsqueeze(0).size())
    if USE_CUDA: grid = grid.cuda()
    img = F.grid_sample(x.unsqueeze(0), grid)[0]
    return img

def training_distribution(x):
    x = rotate(scale(x))
    #x = gauss(x)
    return x

def test_distribution(x):
    x = flip(rotate(scale(x)))
    x = gauss(x)
    return x

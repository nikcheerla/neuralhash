
from __future__ import print_function

import numpy as np
import random, sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

import IPython

from scipy.ndimage import filters

def identity(x):
    x = resize(x, rand_val=False, resize_val=224)
    return x

def affine(data, x=[1, 0, 0], y=[0, 1, 0]):
    return torch.tensor([x, y]).float().repeat(data.shape[0], 1, 1)

def resize(x, min_val=100, max_val=300, rand_val=True, resize_val=224):
    if rand_val: resize_val = random.uniform(min_val, max_val)
    grid = F.affine_grid(affine(x), size=torch.Size((x.shape[0], 3, resize_val, resize_val)))
    img = F.grid_sample(x, grid.to(x.device), padding_mode='border')
    return img

def resize_rect(x, x_val_range=0.3, y_val_range=0.3):
    x_scale = random.uniform(1-x_val_range, 1+x_val_range)
    y_scale = random.uniform(1-y_val_range, 1+y_val_range)
    grid = F.affine_grid(affine(x), size=x.size())
    grid = torch.cat([grid[:, :, :, 0].unsqueeze(3)*y_scale, grid[:, :, :, 1].unsqueeze(3)*x_scale], dim=3)
    img = F.grid_sample(x, grid.to(x.device), padding_mode='border')
    return img

def color_jitter(x, jitter=0.1):

    R, G, B = (random.uniform(1-jitter, 1+jitter) for i in range(0, 3))
    x = torch.cat([x[:, 0].unsqueeze(1)*R, \
                    x[:, 1].unsqueeze(1)*G, \
                    x[:, 2].unsqueeze(1)*B], dim=1)
    x = x.clamp(min=0, max=1)
    return x

def scale(x, min_val=0.6, max_val=1.4, rand_val=True, scale_val=1):
    if rand_val: scale_val = random.uniform(min_val, max_val)
    grid = F.affine_grid(affine(x), size=x.size())
    img = F.grid_sample(x, grid.to(x.device)*scale_val, padding_mode='border')
    return img

def rotate(x, max_angle=30, rand_val=True, theta=0):
    if rand_val: theta = np.radians(random.randint(-max_angle, max_angle))
    c, s = np.cos(theta), np.sin(theta)
    grid = F.affine_grid(affine(x, [c, s, 0], [-s, c, 0]), size=x.size())
    img = F.grid_sample(x, grid.to(x.device), padding_mode='border')
    return img

def translate(x, max_val=0.3):
    
    sx, sy = (random.uniform(-max_val, max_val) for i in range(0, 2))
    grid = F.affine_grid(affine(x, [1, 0, sx], [0, 1, sy]), size=x.size())
    img = F.grid_sample(x, grid.to(x.device), padding_mode='border')
    return img

def gauss(x, min_sigma=0.3, max_sigma=2, rand_val=True, sigma=1):

    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1
    if rand_val: sigma = random.uniform(min_sigma, max_sigma)
    kernel = filters.gaussian_filter(kernel, sigma=sigma)
    gaussian = torch.Tensor(kernel).view(1, 1, 5, 5)
    gaussian = gaussian.repeat(3, 3, 1, 1)
    for i in range(0, 3):
        for j in range(i+1, 3):
            gaussian[i, j] = 0
            gaussian[j, i] = 0

    x = F.conv2d(x, weight=gaussian.to(x.device), padding=2)
    return x

def noise(x, intensity=0.05):
    noise = torch.randn(x.size()).to(x.device)*intensity
    grid = F.affine_grid(affine(x), size=x.size())
    img = F.grid_sample((x+noise).clamp(min=1e-3, max=1), grid.to(x.device), padding_mode='border')
    return img

def flip(x):
    grid = F.affine_grid(affine(x, [-1, 0, 0], [0, 1, 0]), size=x.size())
    img = F.grid_sample(x, grid.to(x.device), padding_mode='border')
    return img

def whiteout(x, n=5, min_scale=0.04, max_scale=0.18):

    for i in range(0, n):
        w = int(random.uniform(min_scale, max_scale)*x.shape[2])
        h = int(random.uniform(min_scale, max_scale)*x.shape[3])
        sx, sy = random.randrange(0, x.shape[2] - w), random.randrange(0, x.shape[3] - h)
        mask, bias = torch.ones_like(x).to(x.device), torch.ones_like(x).to(x.device)
        mask[:, :, sx:(sx+w), sy:(sy+h)] = 0.0

        R, G, B = (random.random() for i in range(0, 3))
        bias = torch.cat([bias[:, 0].unsqueeze(1)*R, \
                        bias[:, 1].unsqueeze(1)*G, \
                        bias[:, 2].unsqueeze(1)*B], dim=1)

        if random.randint(0, 1) == 1:
            bias = noise(bias, intensity=0.5)

        x = mask*x + (1.0-mask)*bias
    return x

def training(x):
<<<<<<< HEAD
    # x = random.choice([gauss, noise, color_jitter, lambda x: x, lambda x: x])(x)
    # x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    # x = random.choice([gauss, noise, color_jitter, lambda x: x])(x)
    # x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    # x = random.choice([gauss, noise, color_jitter, lambda x: x, lambda x: x])(x)
    x = resize_rect(x)
    x = rotate(scale(x, 0.6, 1.4), max_angle=30)
    x = gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = translate(x)
=======
    x = random.choice([gauss, noise, color_jitter, lambda x: x])(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    x = random.choice([gauss, noise, color_jitter, lambda x: x])(x)
>>>>>>> 7e6d1ed0f369f715c368a7bdc5cd74f0253e2789
    x = identity(x)
    return x

def encoding(x):
    return training(x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = im.load("images/house.png")
    img = im.torch(img).unsqueeze(0)

    plt.imsave(f"output/house_orig.jpg", im.numpy(img.squeeze()))
    # returns an image after a series of transformations

    for i in range(15):
        plt.imsave(f"output/house_test_{i}.jpg", im.numpy(training(img).squeeze()))



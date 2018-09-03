
from __future__ import print_function

import numpy as np
import random, sys, os, timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

import IPython

from scipy.ndimage import filters

import torchvision
from io import BytesIO
from PIL import Image

def identity(x):
    x = resize(x, rand_val=False, resize_val=224)
    return x

def affine(data, x=[1, 0, 0], y=[0, 1, 0]):
    return dtype([x, y], device=data.device).float().repeat(data.shape[0], 1, 1)

def resize(x, min_val=100, max_val=300, rand_val=True, resize_val=224):
    if rand_val: resize_val = random.uniform(min_val, max_val)
    resize_val = int(resize_val)
    grid = F.affine_grid(affine(x), size=torch.Size((x.shape[0], 3, resize_val, resize_val)))
    img = F.grid_sample(x, grid, padding_mode='border')
    return img

def resize_rect(x, x_val_range=0.3, y_val_range=0.3, rand_val=True, ratio=0.8):
    if rand_val:
        x_scale = random.uniform(1-x_val_range, 1+x_val_range)
        y_scale = random.uniform(1-y_val_range, 1+y_val_range)
    else:
        x_scale = random.uniform(0, 1 - ratio) + 1
        y_scale = x_scale / ratio

    grid = F.affine_grid(affine(x), size=x.size())
    grid = torch.cat([grid[:, :, :, 0].unsqueeze(3)*y_scale, grid[:, :, :, 1].unsqueeze(3)*x_scale], dim=3)
    img = F.grid_sample(x, grid, padding_mode='border')
    return img

def color_jitter(x, jitter=0.1):

    R, G, B = (random.uniform(1-jitter, 1+jitter) for i in range(0, 3))
    x = torch.cat([x[:, 0].unsqueeze(1)*R, \
                    x[:, 1].unsqueeze(1)*G, \
                    x[:, 2].unsqueeze(1)*B], dim=1)
    return x.clamp(min=0, max=1)

def scale(x, min_val=0.6, max_val=1.4, rand_val=True, scale_val=1):
    if rand_val: scale_val = random.uniform(min_val, max_val)
    grid = F.affine_grid(affine(x), size=x.size())
    img = F.grid_sample(x, grid*scale_val, padding_mode='border')
    return img

def elastic(x, x_val_range=0.3, y_val_range=0.3, p=0.1):
    n = 3
    N, C, H, W = x.shape
    H_c, W_c = int((H*W*p)**0.5), int((H*W*p)**0.5)

    x_scale = random.uniform(1-x_val_range, 1+x_val_range)
    y_scale = random.uniform(1-y_val_range, 1+y_val_range)
    grid = F.affine_grid(affine(x), size=x.size())
    grid_y = grid[:, :, :, 0].unsqueeze(3)
    grid_x = grid[:, :, :, 1].unsqueeze(3)

    # stetch/contract n small image regions
    for i in range(0, n):
        x_coord = int(random.uniform(0, H-H_c))
        y_coord = int(random.uniform(0, W-W_c))
        grid_y[:,x_coord:x_coord+H_c,y_coord:y_coord+W_c] = grid_y[:,x_coord:x_coord+H_c,y_coord:y_coord+W_c] * y_scale
        grid_x[:,x_coord:x_coord+H_c,y_coord:y_coord+W_c] = grid_x[:,x_coord:x_coord+H_c,y_coord:y_coord+W_c] * x_scale

    grid = torch.cat([grid_y, grid_x], dim=3)
    img = F.grid_sample(x, grid, padding_mode='border')
    return img

def rotate(x, max_angle=30, rand_val=True, theta=0):
    if rand_val: theta = np.radians(random.randint(-max_angle, max_angle))
    c, s = np.cos(theta), np.sin(theta)
    grid = F.affine_grid(affine(x, [c, s, 0], [-s, c, 0]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode='border')
    return img

def translate(x, max_val=0.3, rand_val=True, radius=0.15):
    if rand_val:
        sx, sy = (random.uniform(-max_val, max_val) for i in range(0, 2))
    else:
        theta = random.uniform(-np.pi, np.pi)
        sx, sy = np.cos(theta) * radius, np.sin(theta) * radius
    grid = F.affine_grid(affine(x, [1, 0, sx], [0, 1, sy]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode='border')
    return img

def gauss(x, min_sigma=0.3, max_sigma=2, rand_val=True, sigma=1):
    if rand_val: sigma = random.uniform(min_sigma, max_sigma)
    filter = gaussian_filter(kernel_size=7, sigma=sigma)
    x = F.conv2d(x, weight=filter.to(x.device), bias=None, groups=3, padding=2)
    return x.clamp(min=1e-3, max=1)

def motion_blur(x, filter=motion_blur_filter()):
    x = F.conv2d(x, weight=filter.to(x.device), bias=None, groups=3)
    return x.clamp(min=1e-3, max=1)

def noise(x, intensity=0.05):
    noise = dtype(x.size(), device=x.device).normal_().requires_grad_(False)*intensity
    img = (x + noise).clamp(min=1e-3, max=1)
    return img

def impulse_noise(x, intensity=0.1):
    num = 10000
    _, _, H, W = x.shape
    x_coords = np.random.randint(low=0, high=H, size=(int(intensity*num),))
    y_coords = np.random.randint(low=0, high=W, size=(int(intensity*num),))

    R, G, B = (random.uniform(0, 1) for i in range(0, 3))
    mask = torch.ones_like(x) 
    mask[:, 0, x_coords, y_coords] = R
    mask[:, 1, x_coords, y_coords] = G
    mask[:, 2, x_coords, y_coords] = B
    return x * mask

def flip(x):
    grid = F.affine_grid(affine(x, [-1, 0, 0], [0, 1, 0]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode='border')
    return img

def whiteout(x, n=6, min_scale=0.04, max_scale=0.2, rand_val=True, scale=0.1):

    noise = dtype(x.size(), device=x.device).normal_().requires_grad_(False)*0.5

    for i in range(0, n):
        if rand_val:
            w = int(random.uniform(min_scale, max_scale)*x.shape[2])
            h = int(random.uniform(min_scale, max_scale)*x.shape[3])
        else:
            w, h = int(scale*x.shape[2]), int(scale*x.shape[3])

        sx, sy = random.randrange(0, x.shape[2] - w), random.randrange(0, x.shape[3] - h)
        
        mask = torch.ones_like(x)
        mask[:, :, sx:(sx+w), sy:(sy+h)] = 0.0

        R, G, B = (random.random() for i in range(0, 3))
        bias = dtype([R, G, B], device=x.device).view(1, 3, 1, 1).expand_as(mask)

        if random.randint(0, 1): 
            bias = (bias + noise).clamp(min=1e-3, max=1)
        x = mask*x + (1.0-mask)*bias
    return x

def crop(x, p=0.1):
    N, C, H, W = x.shape
    H_c, W_c = int((H*W*p)**0.5), int((H*W*p)**0.5)
    x_coord = int(random.uniform(0, H-H_c))
    y_coord = int(random.uniform(0, W-W_c))

    mask = torch.zeros_like(x)
    mask[:, :, x_coord:x_coord+H_c, y_coord:y_coord+W_c] = 1.0
    return x * mask

def convertToJpeg(x):
    x = x.squeeze()
    x = transforms.ToPILImage()(x)
    with BytesIO() as f:
        x.save(f, format='JPEG')
        f.seek(0)
        ima_jpg = Image.open(f)
        return transforms.ToTensor()(ima_jpg)

def brightness(x, max_brightness=0.4, rand_val=True, brightness_val=0.2):
    if rand_val: brightness_val = random.uniform(-max_brightness, max_brightness)
    x = torch.cat([x[:, 0].unsqueeze(1)+brightness_val, \
                    x[:, 1].unsqueeze(1)+brightness_val, \
                    x[:, 2].unsqueeze(1)+brightness_val], dim=1)
    return x.clamp(min=0, max=1)

def contrast(x, min_contrast=0.4, max_contrast=1.4, rand_val=True, contrast_val=1.1):
    if rand_val: contrast_val = random.uniform(-min_contrast, max_contrast)
    x = torch.cat([x[:, 0].unsqueeze(1)*contrast_val, \
                    x[:, 1].unsqueeze(1)*contrast_val, \
                    x[:, 2].unsqueeze(1)*contrast_val], dim=1)
    return x.clamp(min=0, max=1)

def blur(x, min_blur=2, max_blur=5, rand_val=True, blur_val=4):
    if rand_val: blur_val = int(random.uniform(min_blur, max_blur))
    N, C, H, W = x.shape

    # downsampling
    out_size_h = H//blur_val
    out_size_w = W//blur_val
    a1 = torch.linspace(-1, 1, out_size_h).view(-1, 1).repeat(1, out_size_w)
    b1 = torch.linspace(-1, 1, out_size_w).repeat(out_size_h, 1)
    grid = torch.cat((a1.unsqueeze(2), b1.unsqueeze(2)), 2)
    grid.unsqueeze_(0)
    image_small = F.grid_sample(x, grid)

    # upsampling
    a2 = torch.linspace(-1, 1, H).view(-1, 1).repeat(1, W)
    b2 = torch.linspace(-1, 1, W).repeat(H, 1)
    grid = torch.cat((a2.unsqueeze(2), b2.unsqueeze(2)), 2)
    grid.unsqueeze_(0)
    image = F.grid_sample(image_small, grid)
    
    return image

def training(x):
    x = random.choice([gauss, noise, color_jitter, whiteout, lambda x: x, lambda x: x])(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    x = random.choice([flip, crop, lambda x: x])(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    x = random.choice([gauss, noise, color_jitter, crop, lambda x: x, lambda x: x])(x)
    x = identity(x)
    return x

def holdout(x):
    x = random.choice([noise, color_jitter, whiteout, lambda x: x, lambda x: x])(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    x = random.choice([flip, crop, lambda x: x])(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
    x = random.choice([noise, color_jitter, crop, lambda x: x, lambda x: x])(x)
    x = identity(x)
    return x

def encoding(x):
    return training(x)

# def inference(x):
#     x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
#     x = random.choice([gauss, noise, color_jitter, lambda x: x])(x)
#     x = random.choice([rotate, resize_rect, scale, translate, flip, lambda x: x])(x)
#     x = identity(x)
#     return x

# def easy(x):
#     x = resize_rect(x)
#     x = rotate(scale(x, 0.6, 1.4), max_angle=30)
#     x = gauss(x, min_sigma=0.8, max_sigma=1.2)
#     x = translate(x)
#     x = identity(x)
#     return x

### NOT differentiable ###
def convertToJpeg(x, q=10):
    jpgs = []
    for img in x:
        img = img.squeeze()
        img = torchvision.transforms.ToPILImage()(img.cpu())
        with BytesIO() as f:
            img.save(f, format='JPEG', quality=int(q))
            f.seek(0)
            ima_jpg = Image.open(f)
            jpgs.append(torchvision.transforms.ToTensor()(ima_jpg))
    return torch.stack(jpgs).to(DEVICE)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = im.load("images/house.png")
    img = im.torch(img).unsqueeze(0)
    plt.imsave("output/house-jpeg-transform.jpg", im.numpy(convertToJpeg(img).squeeze()))

    for transform in [identity, resize, resize_rect, color_jitter, crop,
                      scale, rotate, translate, gauss, noise, flip, whiteout,
                      training, encoding]:
        time = timeit.timeit(lambda: im.numpy(transform(img).squeeze()), number=40)
        print (f"{transform.__name__}: {time:0.5f}")



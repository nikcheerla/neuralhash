
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


def affine(data, x=[1, 0, 0], y=[0, 1, 0]):
    return dtype([x, y], device=data.device).float().repeat(data.shape[0], 1, 1)


def sample(min_val=0, max_val=1, random_generator=random.uniform):
    def wrapper(transform):
        class RandomSampler:
            """Wrapper class that turns transforms into dynamic
            callables."""

            def __init__(self, transform, min_val, max_val, random_generator):
                self.min_val = min_val
                self.max_val = max_val
                self.transform = transform
                self.random_generator = random_generator
                self.__name__ = transform.__name__

            def __call__(self, x, val=None, **kwargs):
                if val == None:
                    return self.transform(x)
                return self.transform(x, val, **kwargs)

            def random(self, x, **kwargs):
                val = self.random_generator(self.min_val, self.max_val)
                return self.transform(x, val, **kwargs)

        return RandomSampler(transform, min_val, max_val, random_generator)

    return wrapper


@sample(0, 0)
def identity(x, val=None):
    x = resize(x, 224)
    return x


@sample(100, 300)
def resize(x, val=224):
    val = int(val)
    grid = F.affine_grid(affine(x), size=torch.Size((x.shape[0], 3, val, val)))
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.8, 1.2)
def resize_rect(x, ratio=0.8):

    x_scale = random.uniform(0, 1 - ratio) + 1
    y_scale = x_scale / ratio

    grid = F.affine_grid(affine(x), size=x.size())
    grid = torch.cat(
        [
            grid[:, :, :, 0].unsqueeze(3) * y_scale,
            grid[:, :, :, 1].unsqueeze(3) * x_scale,
        ],
        dim=3,
    )
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.05, 0.2)
def color_jitter(x, jitter=0.1):
    R, G, B = (random.uniform(1 - jitter, 1 + jitter) for i in range(0, 3))
    x = torch.cat(
        [x[:, 0].unsqueeze(1) * R, x[:, 1].unsqueeze(1) * G, x[:, 2].unsqueeze(1) * B],
        dim=1,
    )
    return x.clamp(min=0, max=1)


@sample(0.6, 1.4)
def scale(x, scale_val=1):
    grid = F.affine_grid(affine(x), size=x.size())
    img = F.grid_sample(x, grid * scale_val, padding_mode="border")
    return img


@sample(-60, 60)
def rotate(x, theta=45):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    grid = F.affine_grid(affine(x, [c, s, 0], [-s, c, 0]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.1, 0.4)
def translate(x, radius=0.15):
    theta = random.uniform(-np.pi, np.pi)
    sx, sy = np.cos(theta) * radius, np.sin(theta) * radius
    grid = F.affine_grid(affine(x, [1, 0, sx], [0, 1, sy]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.3, 2)
def gauss(x, sigma=1):
    filter = gaussian_filter(kernel_size=7, sigma=sigma)
    x = F.conv2d(x, weight=filter.to(x.device), bias=None, groups=3, padding=2)
    return x.clamp(min=1e-3, max=1)


@sample(0.01, 0.2)
def noise(x, intensity=0.05):
    noise = dtype(x.size(), device=x.device).normal_().requires_grad_(False) * intensity
    img = (x + noise).clamp(min=1e-3, max=1)
    return img


@sample(0, 1)
def flip(x, val):
    if val < 0.5:
        return x
    grid = F.affine_grid(affine(x, [-1, 0, 0], [0, 1, 0]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.01, 0.2)
def whiteout(x, scale=0.1, n=6):

    noise = dtype(x.size(), device=x.device).normal_().requires_grad_(False) * 0.5

    for i in range(0, n):
        w, h = int(scale * x.shape[2]), int(scale * x.shape[3])
        sx, sy = (
            random.randrange(0, x.shape[2] - w),
            random.randrange(0, x.shape[3] - h),
        )

        mask = torch.ones_like(x)
        mask[:, :, sx : (sx + w), sy : (sy + h)] = 0.0

        R, G, B = (random.random() for i in range(0, 3))
        bias = dtype([R, G, B], device=x.device).view(1, 3, 1, 1).expand_as(mask)

        if random.randint(0, 1):
            bias = (bias + noise).clamp(min=1e-3, max=1)
        x = mask * x + (1.0 - mask) * bias
    return x


@sample(10, 100)
def crop(x, p=0.1):
    N, C, H, W = x.shape
    H_c, W_c = int((H * W * p) ** 0.5), int((H * W * p) ** 0.5)
    x_coord = int(random.uniform(0, H - H_c))
    y_coord = int(random.uniform(0, W - W_c))

    mask = torch.zeros_like(x)
    mask[:, :, x_coord : x_coord + H_c, y_coord : y_coord + W_c] = 1.0
    return x * mask


## NOT DIFFERENTIABLE ##
@sample(10, 100)
def jpeg_transform(x, q=50):
    jpgs = []
    for img in x:
        img = img.squeeze()
        img = torchvision.transforms.ToPILImage()(img.cpu())
        with BytesIO() as f:
            img.save(f, format="JPEG", quality=int(q))
            f.seek(0)
            ima_jpg = Image.open(f)
            jpgs.append(torchvision.transforms.ToTensor()(ima_jpg))
    return torch.stack(jpgs).to(DEVICE)


def training(x):
    _ = sample(0, 0)(lambda x, val: x)
    x = random.choice([gauss, noise, color_jitter, whiteout, _, _]).random(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, _, _]).random(x)
    x = random.choice([flip, crop, _]).random(x)
    x = random.choice([rotate, resize_rect, scale, translate, flip, _]).random(x)
    x = random.choice([gauss, noise, color_jitter, crop, _, _]).random(x)
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = im.load("images/house.png")
    img = im.torch(img).unsqueeze(0)

    for transform in [
        identity,
        resize,
        resize_rect,
        color_jitter,
        crop,
        scale,
        rotate,
        translate,
        gauss,
        noise,
        flip,
        whiteout,
    ]:
        transformed = im.numpy(transform.random(img).squeeze())
        plt.imsave(f"output/{transform.__name__}.jpg", transformed)
        time = timeit.timeit(
            lambda: im.numpy(transform.random(img).squeeze()), number=40
        )
        print(f"{transform.__name__}: {time:0.5f}")

    for i in range(0, 10):
        transformed = im.numpy(encoding(img).squeeze())
        plt.imsave(f"output/encoded_{i}.jpg", transformed)

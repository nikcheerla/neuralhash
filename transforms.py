
from __future__ import print_function

import numpy as np
import random, sys, os, timeit, math

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


def sample(min_val=0, max_val=1, plot_range=None, generator=None):

    if plot_range is None:
        span = max_val - min_val
        min_plot = min_val - span / 2.0
        max_plot = max_val + span / 2.0
        if min_val >= 0 and max_val >= 0:
            min_plot = max(min_plot, 0)
        if min_val <= 0 and max_val <= 0:
            max_plot = min(max_plot, 0)
        plot_range = (min_plot, max_plot)

    if generator is None:
        generator = lambda: random.uniform(min_val, max_val)

    def wrapper(transform):
        class RandomSampler:
            """Wrapper class that turns transforms into dynamic
            callables."""

            def __init__(self, transform, plot_range, generator):
                self.transform = transform
                self.plot_range = plot_range
                self.generator = generator
                self.__name__ = transform.__name__

            def __call__(self, x, val=None, **kwargs):
                if val == None:
                    return self.transform(x)
                return self.transform(x, val, **kwargs)

            def random(self, x, **kwargs):
                val = self.generator()
                return self.transform(x, val, **kwargs)

        return RandomSampler(transform, plot_range, generator)

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


@sample(0.6, 1.4)
def resize_rect(x, ratio=0.8):

    x_scale = random.uniform(ratio, 1)
    y_scale = x_scale / ratio

    grid = F.affine_grid(affine(x), size=x.size())
    grid = torch.cat([grid[:, :, :, 0].unsqueeze(3) * y_scale, grid[:, :, :, 1].unsqueeze(3) * x_scale], dim=3)
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.05, 0.2)
def color_jitter(x, jitter=0.1):
    R, G, B = (random.uniform(1 - jitter, 1 + jitter) for i in range(0, 3))
    x = torch.cat([x[:, 0].unsqueeze(1) * R, x[:, 1].unsqueeze(1) * G, x[:, 2].unsqueeze(1) * B], dim=1)
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


@sample(0.9, 1.1, plot_range=(0.8, 1.2))
def elastic(x, ratio=0.8, n=3, p=0.1):

    N, C, H, W = x.shape
    H_c, W_c = int((H * W * p) ** 0.5), int((H * W * p) ** 0.5)

    grid = F.affine_grid(affine(x), size=x.size())
    grid_y = grid[:, :, :, 0].unsqueeze(3)
    grid_x = grid[:, :, :, 1].unsqueeze(3)

    # stretch/contract n small image regions
    for i in range(0, n):
        x_coord = int(random.uniform(0, H - H_c))
        y_coord = int(random.uniform(0, W - W_c))

        x_scale = random.uniform(0, 1 - ratio) + 1
        y_scale = x_scale / ratio
        grid_y[:, x_coord : x_coord + H_c, y_coord : y_coord + W_c] = (
            grid_y[:, x_coord : x_coord + H_c, y_coord : y_coord + W_c] * y_scale
        )
        grid_x[:, x_coord : x_coord + H_c, y_coord : y_coord + W_c] = (
            grid_x[:, x_coord : x_coord + H_c, y_coord : y_coord + W_c] * x_scale
        )

    grid = torch.cat([grid_y, grid_x], dim=3)
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.1, 0.4)
def translate(x, radius=0.15):
    theta = random.uniform(-np.pi, np.pi)
    sx, sy = np.cos(theta) * radius, np.sin(theta) * radius
    grid = F.affine_grid(affine(x, [1, 0, sx], [0, 1, sy]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0.3, 2, plot_range=(0.01, 4))
def gauss(x, sigma=1):
    filter = gaussian_filter(kernel_size=7, sigma=sigma)
    x = F.conv2d(x, weight=filter.to(x.device), bias=None, groups=3, padding=2)
    return x.clamp(min=1e-3, max=1)


@sample(5, 9)
def motion_blur(x, val):
    filter = motion_blur_filter(kernel_size=int(val))
    x = F.conv2d(x, weight=filter.to(x.device), bias=None, groups=3)
    return x.clamp(min=1e-3, max=1)


@sample(0.03, 0.06)
def noise(x, intensity=0.05):
    noise = dtype(x.size(), device=x.device).normal_().requires_grad_(False) * intensity
    img = (x + noise).clamp(min=1e-3, max=1)
    return img


@sample(0, 1, plot_range=(0, 1))
def flip(x, val):
    if val < 0.5:
        return x
    grid = F.affine_grid(affine(x, [-1, 0, 0], [0, 1, 0]), size=x.size())
    img = F.grid_sample(x, grid, padding_mode="border")
    return img


@sample(0, 0.2)
def impulse_noise(x, intensity=0.1):
    num = 10000
    _, _, H, W = x.shape
    x_coords = np.random.randint(low=0, high=H, size=(int(intensity * num),))
    y_coords = np.random.randint(low=0, high=W, size=(int(intensity * num),))

    R, G, B = (random.uniform(0, 1) for i in range(0, 3))
    mask = torch.ones_like(x)
    mask[:, 0, x_coords, y_coords] = R
    mask[:, 1, x_coords, y_coords] = G
    mask[:, 2, x_coords, y_coords] = B
    return x * mask


@sample(0.01, 0.2, plot_range=(0.01, 0.3))
def whiteout(x, scale=0.1, n=6):

    noise = dtype(x.size(), device=x.device).normal_().requires_grad_(False) * 0.5

    for i in range(0, n):
        w, h = int(scale * x.shape[2]), int(scale * x.shape[3])
        sx, sy = (random.randrange(0, x.shape[2] - w), random.randrange(0, x.shape[3] - h))

        mask = torch.ones_like(x)
        mask[:, :, sx : (sx + w), sy : (sy + h)] = 0.0

        R, G, B = (random.random() for i in range(0, 3))
        bias = dtype([R, G, B], device=x.device).view(1, 3, 1, 1).expand_as(mask)

        if random.randint(0, 1):
            bias = (bias + noise).clamp(min=1e-3, max=1)
        x = mask * x + (1.0 - mask) * bias
    return x


@sample(0.5, 1, plot_range=(0.2, 1))
def crop(x, p=0.6):
    N, C, H, W = x.shape
    H_c, W_c = int((H * W * p) ** 0.5), int((H * W * p) ** 0.5)
    x_coord = int(random.uniform(0, H - H_c))
    y_coord = int(random.uniform(0, W - W_c))

    mask = torch.zeros_like(x)
    mask[:, :, x_coord : x_coord + H_c, y_coord : y_coord + W_c] = 1.0
    return x * mask


## NOT DIFFERENTIABLE ##
@sample(50, 100, plot_range=(10, 100))
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


@sample(-0.4, 0.4)
def brightness(x, brightness_val=0.2):
    x = torch.cat(
        [
            x[:, 0].unsqueeze(1) + brightness_val,
            x[:, 1].unsqueeze(1) + brightness_val,
            x[:, 2].unsqueeze(1) + brightness_val,
        ],
        dim=1,
    )
    return x.clamp(min=0, max=1)


@sample(0.5, 1.5)
def contrast(x, factor=0.1):
    R = (x[:, 0].unsqueeze(1) - 0.5) * factor + 0.5
    G = (x[:, 1].unsqueeze(1) - 0.5) * factor + 0.5
    B = (x[:, 2].unsqueeze(1) - 0.5) * factor + 0.5
    x = torch.cat([R, G, B], dim=1)
    return x.clamp(min=0, max=1)


@sample(2, 6)
def blur(x, blur_val=4):
    N, C, H, W = x.shape

    # downsampling
    out_size_h = H // max(int(blur_val), 2)
    out_size_w = W // max(int(blur_val), 2)
    grid = F.affine_grid(affine(x), size=torch.Size((x.shape[0], 3, out_size_h, out_size_w)))
    x = F.grid_sample(x, grid, padding_mode="border")

    # upsampling
    grid = F.affine_grid(affine(x), size=torch.Size((x.shape[0], 3, H, W)))
    x = F.grid_sample(x, grid, padding_mode="border")

    return x


@sample(2, 8)
def pixilate(x, res=4):
    res = max(2, min(res, 8))
    res = max(2, min(2 ** (math.ceil(math.log(res, 2))), 8))
    return F.upsample(F.avg_pool2d(x, int(res)), scale_factor=int(res))


# def training(x):
#     _ = sample(0, 0)(lambda x, val: x)
#     x = random.choice([gauss, noise, color_jitter, whiteout, _, _]).random(x)
#     x = random.choice([rotate, resize_rect, scale, translate, flip, _, _]).random(x)
#     x = random.choice([flip, crop, _]).random(x)
#     x = random.choice([rotate, resize_rect, scale, translate, flip, _]).random(x)
#     x = random.choice([gauss, noise, color_jitter, crop, _, _]).random(x)
#     x = identity(x)
#     return x


def training(x):
    _ = sample(0, 0)(lambda x, val: x)
    t_list = [
        identity,
        elastic,
        motion_blur,
        impulse_noise,
        jpeg_transform,
        brightness,
        contrast,
        blur,
        pixilate,
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
        _,
        _,
    ]
    x = random.choice(t_list).random(x)
    x = random.choice(t_list).random(x)
    x = random.choice(t_list).random(x)
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
        elastic,
        motion_blur,
        impulse_noise,
        jpeg_transform,
        brightness,
        contrast,
        blur,
        pixilate,
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
        plt.imsave(f"output/encoded_{transform.__name__}.jpg", transformed)
        time = timeit.timeit(lambda: im.numpy(transform.random(img).squeeze()), number=40)
        x_min, x_max = transform.plot_range
        print(f"{transform.__name__}: ({x_min} - {x_max}) {time:0.5f}")

    for i in range(0, 10):
        transformed = im.numpy(encoding(img).squeeze())
        plt.imsave(f"output/encoded_{i}.jpg", transformed)

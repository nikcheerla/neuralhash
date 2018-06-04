from __future__ import print_function
import IPython

import random, sys, os, glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import *
import transforms

from encoding import encode_binary
from models import DecodingNet, DecodingDNN

from scipy.ndimage import filters
from scipy import stats

# returns an image after a series of transformations
def p(x):
    x = transforms.resize_rect(x)
    x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = transforms.translate(x)
    x = transforms.resize(x, rand_val=False, resize_val=224)
    return x

def sweep(image, output_file, min_val, max_val, step, transform, code, model):
    val = min_val
    res_bin = []
    res_mse = []
    # model.orient(image, 0)
    print(avg_gradient(image.cpu().data.numpy()))
    while val <= max_val:
        transformed = transform(image, val)
        preds = model.forward(transformed, distribution=p, n=96).mean(dim=0).data.cpu().numpy()
        mse_loss = binary.mse_dist(preds, code)
        binary_loss = binary.distance(code, preds)

        res_bin.append((val, binary_loss))
        res_mse.append((val, mse_loss))
        # model.orient(transformed, val)
        avg_grad = avg_gradient(transformed.cpu().data.numpy())
        print(avg_grad, np.round(val, 4), np.round(avg_grad - val, 4))

        # print("mse: ", np.round(mse_loss, 4))
        val += step

    x, bits_off = zip(*res_bin)
    x, mse = zip(*res_mse)
    fig, ax1 = plt.subplots()
    ax1.plot(x, bits_off, 'b')
    ax1.set_ylim(0, TARGET_SIZE//2)
    ax1.set_ylabel('Number Incorrect Bits')
    ax2 = ax1.twinx()
    ax2.plot(x, mse, 'r')
    ax2.set_ylim(0, 0.25)
    ax2.set_ylabel('Mean Squared Error')
    plt.savefig(OUTPUT_DIR + output_file); 
    plt.cla()

def avg_gradient(image):
    grey = np.mean(image, axis=2)
    blur = filters.gaussian_filter(grey, 1)
    # Kernel for Gradient in x-direction
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    # Kernel for Gradient in y-direction
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)

    # Apply kernels to the image
    Ix = filters.convolve(grey, Kx)
    Iy = filters.convolve(grey, Ky)

    # return the hypothenuse of (Ix, Iy)
    D = np.arctan2(Ix, Iy)
    # print(Ix)
    bins = np.linspace(-1*np.pi, np.pi, 40)
    quantized = np.digitize(D, bins)
    mode, count = stats.mode(quantized, axis=None)
    # print(quantized)
    # print(bins[mode[0]])
    # print(stats.mode(quantized, axis=None))
    # print(bins[mode], np.mean(D))
    return bins[mode[0]]

def test_transforms(model=None):
    images = ["car.jpg"]
    if model == None:
        model = DecodingNet()
    model.drawLastLayer('output/testviz.png')
    for image_file in images:
        image = im.load("images/" + image_file)
        if image is None: continue
        print(avg_gradient(image))
        code = binary.random(n=TARGET_SIZE)
        encoded_img = encode_binary(image, model, target=code, verbose=True)
        avg_gradient(encoded_img)

        sweep(im.torch(encoded_img), image_file[:-4] + "_rotate.jpg", -0.6, 0.6, 0.02, 
            lambda x, val: transforms.rotate(x, rand_val=False, theta=val), code, model)
        # sweep(im.torch(encoded_img), image_file[:-4] + "_scale.jpg", 0.5, 1.5, 0.02, 
        #     lambda x, val: transforms.scale(x, rand_val=False, scale_val=val), code, model)
        # sweep(im.torch(encoded_img), image_file[:-4] + "_noise.jpg", 0, 0.5, 0.005, 
        #     lambda x, val: transforms.noise(x, max_noise_val=val), code, model)
        # sweep(im.torch(encoded_img), image_file[:-4] + "_translatex.jpg", -50.0, 50.0, 5.0, 
        #     lambda x, val: transforms.translate(x, rand_val=False, shift_vals=(0, val)), code, model)
        # sweep(im.torch(encoded_img), image_file[:-4] + "_translatey.jpg", -50.0, 50.0, 5, 
        #     lambda x, val: transforms.translate(x, rand_val=False, shift_vals=(val, 0)), code, model)

def compare_image(original_file, transformed_file):
    original_img = im.load(original_file)
    transformed_img = im.load(transformed_file)
    original_code = decode(original_img)
    print("original code: " + binary.str(original_code))
    preds = decode_raw(im.torch(transformed_img))
    mse_loss = binary.mse_dist(preds, code)
    res.append((val, mse_loss))
    print("mse: ", np.round(mse_loss, 4))
    print("# of diff: ", binary.distance(code, preds))

if __name__ == "__main__":
    test_transforms()




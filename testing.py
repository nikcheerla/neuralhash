from __future__ import print_function
import IPython

import random, sys, os, glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import im, binary
import transforms

from encoding import encode_binary
from models import DecodingNet


# returns an image after a series of transformations
def p(x):
    # x = transforms.resize_rect(x)
    x = transforms.rotate(transforms.scale(x), max_angle=90)
    x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
    x = transforms.translate(x)
    x = transforms.resize(x, rand_val=False, resize_val=224)
    return x

def sweep(image, output_file, min_val, max_val, step, transform, code, model):
    val = min_val
    res = []
    while val <= max_val:
        transformed = transform(image, val)
        preds = model.forward(transformed, distribution=p, n=96).data.cpu().numpy()
        mse_loss = binary.mse_dist(preds, code)
        res.append((val, mse_loss))
        print("mse: ", np.round(mse_loss, 4))
        print("# of diff: ", binary.distance(code, preds))
        val += step
    x, y = zip(*res)
    plt.plot(x, y); plt.savefig("/output/" + output_file); plt.cla()

def test_transforms():
    images = ["car.jpg"]
    model = DecodingNet()

    for image_file in images:
        image = im.load("images/" + image_file)
        if image is None: continue
        code = binary.random(n=32)

        encoded_img = encode_binary(image, model, target=code, verbose=True)
        code = binary.get(model.forward(im.torch(encoded_img)))

        sweep(im.torch(encoded_img), image_file + "_rotate.jpg", -1.5, 1.5, 0.1, 
            lambda x, val: transforms.rotate(x, rand_val=False, theta=val), code, model)
        sweep(im.torch(encoded_img), image_file + "_scale.jpg", 0.5, 1.5, 0.05, 
            lambda x, val: transforms.scale(x, rand_val=False, scale_val=val), code, model)

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

test_transforms()




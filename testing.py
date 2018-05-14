from __future__ import print_function
import IPython

import random, sys, os, glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from utils import im, binary

from encoding import encode_binary
from transforms import rotate, scale
from models import DecodingNet

def sweep(image, output_file, min_val, max_val, step, transform, code, model):
    val = min_val
    res = []
    while val <= max_val:
        transformed = transform(image, val)
        preds = model.forward(transformed).data.cpu().numpy()
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
            lambda x, val: rotate(x, rand_val=False, theta=val), code, model)
        sweep(im.torch(encoded_img), image_file + "_scale.jpg", -1.5, 1.5, 0.1, 
            lambda x, val: scale(x, rand_val=False, scale_val=val), code, model)

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




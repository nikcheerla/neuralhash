from __future__ import print_function
import IPython

import random, sys, os, glob

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from utils import im, binary

from api import encode, decode, decode_raw
from transforms import rotate, scale

def sweep(image, output_file, min_val, max_val, step, transform, code):
    val = min_val
    res = []
    while val <= max_val:
        transformed = transform(image, val)
        preds = decode_raw(transformed)
        mse_loss = binary.mse_dist(preds, code)
        res.append((val, mse_loss))
        print("mse: ", np.round(mse_loss, 4))
        print("# of diff: ", binary.distance(code, preds))
        val += step
    x, y = zip(*res)
    plt.plot(x, y); plt.savefig("/output/" + output_file); plt.cla()

def test_transforms():
    images = ["cat.jpg"]

    for image_file in images:
        image = im.load("images/" + image_file)
        if image is None: continue

        code = binary.random(n=32)

        encoded_img = im.load("images/Scream Encoded.jpeg")#encode(image, code, verbose=True)
        code = decode(encoded_img)
        sweep(im.torch(encoded_img), image_file + "_rotate.jpg", -1.5, 1.5, 0.8, lambda x, val: rotate(x, rand_val=False, theta=val), code)
        #sweep(image, image_file + "_scale.jpg", -1.5, 1.5, 0.2, lambda x, val: scale(x, rand_val=False, scale_val=val), code)

test_transforms()




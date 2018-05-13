
from __future__ import print_function

import numpy as np
import random, sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import im, binary
from encoding import encode_binary

import IPython

# Returns encoding plus possible data visualizations/images
def encode(img, target=binary.parse("11100011101110001110111000111011"), verbose=False):
    return encode_binary(img, target=target, verbose=verbose, max_iter=200)

# Returns rounded bit pattern that is decoded, i.e. [0,1,0,1, ... ]
def decode(img):
    return binary.get(model(im.torch(img)))

# Returns decoded float bit pattern, i.e. [-0.1, 1.2, 0.34, ... ]
def decode_raw(img):
	return model(img).data.cpu().numpy()

def decode_with_loss(img):
    return model.loss(img)
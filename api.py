
from __future__ import print_function

import numpy as np
import random, sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import im, binary
from encoding import model, encode_binary

import IPython

# Returns encoding plus possible data visualizations/images
def encode(img, target=binary.parse("1110001110"), verbose=False):
    return encode_binary(img, target=target, verbose=verbose, max_iter=200)

def decode(img):
    return binary.get(model(im.torch(img)))

def decode_with_loss(img):
    return model.loss(img)
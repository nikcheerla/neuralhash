
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json, glob

import torch

from models import DecodingNet
from torchvision import models
from logger import Logger
from utils import *

import IPython

import transforms

from encoding import encode_binary

if __name__ == "__main__":
	
	def p(x):
		x = transforms.resize_rect(x)
		x = transforms.rotate(transforms.scale(x, 0.6, 1.4), max_angle=30)
		x = transforms.gauss(x, min_sigma=0.8, max_sigma=1.2)
		x = transforms.translate(x)
		x = transforms.identity(x)
		return x

	model = nn.DataParallel(DecodingNet(n=16, distribution=p))
	model.eval()

	images = [im.load(image) for image in glob.glob("data/colornet/*.jpg")[0:8]]
	images = im.stack(images)
	targets = [binary.random(n=TARGET_SIZE) for _ in range(0, len(images))]

	encode_binary(images, targets, model, verbose=True)


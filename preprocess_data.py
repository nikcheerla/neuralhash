
import numpy as np
import random, sys, os, json, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
import IPython


if __name__ == "__main__":

	path = "data/tiny-imagenet-200/train"
	files = glob.glob(f"{path}/*/images/*.JPEG")
	for image_file in files:
		image = im.load(image_file)
		if image is None: continue
		torch_path = image_file[:-5] + ".pth"
		torch.save(im.torch(image), torch_path)

		print (f"Saved {image_file} to {torch_path}")

from __future__ import print_function

import numpy as np
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models
from utils import *

from skimage import filters
from skimage.morphology import binary_dilation

import IPython

import transforms

def identity(x):
    x = transforms.resize(x, rand_val=False, resize_val=224)
    return x

"""Decoding network that tries to predict a
binary value of size target_size """
class DecodingNet(nn.Module):

    def __init__(self):
        super(DecodingNet, self).__init__()

        self.features = models.vgg11(pretrained=False)
        self.features.classifier = nn.Sequential(
            nn.Linear(25088, TARGET_SIZE))

        self.features.eval()

        if USE_CUDA: self.cuda()

    def forward(self, x, verbose=False, distribution=identity, 
                    n=1, return_variance=False):

        # make sure to center the image and divide by standard deviation
        x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0),
            ((x[1]-0.456)/(0.224)).unsqueeze(0),
            ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

        images = torch.cat([distribution(x).unsqueeze(0) for i in range(0, n)], dim=0)
        predictions = (self.features(images)) + 0.5
        if return_variance:
            return predictions.mean(dim=0), predictions.std(dim=0)
        return predictions.mean(dim=0)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


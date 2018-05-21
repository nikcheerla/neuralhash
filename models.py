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

class GramMatrix(nn.Module):
    def forward(self, input):
        N, C, H, W = input.size()

        features = input.view(N, C, H*W)  # resise F_XL into \hat F_XL

        G = torch.bmm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        print(G.size)
        return G.div(N * C * H * W)

"""Decoding network that tries to predict a
binary value of size target_size """
class DecodingNet(nn.Module):

    def __init__(self):
        super(DecodingNet, self).__init__()

        self.features = models.vgg11(pretrained=True)
        self.features.classifier = nn.Sequential(
            GramMatrix(),
            nn.Linear(262144, TARGET_SIZE)
        )
        # mask = Variable(torch.bernoulli(torch.ones(TARGET_SIZE, 25088)*0.1))/0.1
        # self.features.classifier.weight.data = self.features.classifier.weight.data*mask.data

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
        return predictions

    def predictions(self, x, verbose=False, distribution=identity, n=1):
        return self.forward(x, verbose, distribution, n).mean(dim=0)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


class DecodingGAN(nn.Module):
    def __init__(self):
        super(DecodingGAN, self).__init__()

        ngf, ndf, nc = 64, 64, 3
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.load_state_dict(torch.load("/model/netD_epoch_299.pth"))
        self.classifier = [nn.Linear(64*64, TARGET_SIZE)]

        if USE_CUDA: self.cuda()

    def forward(self, x):

        orig_shape = x.size()
        x = x.view(x.size(0), -1)
        x = (x - x.mean(dim=1))/x.std(dim=1)
        x = 0.5*x + 0.5
        x = x.view(*orig_shape)

        images = torch.cat([distribution(x).unsqueeze(0) for i in range(0, n)], dim=0)

        features = self.main(x).view(x.size(0), -1)
        predictions = self.classifier[0](x) + 0.5

        if return_variance:
            return predictions.mean(dim=0), predictions.std(dim=0)

        return predictions.mean(dim=0)


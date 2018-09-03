
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
import transforms

import IPython


class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.down_size:
            x = self.max_pool(x)
        return x


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super(UNet_up_block, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(
            prev_channel + input_channel, output_channel, 3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, prev_feature_map, x):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        # self.down_block6 = UNet_down_block(256, 512, True)

        self.mid_conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        # self.mid_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.mid_conv3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        # self.bn3 = torch.nn.BatchNorm2d(512)

        # self.up_block1 = UNet_up_block(256, 512, 256)
        self.up_block2 = UNet_up_block(128, 256, 128)
        self.up_block3 = UNet_up_block(64, 128, 64)
        self.up_block4 = UNet_up_block(32, 64, 32)
        self.up_block5 = UNet_up_block(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.BatchNorm2d(16)
        self.last_conv2 = nn.Conv2d(16, 3, 1, padding=0)
        self.relu = nn.ReLU()
        self.to(DEVICE)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        # self.x6 = self.down_block6(self.x5)

        self.x5 = self.relu(self.bn1(self.mid_conv1(self.x5)))
        # self.x6 = self.relu(self.bn2(self.mid_conv2(self.x6)))
        # self.x6 = self.relu(self.bn3(self.mid_conv3(self.x6)))

        # x = self.up_block1(self.x5, self.x6)
        x = self.up_block2(self.x4, self.x5)
        x = self.up_block3(self.x3, x)
        x = self.up_block4(self.x2, x)
        x = self.up_block5(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

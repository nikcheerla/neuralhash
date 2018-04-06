
from __future__ import print_function

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models
from utils import im, binary

from skimage import filters
from skimage.morphology import binary_dilation

import IPython

import transforms
#from transforms import rotate, scale, flip, resize, gauss, noise, resize_rect, translate

torch.manual_seed(1234)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.manual_seed(1234)
EPSILON = 3e-2
MIN_LOSS = 7e-2
BATCH_SIZE = 12

"""Decoding network that tries to predict a
binary value of size target_size """
class DecodingNet(nn.Module):

    def __init__(self, target_size=10):
        super(DecodingNet, self).__init__()
        self.features = models.vgg11(pretrained=True)

        self.features.classifier = nn.Sequential(
            nn.Linear(25088, target_size))

        self.fc = nn.Linear(1000, target_size)

        self.target_size=target_size
        self.features.eval()

        if USE_CUDA: self.cuda()

    def set_target(self, target):
        self.target=target

    def forward(self, x, verbose=False):

        # make sure to center the image and divide by standard deviation
        x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0), 
            ((x[1]-0.456)/(0.224)).unsqueeze(0), 
            ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

        # returns an image after a series of transformations
        def distribution(x):
            
            x = transforms.resize_rect(x)
            x = transforms.rotate(transforms.scale(x), max_angle=90)
            
            #if random.random() < 0.2: x = flip(x)
            x = transforms.resize(x, rand_val=False, resize_val=224)
            x = transforms.translate(x)
            #x = gauss(x, min_sigma=0.8, max_sigma=1.2)
            return x

        images = torch.cat([distribution(x).unsqueeze(0) for i in range(0, BATCH_SIZE)], dim=0)
        predictions = (self.features(images)) + 0.5
        return predictions.mean(dim=0)

    """ returns the accuracy loss as well as the predictions """
    def loss(self, x):
        predictions = self.forward(x)
        return F.mse_loss(predictions, binary.target(self.target)), predictions.cpu().data.numpy().round(2)

model = DecodingNet(target_size=32)

def encode_binary(image, target=binary.parse("1100100110"), max_iter=1, verbose=False):

    image = im.torch(image)
    perturbation_old = None
    
    if USE_CUDA: perturbation = nn.Parameter(torch.randn(image.size()).cuda()+0.0)
    else: perturbation = nn.Parameter(torch.randn(image.size())+0.0)

    model.set_target(target)
    opt = torch.optim.Adam([perturbation], lr=0.1)

    losses = []
    preds = []
    for i in range(0, max_iter):
        def closure():
            opt.zero_grad()

            # perform projected gradient descent by norm bounding the perturbation
            perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
            changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)
            
            loss, predictions = model.loss(changed_image)
            loss.backward()

            preds.append(predictions)
            losses.append(loss.cpu().data.numpy())

            return loss

        opt.step(closure)

        if i % 20 == 0:
            
            perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
            if (perturbation_old is None): perturbation_old = perturbation_zc

            changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

            if verbose:
                print ("Loss: ", np.mean(losses[-20:]))
                #print ("Predictions: ", np.round(np.mean(preds[-20:], axis=0), 2))
                #print ("Modified prediction: ", binary.str(binary.get(model(changed_image))))
                
                #save(image, file="images/image.jpg")
                im.save(im.numpy(perturbation), file="/output/perturbation.jpg")
                im.save(im.numpy(changed_image), file="/output/changed_image.jpg")

        smooth_loss = np.mean(losses[-20:])
        if smooth_loss <= MIN_LOSS:
            break

    perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
    changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

    im.save(im.numpy(perturbation), file="/output/perturbation.jpg")
    im.save(im.numpy(changed_image), file="/output/changed_image.jpg")
    
    if verbose:
        #print("pert max : ", perturbation_zc.data.cpu().numpy().max(), "\tmin: ", perturbation_zc.data.cpu().numpy().min())
        plt.plot(np.array(preds)); plt.savefig("/output/preds.jpg"); plt.cla()
        plt.plot(losses); plt.savefig("/output/loss.jpg"); plt.cla()
        #print ("Original predictions: ", binary.get(model(image)))
        #print ("Perturbation: ", binary.get(model(perturbation_zc)))
        print ("Modified prediction: ", binary.get(model(changed_image)))
        #print ("Final predictions: ", preds[-1])

    return im.numpy(changed_image)

def test():
    images = ["images/cat.jpg"]
    for image_file in images:
        image = im.load(image_file)
        if image is None: continue
        code = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1]
        if len(sys.argv) == 3:
            image = im.torch(im.load(sys.argv[2]))
            model.set_target(code)
            for i in range(10):
                mse_loss, preds = model.loss(image)
                mse_loss = mse_loss.data.cpu().numpy()[0]
                print("mse: ", np.round(mse_loss, 4))
                print("# of diff: ", binary.distance(code, list(preds.clip(min=0, max=1).round().astype(int))))
            sys.exi
        code = binary.random(n=32)
        encoded_img = encode_binary(image, target=code, verbose=True)

        res = []
        for i in np.arange(-1, 1, 0.1):
            rotated = rotate(im.torch(encoded_img), rand_val=False, theta=i)
            mse_loss, preds = model.loss(rotated)
            mse_loss = mse_loss.data.cpu().numpy()[0]
            print(i, " mse: ", np.round(mse_loss, 4))
            print("# of diff: ", binary.distance(code, list(preds.clip(min=0, max=1).round().astype(int))))
            res.append((i, mse_loss))
        thetas, losses = zip(*res)
        plt.plot(thetas, losses); plt.savefig("images/angle_robust.jpg"); plt.cla()
        print("")
        res = []
        for i in np.arange(0.5, 1.5, 0.05):
            scaled = scale(im.torch(encoded_img), rand_val=False, scale_val=i)
            mse_loss, preds = model.loss(scaled)
            mse_loss = mse_loss.data.cpu().numpy()[0]
            print(i, " mse: ", np.round(mse_loss, 4))
            print("# of diff: ", binary.distance(code, list(preds.clip(min=0, max=1).round().astype(int))))
            res.append((i, mse_loss))

        scales, losses = zip(*res)
        plt.plot(scales, losses); plt.savefig("images/scale_robust.jpg"); plt.cla()
        
        print("")
        res = []
        for i in np.arange(0, 0.3, 0.02):
            noised = noise(im.torch(encoded_img), max_noise_val=i)
            mse_loss, preds = model.loss(noised)
            mse_loss = mse_loss.data.cpu().numpy()[0]
            print(i, " mse: ", np.round(mse_loss, 4))
            print("# of diff: ", binary.distance(code, list(preds.clip(min=0, max=1).round().astype(int))))
            res.append((i, mse_loss))

        noises, losses = zip(*res)
        plt.plot(scales, losses); plt.savefig("images/noise_robust.jpg"); plt.cla()

        IPython.embed()

if __name__ == "__main__":
    target = binary.random(n=32)
    print("Target: ", binary.str(target))
    encode_binary(im.load("images/cat.jpg"), target=target, verbose=True)

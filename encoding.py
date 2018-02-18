
from __future__ import print_function

import numpy as np
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

import matplotlib.pyplot as plt
import IPython

from transforms import rotate, scale, flip, resize, gauss, noise, resize_rect, translate

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
USE_CUDA = torch.cuda.is_available()
EPSILON = 3e-2
MIN_LOSS = 7e-2
BATCH_SIZE = 64



"""Decoding network that tries to predict a
binary value of size target_size """
class DecodingNet(nn.Module):

    def __init__(self, target_size=10):
        super(DecodingNet, self).__init__()
        self.features = models.vgg11(pretrained=True)
        #self.features.fc = nn.Linear(2048, 128).cuda()
        self.features.classifier = nn.Sequential(
            #nn.Linear(25088, 4096),
            #nn.ReLU(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(),
            nn.Linear(25088, target_size))

        self.fc = nn.Linear(1000, target_size)

        self.target_size=target_size
        self.features.eval()

        if USE_CUDA: self.cuda()

    def set_target(self, target):
        self.target=target

    def forward(self, x, verbose=False):

        x = torch.cat([((x[0]-0.485)/(0.229)).unsqueeze(0), 
            ((x[1]-0.456)/(0.224)).unsqueeze(0), 
            ((x[2]-0.406)/(0.225)).unsqueeze(0)], dim=0)

        def distribution(x):

            
            x = resize_rect(x)
            x = rotate(scale(x), max_angle=90)
            
            #if random.random() < 0.2: x = flip(x)
            x = resize(x, rand_val=False, resize_val=224)
            x = translate(x)
            #x = gauss(x, min_sigma=0.8, max_sigma=1.2)
            return x

        #x = scale(x, min_val=1, max_val=1)
        images = torch.cat([distribution(x).unsqueeze(0) for i in range(0, BATCH_SIZE)], dim=0)
        predictions = (self.features(images)) + 0.5
        return predictions.mean(dim=0)

    def loss(self, x):
        predictions = self.forward(x)
        return F.mse_loss(predictions, binary.target(self.target)), predictions.cpu().data.numpy().round(2)

    # def predictions(self, x):
    #     if type(x) != Variable: x = Variable(tform(x).cuda())
    #     vals = self.forward(x).data.cpu().numpy().round(2).clip(min=0, max=1)
    #     return vals

    # def binary(self, x):
    #     if type(x) != Variable: x = Variable(tform(x).cuda())
    #     return "".join((str(int(round(val))) for val in self.predictions(x)))

model = DecodingNet(target_size=32)

def encode_binary(image, target=binary.parse("1100100110"), max_iter=1000, verbose=False):
    
    im.save(image, "static/partials.png")
    send = {}
    send["bits"] = list([0.5]*32)
    send["loss"] = -1

    with open("static/partial.json", 'wt') as outfile:
        json.dump(send, outfile)

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

            perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
            changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)
            
            acc_loss, predictions = model.loss(changed_image)
            loss = acc_loss

            loss.backward()

            preds.append(predictions)
            losses.append(acc_loss.cpu().data.numpy())

            return loss

        opt.step(closure)

        if i % 20 == 0:
            
            perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
            if (perturbation_old is None): perturbation_old = perturbation_zc

            changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

            if verbose:
                print ("Loss: ", np.mean(losses[-20:]))
                #print ("Predictions: ", np.round(np.mean(preds[-20:], axis=0), 2))
                print ("Modified prediction: ", binary.str(binary.get(model(changed_image))))
                
                #print("Gradient MSE: ", F.mse_loss(G_im, G_ch).data.cpu().numpy())
                #save(image, file="images/image.jpg")
                im.save(im.numpy(perturbation), file="images/perturbation.jpg")
                im.save(im.numpy(changed_image), file="images/changed_image.jpg")

        perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
        if (perturbation_old is None): perturbation_old = perturbation_zc
        diff_p = 4*perturbation_zc*perturbation_zc - 3.9*perturbation_old*perturbation_old
        diff_p = (diff_p.mean(dim=0) - diff_p.mean())/(diff_p.std())
        values = diff_p.data.cpu().numpy()
        mask = binary_dilation(binary_dilation(binary_dilation(binary_dilation(values > 4.9))))

        image_values = im.numpy(image)
        image_values[:, :, 0][mask] = 1.0
        image_values[:, :, 1][mask] = 0.0
        image_values[:, :, 2][mask] = 0.0

        image_values = filters.gaussian(image_values, sigma=2.8, truncate=5.0)
        im.save(image_values, "static/partial.jpg")

        perturbation_old = Variable(perturbation_zc.data)

        send = {}
        send["bits"] = list(preds[-1].astype(float))
        send["loss"] = float(losses[-1])

        smooth_loss = np.mean(losses[-20:])

        send["percent"] = int(100.0*MIN_LOSS/smooth_loss)

        if smooth_loss <= MIN_LOSS:
            break

        with open("static/partial.json", 'wt') as outfile:
            json.dump(send, outfile)

    perturbation_zc = (perturbation - perturbation.mean())/(perturbation.std()) * EPSILON
    changed_image = (image + perturbation_zc).clamp(min=0.1, max=0.99)

    im.save(im.numpy(perturbation), file="images/perturbation.jpg")
    im.save(im.numpy(changed_image), file="images/changed_image.jpg")
    
    if verbose:
        #print("pert max : ", perturbation_zc.data.cpu().numpy().max(), "\tmin: ", perturbation_zc.data.cpu().numpy().min())
        plt.plot(np.array(preds)); plt.savefig("images/preds.jpg"); plt.cla()
        plt.plot(losses); plt.savefig("images/loss.jpg"); plt.cla()
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

    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        test()
    else:
        target = binary.random(n=32)
        print("Target: ", binary.str(target))
        encode_binary(im.load("images/cat.jpg"), target=target, verbose=True)


import numpy as np
import random, sys, os, json, glob
import tqdm, itertools, shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
import transforms
from models import UNet
from logger import Logger

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import IPython

DATA_PATH = "data/encode_120"
logger = Logger("train_dsc", ("loss", "corr"), print_every=5, plot_every=20)


def loss_func(model, x, y):
    cleaned = model.forward(x)
    corr, p = pearsonr(
        cleaned.data.cpu().numpy().flatten(), y.data.cpu().numpy().flatten()
    )
    return (cleaned - y).pow(2).sum(), corr


def data_gen(files, batch_size=64):
    while True:
        enc_files = random.sample(files, batch_size)
        orig_files = [f.replace("encoded", "original") for f in enc_files]
        encoded_ims = [im.load(image) for image in enc_files]
        original_ims = [im.load(image) for image in orig_files]
        encoded, original = im.stack(encoded_ims), im.stack(original_ims)

        yield original, (encoded - original)


def viz_preds(model, x, y):
    preds = model(x)
    for i, (pred, truth, enc) in enumerate(zip(preds, y, x)):
        im.save(im.numpy(enc + truth), f"{OUTPUT_DIR}{i}_encoded.jpg")
        im.save(3 * np.abs(im.numpy(pred)), f"{OUTPUT_DIR}{i}_pred.jpg")
        im.save(3 * np.abs(im.numpy(truth)), f"{OUTPUT_DIR}{i}_truth.jpg")


if __name__ == "__main__":

    model = nn.DataParallel(UNet())
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # optimizer.load_state_dict('output/unet_opt.pth')
    model.module.load("output/train_unet.pth")

    logger.add_hook(
        lambda: [
            print(f"Saving model/opt to {OUTPUT_DIR}train_unet.pth"),
            model.module.save(OUTPUT_DIR + "train_unet.pth"),
            torch.save(optimizer.state_dict(), OUTPUT_DIR + "unet_opt.pth"),
        ],
        freq=100,
    )

    files = glob.glob(f"{DATA_PATH}/*encoded*.jpg")
    train_files, val_files = files[:-142], files[-142:]
    x_val, y_val = next(data_gen(val_files, 142))

    for i, (x, y) in enumerate(data_gen(train_files, 142)):
        loss, corr = loss_func(model, x, y)

        logger.step("loss", min(5000, loss))
        logger.step("corr", corr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            model.eval()
            val_loss = loss_func(model, x_val, y_val)
            viz_preds(model, x_val[:8], y_val[:8])
            model.train()
            print(f"val_loss = {val_loss}")

        if i == 2000:
            break

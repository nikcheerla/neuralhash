from __future__ import print_function
import IPython
IPython.embed()

import random, sys, os, glob

import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from utils import im, binary

from api import encode, decode
from transforms import rotate 

#images = glob.glob("test_data/*.jpg")
#random.shuffle(images)
TP, TP_zoomed, total = 0, 0, 0
images = ["images/cat.jpg"]

for image_file in images:
	IPython.embed()
	image = im.load(image_file)
	if image is None: continue

	code = binary.random(n=32)

	encoded_img = encode(image, binary.redundant(code), verbose=True)

	res = []
	for theta in range(np.radians(-60), np.radians(60), 0.02):
		preds, mse_loss = decode_with_loss(rotate(encoded_img, rand_val=False, theta=theta))
		res.append((theta, mse_loss))

	IPython.embed()
	plt.plot()
	total, TP = total + 1, TP + (1 if decoded == code else 0)

	print ("{0}/{1}, TPR={2:.2}, image={3}".format(code, decoded, TP/total, image_file))
	
	"""
	image_zoomed = np.array(transforms.RandomRotation(degrees=30) (Image.fromarray(image)))

	mse, protected = encode(image_zoomed, code_redundant)
	preds, binary = decode(protected)
	binary = [int(binary[i])+int(binary[i+10])+int(binary[i+20]) for i in range(0, 10)]
	print (binary)
	binary = "".join(['1' if b >= 2 else '0' for b in binary])

	TP_zoomed += 1 if binary == code else 0

	print ("Zoomed: {0}/{1}, TPRZ={2:.2}".format(code, binary, TP_zoomed/total))
	"""






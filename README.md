## Description:

Deep neural networks such as VGG16 and ResNet101 have been used to achieve state-of-the-art results in image classification. We combine these pretrained deep neural networks — whose parameters were expertly honed to detect salient shapes and features — along with stochastic connections and layers to develop a “decoder” model. This decoder acts as a robust, cryptographically-secure, transformation-invariant hash function for images, mapping input images to 32-bit codes.

By performing projective gradient descent on the decoder model with respect to a given image, we can use it to “sign” images robustly. We start with the original image, then repeatedly tweak the pixel values such that the image (and all transformations, including scaling, rotation, adding noise, blurring, random cropping, and more) decodes to a specified 32-bit code. The resultant image will be almost imperceptible from the original image, yet contain an easily-decodable signature that cannot be removed even by the most dedicated of adversaries.

We apply this to the problem of giving creators a unique way to provide proof-of-authenticity for their work. In the website we developed, we enable users to embed Facebook profile IDs in images, and decode images to find the Facebook IDs of the people who authored them.

## Problem Formulation

Our encoder model is designed to extend images that are robust to scaling, rotation, translation, gauss, noise, random cropping, and other affine or differentiable transformations. To that end, we wish to accurately extract embedded codes not only from the input image itself, but across the space of all possible transformations. To that end, we can use the Expectation-Over-Transformations (EOT) model to define the loss, with an additional penalty on the gradient to discourage sharp edges and changes.


## Robustness to transformations:

Below are plots for scaling, rotation, and noise. These show the changes in mean squared error (MSE) by varying the magnitude and direction of the transformation values.

<img src="https://github.com/nikcheerla/neuralhash/blob/master/images/Scaling.jpg" height="315"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/Rotation.jpg" height="315"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/Noise.jpg" height="315">

## Robust transformation examples:

The images on the left are the original images and the images on the right are perturbed and transformed images.

<img src="https://github.com/nikcheerla/neuralhash/blob/master/images/cat.jpg" height="400"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/cat-photo-transformation.jpg" height="400">

<img src="https://github.com/nikcheerla/neuralhash/blob/master/images/meme.jpg" height="200"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/meme-photo-transformation.png" height="200">

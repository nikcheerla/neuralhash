# NeuralHash: An Adversarial Steganographic Method For Robust, Imperceptible Watermarking
TreeHacks 2018: Nikhil Cheerla, Rohan Suri, Isaac Pohl-Zaretsky, Evani Radiya-Dixit

## What it does:
Given an image (like Scream):

<img src="https://raw.githubusercontent.com/nikcheerla/neuralhash/cb20c8b848fc85bd6be55785c0acd0ea1f64e5fb/images/Scream.jpg" height="400">

Neuralhash makes small perturbations to visually encode data (in this case, a facebook ID):

<img src="https://raw.githubusercontent.com/nikcheerla/neuralhash/cb20c8b848fc85bd6be55785c0acd0ea1f64e5fb/images/Scream%20Encoded.jpeg" height="400">

Which is able to be decoded even after extreme transformations (like a cellphone photo of the encoded image):

<img src="https://github.com/nikcheerla/neuralhash/blob/214803c0b805d10b87b611316fd8818e42f90ebc/images/neuralhash-result.png" height="400">


## Description:

The development of a secure watermarking scheme is an important problem that has applications in content owner- ship and piracy prevention. Current state-of-the-art techniques are unable to document robustness across a vari- ety of affine transformations. We propose a method that harnesses the expressiveness of deep neural networks to covertly embed imperceptible, transformation-resilient binary signatures into images. Given a decoder network, our key insight is that adversarial example generation techniques can be used to encode images by performing pro- jected gradient descent on the image to embed a chosen signature.

By performing projective gradient descent on the decoder model with respect to a given image, we can use it to “sign” images robustly (think of a more advanced watermark). We start with the original image, then repeatedly tweak the pixel values such that the image (and all transformations, including scaling, rotation, adding noise, blurring, random cropping, and more) decodes to a specified 32-bit code. The resultant image will be almost imperceptible from the original image, yet contain an easily-decodable signature that cannot be removed even by the most dedicated of adversaries.

We also propose a method to train our decoder network under the Expectation-Maximization (EM) framework to learn feature transformations that are more resilient to the threat space of attacks. Experimental results indicate that our model achieves robustness across different transformations such as scaling and rotating, with improved results over the length of EM training. Furthermore, we show an inherent trade-off between robustness and imperceptibility, which allows the user of the model flexibility in adjusting parameters to fit a particular task.

## Problem Formulation

Our encoder model is designed to extend images that are robust to scaling, rotation, translation, gauss, noise, random cropping, and other affine or differentiable transformations. To that end, we wish to accurately extract embedded codes not only from the input image itself, but across the space of all possible transformations. To that end, we can use the Expectation-Over-Transformations (EOT) model to define the loss, with an additional penalty on the gradient to discourage sharp edges and changes.

## Related Work

Traditionally, watermarks have been used to prevent piracy and protect ownership of media. Inspired by the widespread utility of this technique, we propose a “next-gen” watermarking tool.

Current DRM (digital rights management) solutions to this problem mostly use an active search approach. For example, Youtube’s Content ID can detect pirated media by comparing to databases of preregistered content, but is computationally intensive and has been documented to fail on simple transforms like speeding up, reflections, and adding visual noise or bounding boxes. Even when effective, active search is inherently opaque and is almost never accessible to smaller content creators (such as artists selling their work online).

We argue that an effective open-source watermarking scheme would be both more accurate and simple enough to be used by anyone. However, the field of deep learning watermarking has not been developed yet for good reason: it’s extremely difficult to develop efficient encodings that are both imperceptible and robust/uneditable. Research in DL-based steganography has traditionally focused on imperceptibility without involving robustness, while research in “blind watermarking” has tended to tackle the development of robust, yet with watermarks. 

Previous research has explored the usage of many different architecture types and problem framings. For example, [Hiding Images in Plain Sight: Deep Steganography](https://research.google.com/pubs/pub46526.html) explores the use of a GAN model to encode and detect steganographically signed images. Their transforms are undetectable (not only to humans, but even to a determined AI observer) and carry many bits of information, but they can be easily fooled by small modifications to the image. Similarly, blind watermarking techniques such as [1](https://www.sciencedirect.com/science/article/pii/S0167404816301699), [2](https://arxiv.org/abs/1703.05502) maintain robustness in the face of noise transformations, but not spatial warping attacks or generalized affine transforms.

[A Robust Blind Watermarking Using Convolutional Neural Networks](https://arxiv.org/abs/1704.03248) offers the most comprehensive approach to transformation invariance, achieving robustness for attacks such as cropping and scaling. Their approach is to break up the image into sub-blocks, allowing for a 1-bit message for each block. However, because of this technique, the model is not robust to rotation without the use of registration (realigning corners).


## Robustness to transformations:

Below are plots for scaling, rotation, and noise. These show the changes in mean squared error (MSE) by varying the magnitude and direction of the transformation values.

<img src="https://github.com/nikcheerla/neuralhash/blob/master/images/Scaling.jpg" height="315"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/Rotation.jpg" height="315"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/Noise.jpg" height="315">

## Further examples:

The images on the left are the original images and the images on the right are perturbed and transformed images.

<img src="https://github.com/nikcheerla/neuralhash/blob/master/images/cat.jpg" height="400"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/cat-photo-transformation.jpg" height="400">

<img src="https://github.com/nikcheerla/neuralhash/blob/master/images/meme.jpg" height="200"> <img src="https://github.com/nikcheerla/neuralhash/blob/master/images/meme-photo-transformation.png" height="200">

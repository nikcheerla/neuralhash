# NeuralHash: An Adversarial Steganographic Method For Robust, Imperceptible Watermarking
Building the next-gen watermark with deep learning: imperceptibly encoding images with un-erasable patterns to verify content ownership.

## What it does:
Given an image (like Scream), Neuralhash makes small perturbations to visually encode a unique signature of the author:

<img width="1011" alt="original_to_watermarked" src="https://user-images.githubusercontent.com/10892180/43042515-fa9123c2-8d34-11e8-98d2-b65e05c18ead.png">

Which is able to be decoded even after extreme transformations (like a cellphone photo of the encoded image):

<p align="center">
  <img src="https://github.com/nikcheerla/neuralhash/blob/214803c0b805d10b87b611316fd8818e42f90ebc/images/neuralhash-result.png" height="300">
</p>

Our secure watermarking scheme represents significant advances in protecting content ownership and piracy prevention on the Internet.
## Harnessing Adversarial Examples

Our key insight is that we can use adversarial example techniques on a Decoder Network (that maps input images to 32-bit signatures) to generate perturbations that decode to the desired signature. We perform projected gradient descent under the Expectation over Transformation framework to do this as follows:
<p align="center">
  <img width="700" src="https://user-images.githubusercontent.com/10892180/43042615-c0189c40-8d37-11e8-8ff5-e71d3e33b3ef.png">
</p>
We simulate an attack distrubtion using a set of differentiable transformations over which we train over. Here are some sample transforms:
<p align="center">
  <img src="https://user-images.githubusercontent.com/10892180/43042616-d490cf8a-8d37-11e8-876d-c2e2382600ab.png" width="400">
</p>

## Training the Network
We also propose a method to train our decoder network under the Expectation-Maximization (EM) framework to learn feature transformations that are more resilient to the threat space of attacks. As shown below, we alternate between encoding images using the network and then updating the network's weights to be more robust to attacks.
<p align="center">
<img src="https://user-images.githubusercontent.com/10892180/43042623-ef164dc6-8d37-11e8-86be-f54780e7f9df.png" width="400">
</p>

The below plots show robustness of our encoded images during the training process. As you can see, over many iterations, the line becomes flatter, indicating robustness over rotation and scaling. Shown later, our approach generalizes to more extreme transformations.
<p align="center">
<img src="https://user-images.githubusercontent.com/10892180/43042617-de13f64a-8d37-11e8-9ed0-91673390f20d.png" width="300"> <img src="https://user-images.githubusercontent.com/10892180/43042618-df3f1e1e-8d37-11e8-8532-ba1437674b45.png" width="300">
</p>

## Sample Encodings
Here are some sample original images (top row) and the corresponding watermarked image (bottom row):

<img src="https://user-images.githubusercontent.com/10892180/43042620-e5f128ec-8d37-11e8-8c30-a88812614c01.png" width="1100">

## Example Attacks
Some examples where our approach succeessfully decodes the correct signature and examples where it fails:

<img src="https://user-images.githubusercontent.com/10892180/43042624-f389c4d2-8d37-11e8-9820-c62c0363ba4b.png" width="1100">

## Final Thoughts:

The development of a secure watermarking scheme is an important problem that has applications in content ownership and piracy prevention. Current state-of-the-art techniques are unable to document robustness across a variety of affine transformations. We propose a method that harnesses the expressiveness of deep neural networks to covertly embed imperceptible, transformation-resilient binary signatures into images. Given a decoder network, our key insight is that adversarial example generation techniques can be used to encode images by performing projected gradient descent on the image to embed a chosen signature.

By performing projective gradient descent on the decoder model with respect to a given image, we can use it to “sign” images robustly (think of a more advanced watermark). We start with the original image, then repeatedly tweak the pixel values such that the image (and all transformations, including scaling, rotation, adding noise, blurring, random cropping, and more) decodes to a specified 32-bit code. The resultant image will be almost imperceptible from the original image, yet contain an easily-decodable signature that cannot be removed even by the most dedicated of adversaries.

We also propose a method to train our decoder network under the Expectation-Maximization (EM) framework to learn feature transformations that are more resilient to the threat space of attacks. Experimental results indicate that our model achieves robustness across different transformations such as scaling and rotating, with improved results over the length of EM training. Furthermore, we show an inherent trade-off between robustness and imperceptibility, which allows the user of the model flexibility in adjusting parameters to fit a particular task.

Paper and more details coming soon.

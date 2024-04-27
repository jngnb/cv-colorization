# From https://arxiv.org/abs/2204.02980
# Analysis of Different Losses for Deep Learning Image Colorization

# ... To that goal, we review the different losses and evaluation metrics 
# that are used in the literature. We then train a baseline network with 
# several of the reviewed objective functions: classic L1 and L2 losses, 
# as well as more complex combinations such as Wasserstein GAN and VGG-based LPIPS loss. 
# Quantitative results show that the models trained with VGG-based LPIPS provide overall 
# slightly better results for most evaluation metrics. Qualitative results exhibit 
# more vivid colors when with Wasserstein GAN plus the L2 loss or again with the VGG-based LPIPS. 
# Finally, the convenience of quantitative user studies is also discussed to 
# overcome the difficulty of properly assessing on colorized images, 
# notably for the case of old archive photographs where no ground truth is available.

# Note that to compute the VGG-based LPIPS loss, the output
# colorization always has to be converted to RGB (in a differentiable way), even for
# Lab color space, because this loss is computed with a pre-trained VGG expecting
# RGB images as input. To this end, we have used the Kornia implementation of
# differentiable color space conversions


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import kornia
import os
import cv2
import sys
import pdbp

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)

from PerceptualSimilarity.lpips import *


# Define a function to convert LAB images to RGB using Kornia

def lab_to_rgb(lab_images):
    rgb_images = kornia.color.lab_to_rgb(lab_images)
    return rgb_images

# Load pre-trained VGG network for feature extraction
# vgg_net = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
# vgg_net.eval()

loss_fn = LPIPS(net='vgg', verbose=True)

def lpips_loss(img1_lab, img2_lab):
    # Define LPIPS loss function
    img1_rgb = lab_to_rgb(img1_lab)
    img2_rgb = lab_to_rgb(img2_lab)
    lpips_values = loss_fn.forward(img1_rgb, img2_rgb, normalize=True)

    lpips_value =torch.mean(lpips_values)

    # for i, loss in enumerate(lpips_value):

    #     print(f"LPIPS Loss for image {i}:", loss)

    return lpips_value


# for testing

# img1_rgb = cv2.imread('imgs/ansel_adams.jpg')
# img1_rgb = cv2.resize(img1_rgb, (64, 64), interpolation=cv2.INTER_AREA)
# img1_rgb = torch.tensor(np.array([img1_rgb]))
# img1_rgb = img1_rgb.permute(0, 3, 1, 2)
# img1_rgb = img1_rgb / 255.0
# print(img1_rgb)

# img2_rgb = cv2.imread('imgs_out/ansel_adams_colorized_eccv16.png')
# img2_rgb = cv2.resize(img2_rgb, (64, 64), interpolation=cv2.INTER_AREA)
# img2_rgb = torch.tensor(np.array([img2_rgb]))
# img2_rgb = img2_rgb.permute(0, 3, 1, 2)
# img2_rgb = img2_rgb / 255.0
# print(img2_rgb)





import os
from PIL import Image
#import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

images_gray = np.load('raw_data/gray_scale.npy')
image_gray = images_gray[0]
print(image_gray)
images_ab1 = np.load('raw_data/ab1.npy')
images_ab2 = np.load('raw_data/ab2.npy')
images_ab3 = np.load('raw_data/ab3.npy')

gray_array = []
ab_array = []
for images_ab in [images_ab1, images_ab2, images_ab3]:
    for i in range(len(images_ab)):
        image_gray = images_gray[i]
        image_ab = images_ab[i]

        img = np.zeros((224, 224, 3))
        img[:, :, 0] = image_gray
        img[:, :, 1:] = image_ab
        
        angle = np.random.randint(0, 360)
        
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (width, height))

        size = (64, 64)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        
        gray_array.append(img[:, :, 0])
        ab_array.append(img[:, :, 1:])

    for i in range(len(images_ab)):
        image_gray = images_gray[i]
        image_ab = images_ab[i]

        img = np.zeros((224, 224, 3))
        img[:, :, 0] = image_gray
        img[:, :, 1:] = image_ab

        size = (64, 64)   
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        gray_array.append(img[:, :, 0])
        ab_array.append(img[:, :, 1:])


gray_array = np.array(gray_array)
ab_array = np.array(ab_array)
np.save(f'data/grayscale_downsample_aug/grayscale.npy', gray_array)
np.save(f'data/true_color_downsample_aug/color.npy', ab_array)

import os
from PIL import Image
#import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
# from .load_data import *
# Set the directory where your images are located
l_directory = 'kaggle_images/l/gray_scale.npy'
# Set the directory where you want to save the downsampled images
ab_directory = 'kaggle_images/ab/ab/ab'

# Create the output directory if it does not exist
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# Set the desired size
# size = (64, 64)

# images_l = np.load(l_directory)
# images_ab = []

# for i in range(3):
#     images_ab.append(np.load(ab_directory + str(i+1) + '.npy'))

# print(ab_directory)
# print(images_l.shape, len(images_ab))

# downsampled_l = np.zeros((25000, 64, 64))

# for i in range(images_l.shape[0]):
#     downsampled_l[0] = cv2.resize(images_l[i], size, interpolation=cv2.INTER_AREA)
    
# downsampled_ab = np.zeros((25000, 64, 64, 2))

# for i in range(3):
#     for j in range(images_ab[i].shape[0]):
#         downsampled_ab[i*10000 + j][:, :, 0] = cv2.resize(images_ab[i][j][:, :, 0], size, interpolation=cv2.INTER_AREA)
#         downsampled_ab[i*10000 + j][:, :, 1] = cv2.resize(images_ab[i][j][:, :, 1], size, interpolation=cv2.INTER_AREA)

# # os.makedirs('kaggle_images/l_downsampled')
# # os.makedirs('kaggle_images/ab_downsampled')
# np.save('kaggle_images/l_downsampled/gray_scale.npy', downsampled_l)

# np.save('kaggle_images/ab_downsampled/ab.npy', downsampled_ab)

images_gray = np.load('kaggle_images/l/gray_scale.npy')
image_gray = images_gray[0]
print(image_gray)
images_ab = np.load('kaggle_images/ab/ab/ab1.npy')

gray_array = []
ab_array = []
for i in range(1000):
    image_gray = images_gray[i]
    image_ab = images_ab[i]

    img = np.zeros((224, 224, 3))
    img[:, :, 0] = image_gray
    img[:, :, 1:] = image_ab

    size = (64, 64)   
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # image_gray = cv2.resize(image_gray, size, interpolation=cv2.INTER_AREA)

    # img = img.astype('uint8')

    # img_ = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    gray_array.append(img[:, :, 0])
    ab_array.append(img[:, :, 1:])
    # print(f'gray_array: {min(image_gray), max(image_gray)}')
    # print(f'ab_array: {min(image_ab), max(image_ab)}')
    # plt.imshow(img[:, :, 0], cmap='gray')
    # plt.show()

    # img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    # print(img[0, :, 1:])
    # rgb_image = lab2rgb(img)
    # plt.imshow(img)
    # plt.show()


# print('Gray scale image')
gray_array = np.array(gray_array)
ab_array = np.array(ab_array)
np.save(f'data/grayscale_downsampled/grayscale', gray_array)
np.save(f'data/true_color_downsampled/color', ab_array)

# print('Recreated image')
# plt.imshow(img_)
# plt.show()

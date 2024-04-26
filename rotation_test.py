import numpy as np
import matplotlib.pyplot as plt
import cv2

grays = np.load('data/grayscale_rotated/grayscale.npy')
colors = np.load('data/true_color_rotated/color.npy')

while 1:
    idx = int(input("Index (-1 to quit): "))
    if idx < 0:
        break
    img = np.zeros((64, 64, 3))
    img[:, :, 0] = grays[idx]
    img[:, :, 1:] = colors[idx]
    img = img.astype(np.uint8)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.imshow(rgb_image)
    plt.show()

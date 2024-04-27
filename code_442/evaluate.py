import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import color
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss
import random
from colorizers.util import *
from skimage.color import lab2rgb
import pdbp


def calculate_psnr(img1, img2):
    mse = mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def evaluate_model_and_save_random_images(model, dataloader, device, output_folder, num_images=15):
    model.eval()
    total_mse = 0
    total_ssim = 0
    total_psnr = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine indices for random samples
    num_batches = len(dataloader)
    sample_indices = random.sample(range(num_batches), num_images)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print("Shape of outputs:", outputs.shape)
            # print("Min and max of outputs:", outputs.min(), outputs.max())
            mse = mse_loss(outputs, labels)
            total_mse += mse.item()

            outputs_np = outputs.cpu().numpy()[0]
            labels_np = labels.cpu().numpy()[0]
            outputs_np = outputs_np.transpose(1, 2, 0)
            labels_np = labels_np.transpose(1, 2, 0)

            # batch_ssim = np.mean([ssim(out, lab, multichannel=True)
            #                      for out, lab in zip(outputs_np, labels_np)])
            batch_ssim = ssim(outputs_np, labels_np, channel_axis=2, multichannel=True, win_size=5, data_range=1)
            total_ssim += batch_ssim

            batch_psnr = np.mean([calculate_psnr(torch.tensor(out), torch.tensor(
                lab)) for out, lab in zip(outputs_np, labels_np)])
            total_psnr += batch_psnr

            # breakpoint()


            # Save images if in the random sample
            if i in sample_indices:
                # Convert model's output to RGB and save the image
                # breakpoint()

                inputs = (inputs * 100)
                outputs = (outputs * 255) - 128
                out_rgb = postprocess_tens(inputs,outputs)
                plt.imsave(os.path.join(output_folder, f'output_{i}.png'), out_rgb)

                # Save corresponding ground truth image
                labels = (labels * 255) - 128
                gt_rgb = postprocess_tens(inputs, labels)
                
                print(f'gt_rgb: {gt_rgb.shape}')
                # print(f'gt_rgb: {gt_rgb}')
                print(f'gt_rgb c1: {gt_rgb[30,:,0]}')
                print(f'gt_rgb c2: {gt_rgb[30,:,1]}')
                print(f'gt_rgb c3: {gt_rgb[30,:,2]}')
                plt.imsave(os.path.join(output_folder,
                           f'ground_truth_{i}.png'), gt_rgb)
                
                ##should be between -128 and 127
                print(labels_np.min(), labels_np.max())
                
                ##should be between 0 and 1
                print(out_rgb.min(), out_rgb.max())

    avg_mse = total_mse / num_batches
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    return avg_mse, avg_ssim, avg_psnr

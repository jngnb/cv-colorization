import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import color
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss
import random


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
            mse = mse_loss(outputs, labels)
            total_mse += mse.item()

            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            batch_ssim = np.mean([ssim(out, lab, multichannel=True)
                                 for out, lab in zip(outputs_np, labels_np)])
            total_ssim += batch_ssim

            batch_psnr = np.mean([calculate_psnr(torch.tensor(out), torch.tensor(
                lab)) for out, lab in zip(outputs_np, labels_np)])
            total_psnr += batch_psnr

            # Save images if in the random sample
            if i in sample_indices:
                # Convert model's output to RGB and save the image
                out_lab = np.concatenate((inputs.cpu().numpy().squeeze(
                    1) * 100 + 50, outputs_np.squeeze(0) * 128), axis=0)
                # Reorder dimensions to WxHxC
                out_lab = out_lab.transpose(1, 2, 0)
                out_rgb = color.lab2rgb(out_lab)
                plt.imsave(os.path.join(output_folder,
                           f'output_{i}.png'), out_rgb)

                # Save corresponding ground truth image
                gt_lab = np.concatenate((inputs.cpu().numpy().squeeze(
                    1) * 100 + 50, labels_np.squeeze(0) * 128), axis=0)
                gt_lab = gt_lab.transpose(1, 2, 0)
                gt_rgb = color.lab2rgb(gt_lab)
                plt.imsave(os.path.join(output_folder,
                           f'ground_truth_{i}.png'), gt_rgb)

    avg_mse = total_mse / num_batches
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    return avg_mse, avg_ssim, avg_psnr

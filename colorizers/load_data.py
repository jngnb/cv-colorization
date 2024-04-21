import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LabDataset(Dataset):
    def __init__(self, l_images, ab_images):
        """
        Args:
            l_images (numpy array): Array of L channel images with shape (n_images, 64, 64).
            ab_images (numpy array): Array of ab channel images with shape (n_images, 64, 64, 2).
        """
        self.l_images = l_images
        self.ab_images = ab_images

    def __len__(self):
        return len(self.l_images)

    def __getitem__(self, idx):
        # Get L and ab images
        img_l = self.l_images[idx]
        img_ab = self.ab_images[idx]

        # Normalize the L and ab channels
        img_l = (img_l - 50.) / 100.  # Assuming L in [0, 100]
        img_ab = img_ab / 128.  # Assuming ab in [-128, 128]

        # Convert to PyTorch tensors
        img_l = torch.from_numpy(img_l).float().unsqueeze(0)  # Shape: (1, 64, 64)
        img_ab = torch.from_numpy(img_ab).float().permute(2, 0, 1)  # Shape: (2, 64, 64)

        return img_l, img_ab

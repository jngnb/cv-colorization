
import torch
import torch.nn as nn
import numpy as np
from IPython import embed
import torch.optim as optim
from torch.utils.data import DataLoader
from .base_color import *
import torch.nn.functional as F


class Simple(BaseColor):

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Upsampling layers to restore the spatial resolution
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        
        # Final convolution to output the 2-channel ab image
        self.final_conv = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Input x is a grayscale image: 1 x 64 x 64
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.deconv1(x)))
        x = F.relu(self.bn6(self.deconv2(x)))
        x = F.relu(self.bn7(self.deconv3(x)))
        x = self.final_conv(x)
        
        # Output x is the colorized image in ab channels: 2 x 64 x 64
        return x


def save_model_state_dict(model, path="model/model_state_dict.pth"):
    torch.save(model.state_dict(), path)


def simple():
    model = Simple()
    return model


def train_model_simple(model, dataloader, epochs=10, learning_rate=0.001):
    """
    Trains the ECCVGenerator model.

    Parameters:
    - model: The ECCVGenerator neural network.
    - dataloader: DataLoader providing the dataset.
    - epochs: Integer, the number of epochs to train the model.
    - learning_rate: Float, the learning rate for the optimizer.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("begin training simple")
    # Iterate over the number of epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # Assuming data contains input images and their corresponding lab color images
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass to get output
            outputs = model(inputs)

            # Calculate loss
            # breakpoint()
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(
                    f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} completed')

    save_model_state_dict(model)
    print('Finished Training')
    return model

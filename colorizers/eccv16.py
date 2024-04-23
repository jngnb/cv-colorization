
import torch
import torch.nn as nn
import numpy as np
from IPython import embed
import torch.optim as optim
from torch.utils.data import DataLoader
from .base_color import *

class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Initial input size: [batch_size, 1, 64, 64]
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Maintains size: [batch_size, 64, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Downsampling to [batch_size, 64, 32, 32]
            nn.ReLU(True),
            norm_layer(64)
        )
        
        # Instead of further downsampling, maintain the dimension and increase depth
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Maintains size: [batch_size, 128, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Maintains size: [batch_size, 128, 32, 32]
            nn.ReLU(True),
            norm_layer(128)
        )
        
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Maintains size: [batch_size, 256, 32, 32]
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Maintains size: [batch_size, 256, 32, 32]
            nn.ReLU(True),
            norm_layer(256)
        )
        
        # Upsampling layers to go back to the original dimension
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsampling to [batch_size, 128, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Adjusting channel size, maintaining dimension: [batch_size, 64, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)  # Output the 2 channel image: [batch_size, 2, 64, 64]
        )

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.upsample(x)
        return x

def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model

def save_model_state_dict(model, path="model/model_state_dict.pth"):
    torch.save(model.state_dict(), path)


def train_model(model, dataloader, epochs=10, learning_rate=0.001):
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
    criterion = torch.nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("begin training")
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
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} completed')

    save_model_state_dict(model)
    print('Finished Training')
    return model



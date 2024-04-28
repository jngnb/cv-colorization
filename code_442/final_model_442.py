
import torch
import torch.nn as nn
import numpy as np
from IPython import embed
import torch.optim as optim
from .lpips_loss_fn import lpips_loss
from colorizers.base_color import *

class ModelColorizer442(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Layer Model 1: Input is grayscale images of size [batch_size, 1, 64, 64]
        self.model1 = nn.Sequential(
            # Output: [batch_size, 64, 64, 64]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Output: [batch_size, 64, 32, 32]
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            norm_layer(64)
        )

        # Layer Model 2: Increases channel depth while maintaining dimension
        self.model2 = nn.Sequential(
            # Output: [batch_size, 128, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Output: [batch_size, 128, 32, 32]
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            norm_layer(128)
        )

        # Layer Model 3: Further increases channel depth
        self.model3 = nn.Sequential(
            # Output: [batch_size, 256, 32, 32]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Output: [batch_size, 256, 32, 32]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            norm_layer(256)
        )

        # Layer Model 4: Maintains the high channel depth
        self.model4 = nn.Sequential(
            # Output: [batch_size, 512, 32, 32]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            norm_layer(512),
            # Output: [batch_size, 512, 32, 32]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            norm_layer(512)
        )

        # Layer Model 5: Continues processing at the same resolution and channel depth
        self.model5 = nn.Sequential(
            # Output: [batch_size, 512, 32, 32]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            norm_layer(512),
            # Output: [batch_size, 512, 32, 32]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            norm_layer(512)
        )
        self.softmax = nn.Softmax(dim=1)

        # Upsampling layers to restore the original image size with a final channel depth for colorization
        self.upsample = nn.Sequential(
            # Output: [batch_size, 256, 64, 64]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # Output: [batch_size, 128, 64, 64]
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Output: [batch_size, 64, 64, 64]
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Output: [batch_size, 2, 64, 64]
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.upsample(self.softmax(x))
        return self.unnormalize_ab(x)

def modelColorizer442(pretrained=True):
	model = ModelColorizer442()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model

def save_model_state_dict(model, path="model/model_state_dict.pth"):
    torch.save(model.state_dict(), path)


# FROM eccv.py
def train_model_442(model, dataloader, epochs=10, learning_rate=0.001):
    """
    Trains the final 442 model.

    Parameters:
    - model: The final 442 neural network
    - dataloader: DataLoader providing the dataset.
    - epochs: Integer, the number of epochs to train the model.
    - learning_rate: Float, the learning rate for the optimizer.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define the loss function and optimizer
    # criterion = torch.nn.L1Loss()

    # CUSTOM LOSS FUNCTION FOR 442 PROJECT
    criterion = lpips_loss

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
            batches = outputs.shape[0]
            outputs = torch.cat((outputs, torch.zeros(batches, 1, 64, 64)), dim=1)
            labels = torch.cat((labels, torch.zeros(batches, 1, 64, 64)), dim=1)

            loss = criterion(outputs, labels)  
            print(loss)
            
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



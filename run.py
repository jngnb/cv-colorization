import argparse
import matplotlib.pyplot as plt
from code_442 import *
import os
import pdbp
from code_442 import final_model_442

from code_442.data_processing.load_data import LabDataset
from code_442.data_processing.data_split import data_split
from code_442.evaluate import evaluate_model_and_save_random_images


parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

L_train, L_test, AB_train, AB_test = data_split()

# print(f'L_train 30th row is {L_train[0, 30, :]}')
# print(f'L_test 30th row is {L_test[0, 30, :]}')
# print(f'AB_train 30th row is {AB_train[0, 30, :, 0]}')
# print(f'AB_test 30th row is {AB_test[0, 30, :, 0]}')

train_dataset = LabDataset(L_train, AB_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# load colorizers
colorizer_442 = final_model_442.modelColorizer442(pretrained=False)
if opt.use_gpu:
	colorizer_442.cuda()
	
# Train the model
final_model_442.train_model_442(colorizer_442, train_dataloader, 10)

colorizer_442.load_state_dict(torch.load("model/model_state_dict.pth"))
colorizer_442.eval()


# Ensure the output directory exists
output_dir = "imgs_out/test/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# breakpoint()
test_dataset = LabDataset(L_test, AB_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  

      

# Ensure model is in evaluation mode and on the correct device
device = "cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu"
colorizer_442.to(device)
colorizer_442.eval()


avg_mse, avg_ssim, avg_psnr = evaluate_model_and_save_random_images(colorizer_442, test_dataloader, device, output_dir)
print(f"Average MSE: {avg_mse}, Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}")
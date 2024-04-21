import argparse
import matplotlib.pyplot as plt
from colorizers import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

L_train, L_test, AB_train, AB_test = dataSPLIT()

dataset = LabDataset(L_train, AB_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# load colorizers
colorizer_eccv16 = eccv16(pretrained=False)
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	
# Train the model
colorizer_eccv16.train(colorizer_eccv16, dataloader, epochs=10)


# Ensure the output directory exists
output_dir = "../imgs_out/test/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

test_dataset = LabDataset(L_test, AB_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  

# Ensure model is in evaluation mode and on the correct device
device = "cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu"
colorizer_eccv16.to(device)
colorizer_eccv16.eval()

avg_mse, avg_ssim, avg_psnr = evaluate_model_and_save_random_images(colorizer_eccv16, test_loader, device, output_dir)
print(f"Average MSE: {avg_mse}, Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}")
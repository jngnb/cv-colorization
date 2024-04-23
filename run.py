import argparse
import matplotlib.pyplot as plt
from colorizers import *
import os
import pdbp


parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

L_train, L_test, AB_train, AB_test = dataSPLIT.dataSplit()
print('TYPEEEEEE ABABABABAB', type(AB_train[0,0,0,0]))
print('TYPEEEEEE LLLLLL', type(L_train[0,0,0]))


print(f'L_train 30th row is {L_train[0, 30, :]}')
print(f'L_test 30th row is {L_test[0, 30, :]}')
print(f'AB_train 30th row is {AB_train[0, 30, :, 0]}')
print(f'AB_test 30th row is {AB_test[0, 30, :, 0]}')


train_dataset = LabDataset(L_train, AB_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# load colorizers
colorizer_eccv16 = eccv16(pretrained=False)
if opt.use_gpu:
	colorizer_eccv16.cuda()
	
# Train the model
train_model(colorizer_eccv16, train_dataloader, 10)

colorizer_eccv16.load_state_dict(torch.load("model/model_state_dict.pth"))
colorizer_eccv16.eval()


# Ensure the output directory exists
output_dir = "imgs_out/test/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# breakpoint()
test_dataset = LabDataset(L_test, AB_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  

for i, (input,label) in enumerate(test_dataloader):
      
      print(input.shape, label.shape)
      break


      

# Ensure model is in evaluation mode and on the correct device
device = "cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu"
colorizer_eccv16.to(device)
colorizer_eccv16.eval()


avg_mse, avg_ssim, avg_psnr = evaluate_model_and_save_random_images(colorizer_eccv16, test_dataloader, device, output_dir)
print(f"Average MSE: {avg_mse}, Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}")
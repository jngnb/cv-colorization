import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Assuming modelColorizer442 and ModelColorizer442 are defined as above or imported appropriately
def load_model():
    # Load the model
    model = modelColorizer442(pretrained=True)
    model.eval()
    return model

def prepare_image(image_path):
    # Load the image
    img = Image.open(image_path).convert("L")  # Convert image to grayscale
    img = img.resize((64, 64))  # Resize to match model input

    # Transform the image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def colorize_image(model, input_tensor):
    # Colorize the image
    with torch.no_grad():
        output = model(input_tensor)
        return output

def save_image(output_tensor, output_path):
    # Convert the output tensor to image
    output_tensor = output_tensor.squeeze(0).cpu().detach()
    img_out = torch.clamp(output_tensor, 0, 1)  # Clamp the output to valid image range

    # Convert to PIL image and save
    img_out_pil = transforms.ToPILImage()(img_out).convert("RGB")
    img_out_pil.save(output_path)

if __name__ == "__main__":
    model = load_model()
    #TODO: Make sure to change image.jpg to our image name
    input_tensor = prepare_image("/code_442/inputIMG/image.jpg")
    output_tensor = colorize_image(model, input_tensor)
    
    save_image(output_tensor, "/code_442/outputIMG/OUTPUTcolorized_image.jpg")
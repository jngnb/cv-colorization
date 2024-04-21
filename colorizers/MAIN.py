import torch
from torchvision import transforms
from PIL import Image

def load_pretrained_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


def load_and_preprocess_image(image_path, target_size=(64, 64)):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    img = img.resize(target_size)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization, adjust if needed
    ])
    return transform(img).unsqueeze(0)
def save_colorized_image(output, save_path):
    # Assuming the output is a tensor with the AB channels and you have the L channel
    # You would need to convert this to RGB
    output = (output.squeeze().detach().numpy() * 255).astype('uint8')
    Image.fromarray(output, 'RGB').save(save_path)


def main(image_path, model_path, output_path):
    model = load_pretrained_model(model_path)

    input_tensor = load_and_preprocess_image(image_path)

    #TODO: Have code here that runs the pre-trained model with input tensor

    save_colorized_image(output, output_path)

if __name__ == '__main__':
    image_path = '.jpg'
    model_path = '.pth'
    output_path = 'path/to/save/colorized_image.jpg'
    main(image_path, model_path, output_path)
from utils import *
import torch
import torch.optim as optim
import cv2
from torchvision.utils import save_image
from argparse import ArgumentParser
from weights import *
from tqdm import tqdm
from time import time

# Argument Parsing 
parser = ArgumentParser(description='Content and style arguments for the neural style transfer model')
parser.add_argument('--content', help='Content image for the neural style transfer model', default='content_sample.jpg')
parser.add_argument('--style', help='Style image for the neural style transfer model', default='style_sample.jpg')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    Main function for training the Neural Style Transfer model
    """
    start = time()
    content = load_image(f'./data/{args.content}').to(device) # load content image
    style = load_image(f'./data/{args.style}', shape=content.shape[-2:]).to(device) # load style image

    model = get_model()

    content_features = extract_features(content, model)
    style_features = extract_features(style, model)
    
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
 
    target = content.clone().requires_grad_(True).to(device)
 
    optimizer = optim.Adam([target], lr=0.01)
    steps = 20000 # number of iterations on image
 
    # Main training loop
    print("Starting training run")
    for step in tqdm(range(1, steps+1)):
 
        target_features = extract_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
 
        style_loss = 0
 
        for layer in style_weights:
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
 
            target_gram = gram_matrix(target_feature)
 
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
 
            style_loss += layer_style_loss / (d * h * w)
 
        total_loss = content_loss * content_loss + style_weight * style_loss
 
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    save_image(target, 'out_img.jpg') # have final image
    print(f"Run completed in {(time() - start) / 60 :.2f} minutes.")

if __name__ == "__main__":
    main()

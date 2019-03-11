from utils import load_image, convert_image, get_model, gram_matrix
import torch
import torch.optim as optim
import cv2
from argparse import ArgumentParser
from features import get_features
from weights import *
from tqdm import tqdm

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
    content = load_image(f'./data/{args.content}') # load content image
    style = load_image(f'./data/{args.style}') # load style image

    model = get_model()

    content_features = get_features(content, model)
    style_features = get_features(style, model)
    
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
 
    target = content.clone().requires_grad_(True).to(device)
 
    optimizer = optim.Adam([target], lr=0.01)
    steps = 20000 # number of iterations on image
 
    # Main training loop
    print("Starting training run")
    for step in tqdm(range(1, steps+1)):
 
        target_features = get_features(target, model)
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

    final = convert_image(target).astype('float64')
    cv2.imwrite('./data/out_img.jpg', final)
    print("Run completed successfully")

if __name__ == "__main__":
    main()

import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image

def load_image(im_path, max_size=400, shape=None):
    """
    Load and transform an image (cutoff size is 400x400)
    """

    img = Image.open(im_path).convert('RGB')
    size = shape if shape else max_size if max(img.size) > max_size else max(img.size)
    
    trans = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img = trans(img)[:3,:,:].unsqueeze(0)

    return img

def convert_image(tensor):
    """ 
    Display a tensor as an image
    """

    img = tensor.to("cpu").clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1,2,0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)

    return img

def get_model():
    """
    Load pretrained VGG19 model
    """

    vgg = models.vgg19(pretrained=True).features

    for param in vgg.parameters():
        param.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg.to(device)

    return vgg

def extract_features(img, model, layers=None):
    """
    Run an image forward through the given model, collecting layer outputs as features.
    Default layers are for VGG19 as stated in Gatys et al.
    """

    if not layers:
        layers = {'0': 'conv1_1',
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21': 'conv4_2',
                '28': 'conv5_1'}

    features = {}
    x = img
    for lname, layer in model._modules.items(): 
        x = layer(x)
        if lname in layers:
            features[layers[lname]] = x 

    return features

def gram_matrix(tensor):
    """
    Compute the gram matrix of the given tensor
    """

    bs, d, h, w = tensor.size()
    flattened = tensor.view(d, h * w)
    gram = torch.mm(flattened, torch.t(flattened))

    return gram
    

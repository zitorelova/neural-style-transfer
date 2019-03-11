def get_features(img, model, layers=None):
    """
    Run an image forward through the given model, collecting layer outputs as features
    Default layers are for VGGNet as shown in Gatys et al.

    """ 
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}

    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


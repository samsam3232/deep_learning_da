import torch.nn as nn
import torch
import torchvision.models as md
from models.funcs import ReverseLayerF

VGG_SIZES = {16: md.vgg16, 19: md.vgg19, 11: md.vgg11, 13: md.vgg13}

def get_vgg(size, pretrained):

    model = VGG_SIZES[size](pretrained=pretrained)
    if pretrained:
        model = freeze_features(model)
        model = classifier_init(model)

    return model

def freeze_features(model):

    for param in model.features.parameters():
        param.requires_grad = False

    return model

def classifier_init(model):

    for mod in model.classifier.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_normal_(mod.weight.data)
            nn.init.normal_(mod.bias.data)
    
    return model

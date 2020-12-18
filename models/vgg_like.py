import torch.nn as nn
import torch.nn.init as init
import torchvision.models as md
from models.funcs import ReverseLayerF

VGG_SIZES = {16: md.vgg16, 19: md.vgg19, 11: md.vgg11, 13: md.vgg13}

def get_vgg_regular(size, pretrained, all_features = False):

    model = VGG_SIZES[size](pretrained=pretrained)
    if pretrained:
        model = freeze_features(model, all_features)
        model = model_reinit(model)

    return model

def freeze_features(model, all_features = False):

    if all_features:
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        for name, param in model.name_parameters():
            if int(name.split('.')[1]) > 13:
                param.requires_grad = False
    return model

def model_reinit(model, all_features = False):

    if all_features:
        for mod in model.classifier.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight.data)
                nn.init.normal_(mod.bias.data)
    else:
        count = 0
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                if count > 6:
                    if isinstance(mod, nn.Conv2d):
                        init.xavier_normal_(mod.weight.data)
                        if mod.bias is not None:
                            init.normal_(mod.bias.data)
                else:
                    count += 1
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight.data)
                nn.init.normal_(mod.bias.data)

    return model


class CCN_Model(nn.Module):

    def __init__(self, batch_layer = True):

        super(CCN_Model, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('f_conv1', nn.Conv2d(3, 30, kernel_size=3, padding=1))
        if batch_layer:
            self.features.add_module('f_norm1', nn.BatchNorm2d(30))
        self.features.add_module('f_relu1', nn.ReLU(True))
        self.features.add_module('f_pool1', nn.MaxPool2d(kernel_size=4, stride=4))
        self.features.add_module('f_conv2', nn.Conv2d(30, 30, kernel_size=3, padding=1))
        if batch_layer:
            self.features.add_module('f_norm2', nn.BatchNorm2d(30))
        self.features.add_module('f_relu2', nn.ReLU(True))
        self.features.add_module('f_pool2', nn.MaxPool2d(kernel_size=4, stride=4))
        self.features.add_module('f_maxpool', nn.MaxPool2d(kernel_size=4, stride=4))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(30 * 2 * 2, 100))
        if batch_layer:
            self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 50))
        if batch_layer:
            self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(50))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(50, 10))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(30*2*2, 50))
        if batch_layer:
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(50))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(50, 2))


    def forward(self, input, alpha):

        activations = dict()
        feats = input
        for m in self.features._modules:
            feats_curr = self.features._modules[m].forward(feats)
            if ("conv" in m):
                activations[m] = [feats, feats_curr]
            feats = feats_curr
        feats = feats.view(-1, 30*2*2)
        reversed_feats = ReverseLayerF.apply(feats, alpha)
        class_pred = self.class_classifier(feats)
        domain_pred = self.domain_classifier(reversed_feats)

        return class_pred, domain_pred, activations

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VggGetter:

    SETUPS = {"one_ds" : get_vgg_regular, "if_dann" : CCN_Model}

    def __init__(self, setup, **kwargs):
        self.setup = setup
        if setup == "one_ds":
            self.pretrained = kwargs["pretrained"]
            self.size = kwargs["size"]
            self.all_features = kwargs["all_features"]

    def get_model(self):

        if self.setup == "one_ds":
            model = self.SETUPS[self.setup](self.size, self.pretrained, self.all_features)
        else:
            model = self.SETUPS[self.setup](True)

        return model
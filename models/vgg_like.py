import torch.nn as nn
from models import ganinDann
import torchvision.models as md
from models.funcs import ReverseLayerF

VGG_SIZES = {16: md.vgg16, 19: md.vgg19}
STOPS = {16: 21, 19: 25}
STOPS_INITS = {16: 11, 19: 13}

class VggGetter(nn.Module):

    def __init__(self, size = 16, pretrained = False, freeze_all = True, ganin_da = False):
        super().__init__()
        self.ganin = False
        self.model = VGG_SIZES[size](pretrained=pretrained)
        if pretrained:
            self.model = self.freeze_features(freeze_all, STOPS[size])
        if ganin_da:
            self.model.domain_classifier = ganinDann.DomainClassifier().get_discriminator()
            self.ganin = True
        self.init_weights(pretrained, freeze_all, count_stp=STOPS_INITS[size])

    def freeze_weights(self, freeze_all = True, stop = 21):

        if freeze_all:
            for param in self.model.features.parameters():
                param.requires_grad = False

        else:
            for name, param in self.model.features.named_parameters():
                if int(name.split(".")[0]) <= stop:
                    param.requires_grad = False

    def init_weights(self, pretrained, freeze_all, count_stp = 11):

        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if self.ganin:
            for m in self.model.domain_classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        if (not pretrained) or (not freeze_all):
            count = 1
            for m in self.model.features.modules():
                if isinstance(m, nn.Conv2d):
                    if (pretrained) and (count < count_stp):
                        count += 1
                        continue
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    count += 1

    def forward(self, input, alpha, ganin=False, keep_feature=True, keep_classifier=False):

        activations_features, activations_classifier, domain_pred = dict(), dict(), None
        feats = input
        for m in self.model.features._modules.keys():
            if isinstance(self.model.features._modules[m], nn.Sequential):
                continue
            module = self.model.features._modules[m]
            feats_curr = module.forward(feats)
            if keep_feature:
                if "conv" in module._get_name().lower():
                    activations_features[m] = [feats, feats_curr]
            feats = feats_curr
        feats = feats.view(-1, 25088)
        if ganin:
            reversed_feats = ReverseLayerF.apply(feats, alpha)
            domain_pred = self.model.domain_classifier(reversed_feats)
        for m in self.model.classifier._modules.keys():
            if isinstance(self.model.features._modules[m], nn.Sequential):
                continue
            module = self.model.classifier._modules[m]
            feats_curr = module.forward(feats)
            if keep_classifier:
                if "linear" in module._get_name().lower():
                    activations_classifier[m] = [feats, feats_curr]
            feats = feats_curr
        class_pred = feats

        return class_pred, domain_pred, activations_classifier, activations_features
import torch.nn as nn
import torchvision.models as md
from models import ganinDann
from models.funcs import ReverseLayerF

class alexNetGetter(nn.Module):
    
    def __init__(self, pretrained=False, freeze_all=True, ganin_da=False):
        self.ganin = False
        self.model = md.alexnet(pretrained=pretrained)
        if pretrained:
            self.model = self.freeze_weight(freeze_all)
        if ganin_da:
            self.model.domain_classifier = ganinDann.DomainClassifier().get_discriminator()
            self.ganin = True
        self.model = self.init_weights(pretrained, freeze_all)


    def freeze_weight(self, freeze_all = True):

        if freeze_all:
            for param in self.model.features.params():
                param.requires_grad = False

        else:
            for name, param in self.model.named_parameters():
                if int(name.split(".")[1]) < 8:
                    param.requires_grad = False

    def init_weights(self, pretrained, freeze_all):

        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if self.ganin:
            for m in self.model.domain_classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        if pretrained:
            if not freeze_all:
                count = 1
                for m in self.model.features.modules():
                    if isinstance(m, nn.Conv2d):
                        if count >= 4:
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                        count += 1
    
    def forward(self, input, alpha, ganin=False, keep_feature=False, keep_classifier=False):

        activations_features, activations_classifier, domain_pred = dict(), dict(), None
        feats = input
        count = 1
        for m in self.model.features.modules():
            if isinstance(m, nn.Sequential):
                continue
            module = self.model.features._modules[m]
            feats_curr = module.forward(feats)
            if keep_feature:
                if "conv" in module._get_name().lower():
                    activations_features[module._get_name() + "_{}".format(str(count))] = [feats, feats_curr]
            feats = feats_curr
        count = 1
        feats = feats.view(-1, 4096)
        if ganin:
            reversed_feats = ReverseLayerF.apply(feats, alpha)
            domain_pred = self.model.domain_classifier(reversed_feats)
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Sequential):
                continue
            module = self.model.classifier._modules[m]
            feats_curr = module.forward(feats)
            if keep_classifier:
                if "linear" in module._get_name().lower():
                    activations_classifier[module._get_name() + "_{}".format(str(count))] = [feats, feats_curr]
            feats = feats_curr
        class_pred = feats
        return class_pred, domain_pred, activations_classifier, activations_features
import numpy as np
import torch
import torch.nn as nn
import models.vgg_like as vgg_like
import seaborn as sns
import torch.nn.init as init
import numpy as np
import utils
import pickle
from pruners.pruner_utils import get_distance_mask, simplify_locs, COMBINATIONS
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import models
import datasets.data_utils as data_utils
import copy

class IFPruner():
    
    def __init__(self, masks, device = "cuda", prune_features = True, prune_classifier=False, percent = 20, batch_size = 64):
        self.masks = masks
        for key in self.masks:
            for s_key in self.masks[key]:
                self.masks[key][s_key].to(device)
        self.device = device
        self.percent = percent
        self.prune_features = prune_features
        self.prune_classifier = prune_classifier
        self.batch_size = batch_size
        self.IFFeaturesPruner = IFFeaturesPruner(self.masks, self.batch_size, self.percent, self.device)

    def next_masks(self, percent, activations, model, prune_other = False):

        if self.prune_features:
            self.IFFeaturesPruner.prune_by_perc(percent, activations, model, prune_other)
            self.masks = self.IFFeaturesPruner.masks
            for key in self.masks:
                for s_key in self.masks[key]:
                    self.masks[key][s_key].to(self.device)


class IFFeaturesPruner():

    def __init__(self, masks, batch_size, percent = 20, device = "cuda"):
        self.masks = masks
        for key in self.masks:
            for s_key in self.masks[key]:
                self.masks[key][s_key].to(device)
        self.percent = percent
        self.device = device
        self.batch_size = batch_size

    def get_if(self, mask, source, weight):

        base = torch.transpose(source, 0, 1)
        base = torch.transpose(base, 1, 2)[:, :, :, None, None]
        base = base.detach() * torch.ones(weight.shape[-2:])
        base = base[:, :, None, :, :, :] * weight
        mask = torch.transpose(mask, 0, 1)
        mask = torch.transpose(mask, 1, 2)
        locs_src = list(np.where((mask.numpy() != 0)))
        locs_src = simplify_locs(locs_src, base)
        base_copy = base.copy()
        for i in COMBINATIONS:
            locs_curr = list(locs_src)
            locs_curr[2], locs_curr[1] = locs_curr[2] - i[0], locs_curr[1] - i[1]
            base[locs_curr][:][:, i[0] + 1, i[1] + 1] = base[locs_curr][:][:, i[0] + 1, i[1] + 1] / mask[
                locs_curr].view(-1, 1)
        base = np.where(base.detach() != base_copy.detach(), 0, base.detach())
        return base


    def get_if_single(self, source, target, weight, mask_source, mask_target):
        if_source = self.get_if(mask_source, source, weight)
        if_target = self.get_if(mask_target, target, weight)
        return if_source, if_target

    def next_masks(self, activations, model):

        activations_source, activations_target = activations
        if_source, if_target = dict(), dict()
        for key in activations_source:
            if_source_curr = torch.zeros(activations_source[key][0].size).to(self.device)
            if_target_curr = torch.zeros(activations_target[key][0].size).to(self.device)
            weight, _ = model.features._modules[key].parameters()
            mask_source, mask_target = get_distance_mask(activations_source[key][1],
                                          activations_target[key][1], self.batch_size, self.percent)
            for i in range(len(activations_source[key][0])):
                if_source_ex, if_target_ex = self.get_if_single(activations_source[key][0][i], activations_target[key][0][i],
                                                           weight, mask_source[i], mask_target[i])
                if_source_curr += if_source_ex
                if_target_curr += if_target_ex
            if_source_curr, if_target_curr = (if_source_curr / len(activations_source[key][0])), (if_target_curr / len(activations_target[key][0]))
            if_source[key], if_target[key] = if_source_curr, if_target_curr

        return if_source, if_target

    def prune_by_perc(self, percent, activations, model, classifier = False):

        if_source, if_target = self.next_masks(activations=activations, model=model)

        for m in model.features._modules.keys():
            if m in if_source:
                module = model.features._modules[m]
                weight, _ = module.parameters()
                tensor = weight.data.cpu().numpy()
                influence_factors = if_source[m] + if_target[m]
                nz = influence_factors[np.nonzero(influence_factors)]
                perc_value = np.percentile(abs(nz), (100 - percent))

                weights = weight.device
                nm = np.where(abs(influence_factors) > perc_value, 0, self.masks["features"][m])
                self.masks["features"][m] = nm.to(self.device)
                weight.data = torch.from_numpy(tensor * nm).to(weights)

        if classifier:
            percent = 20
            for m in model.classifier._modules.keys():
                module = model.classifier._modules[m]
                if "linear" in module._get_name().lower():
                    weight, _ = module.parameters()
                    tensor = weight.data.cpu().numpy()

                    nz = tensor[np.nonzero(tensor)]
                    perc_value = np.percentile(abs(nz), percent)

                    weights = weight.device
                    nm = np.where(abs(tensor) < perc_value, 0, self.masks["classifier"][m])
                    self.masks["classifier"][m] = nm.to(self.device)
                    weight.data = torch.from_numpy(tensor * nm).to(weights)

        return model

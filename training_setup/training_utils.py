import torch
import torch.nn as nn
import numpy as np

def get_masks(model):

    mask = {"features" : dict(), "classifier": dict()}
    for m in model.features._modules.keys():
        if isinstance(model._modules[m], nn.Sequential):
            continue
        module = model.features._modules[m]
        if "conv" in module._get_name().lower():
            weight, _ = module.parameters()
            mask["features"][m] = np.ones_like(weight.cpu().numpy())
    for m in model.classifier._modules.keys():
        if isinstance(model._modules[m], nn.Sequential):
            continue
        module = model.classifier._modules[m]
        if "linear" in module._get_name().lower():
            weight, _ = module.parameters()
            mask["classifier"][m] = np.ones_like(weight.cpu().numpy())

    return mask

def init_lists(iterations, end_iter):

    best_accuracy = 0
    best_accuracy_source = 0
    best_accuracy_target = 0
    comp = np.zeros(iterations, float)
    bestacc = np.zeros(iterations, float)
    bestacc_source = np.zeros(iterations, float)
    bestacc_target = np.zeros(iterations, float)
    all_loss = np.zeros(end_iter, float)
    all_accuracy = np.zeros(end_iter, float)
    all_accuracy_source = np.zeros(end_iter, float)
    all_accuracy_target = np.zeros(end_iter, float)
    return best_accuracy, best_accuracy_source, best_accuracy_target, comp, bestacc, bestacc_source, bestacc_target, \
           all_loss, all_accuracy, all_accuracy_source, all_accuracy_target


import torch
import torchvision.transforms as transforms
import dalib.vision.datasets as ds
from datasets import OfficeCaltech as OC

ME_STDS = {"OfficeCaltech" : {"A" : {"mean" : [0.6987, 0.6951, 0.6977], "std" : [0.2941, 0.2957, 0.2939]},
                              "W" : {"mean" : [0.5698, 0.5832, 0.5875], "std" : [0.2480, 0.2538, 0.2524]},
                              "D" : {"mean" : [0.3697, 0.3519, 0.3124], "std" : [0.1795, 0.1690, 0.1689]},
                              "C" : {"mean" : [0.5810, 0.5704, 0.5721], "std" : [0.2710, 0.2689, 0.2648]}}}

FILES  = {"OfficeCaltech" : OC}


def data_discr(source, target, batch_size):

    mixed_dis, mixed_class = list(), list()
    source_iter, target_iter = iter(source), iter(target)
    for i in range(min(len(source_iter), len(target_iter))):
        imgs_source, lb_source = source_iter.next()
        imgs_target, lb_target = target_iter.next()
        lst_images = torch.cat((imgs_source, imgs_target), 0)
        lst_labs_cls = torch.cat((lb_source, lb_target), 0)
        lst_labs_dis = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)), 0)
        perm = torch.randperm(2*batch_size)
        mixed_dis.append((lst_images[perm][:batch_size], lst_labs_dis[perm][:batch_size]))
        mixed_dis.append((lst_images[perm][batch_size:], lst_labs_dis[perm][:batch_size]))
        mixed_class.append((lst_images[perm][:batch_size], lst_labs_cls[perm][:batch_size]))
        mixed_class.append((lst_images[perm][batch_size:], lst_labs_cls[perm][:batch_size]))


    return mixed_dis, mixed_class

def get_transforms():

    train_transform = get_transform_train()
    val_tranform = get_transform_val()

    return train_transform, val_tranform

def get_transform_train():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return train_transform


def get_transform_val():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_tranform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return val_tranform


def get_data(ds, source, target, data_dir, batch_size, setup = "one_ds"):

    train_transform, val_tranform = get_transforms()
    train_source_loader, train_target_loader, val_source_loader, val_target_loader = FILES[ds].get_loaders_tf(source, target, f"{data_dir}/data/",
                                                                                           train_transform, val_tranform, batch_size)
    if setup == "one_ds":
        return train_source_loader, val_source_loader, val_target_loader

    if setup == "mixed":
        mixed_disc, mixed_class = data_discr(train_source_loader, train_target_loader, batch_size)
        return mixed_disc, mixed_class, val_source_loader

    else:
        return train_source_loader, train_target_loader, val_source_loader, val_target_loader
import argparse
import dalib.vision.datasets as ds
from torch.utils.data import DataLoader
from torchvision import transforms


def main(data_set, domains, data_dir, batch_size):

    sets_dict = {"Office31": ds.Office31, "OfficeCaltech" : ds.OfficeCaltech, "OfficeHome" : ds.OfficeHome,
                 "DomainNet": ds.DomainNet, "VisDA" : ds.VisDA2017}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_tranform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_source_dataset = sets_dict[data_set](root=data_dir, task=domains[0], download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_target_dataset = sets_dict[data_set](root=data_dir, task=domains[1], download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = sets_dict[data_set](root=data_dir, task=domains[0], download=True, transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_source_dataset, train_source_loader, train_target_loader, val_loader

#train_source_loader, train_target_loader, val_loader = main("OfficeCaltech", ["A", "W"], "/Users/samuelamouyal/PycharmProjects/deep_learning_da/", 32)



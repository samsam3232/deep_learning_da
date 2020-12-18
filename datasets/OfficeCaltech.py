import dalib.vision.datasets as ds
from torch.utils.data import DataLoader

DATASET_DOMAINS = ["A", "W", "D", "C"]

def get_loaders_tf(source, target, data_dir, transform_train, transform_val, batch_size):


    train_source_dataset = ds.OfficeCaltech(root=data_dir, task=source, download=True, transform=transform_train)
    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_target_dataset = ds.OfficeCaltech(root=data_dir, task=target, download=True, transform=transform_train)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_source_dataset = ds.OfficeCaltech(root=data_dir, task=source, download=True, transform=transform_val)
    val_source_loader = DataLoader(val_source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_target_dataset = ds.OfficeCaltech(root=data_dir, task=target, download=True, transform=transform_val)
    val_target_dataset = DataLoader(val_target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_source_loader, train_target_loader, val_source_loader, val_target_dataset


        
import dalib.vision.datasets as ds
from datasets.data_utils import get_transforms
import datasets

DATASETS_DICT = {"Office31" : datasets.Office31.get_loaders_tf, "OfficeHome" : datasets.OfficeHome.get_loader_tf,
                 "OfficeCaltech" : datasets.OfficeCaltech.get_loader_tf}
DATASETS_DOMAINS = {"Office31":["A", "W", "D"], "OfficeHome" : ["Ar", "Cl", "Pr", "Rw"], "OfficeCaltech" : ["A", "W", "D",
                   "C"], "DomainNet" : ["c", "i", "s", "r", "p", "q"], "VisDA2017" : ["T", "V"] }

def get_loaders(ds, data_dir, batch_size, source, target):

    train_transform, val_transform = get_transforms()
    train_source_loader, train_target_loader, val_source_loader, val_target_dataset \
        = DATASETS_DICT[ds](source, target, data_dir, train_transform, val_transform, batch_size)

    return train_source_loader, train_target_loader, val_source_loader, val_target_dataset
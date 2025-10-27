import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_cifar10_loaders(root='./data', labeled_fraction=1.0, batch_size=32, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,4),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    val = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_val)

    if labeled_fraction < 1.0:
        n = len(full_train)
        n_labeled = int(n * labeled_fraction)
        indices = np.random.permutation(n)
        labeled_idx = indices[:n_labeled]
        unlabeled_idx = indices[n_labeled:]
        labeled_ds = Subset(full_train, labeled_idx)
        unlabeled_ds = Subset(full_train, unlabeled_idx)
        labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return labeled_loader, unlabeled_loader, val_loader
    else:
        train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, None, val_loader

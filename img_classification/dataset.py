#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Data loader builders for image classification
'''

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def get_dataloaders(dataset, batch_size, workers, train_transforms, test_transforms):
    """Create training and test dataloaders

    Args:
        dataset (str): dataset being used
        batch_size (int): number of samples in each batch
        workers (int): number of workers being used for data loading
        train_transforms (torchvision.transforms.transforms.Compose): training set transformations
        test_transforms (torchvision.transforms.transforms.Compose): test set transformations

    Returns:
        train_loader (torch.utils.data.DataLoader): training dataloader
        test_loader (torch.utils.data.DataLoader): test dataloader
    """

    if dataset == 'mnist':
        # Train & test sets
        train_set = MNIST(root='./data', train=True, download=True, transform=train_transforms)
        val_set = MNIST(root='./data', train=False, download=True, transform=test_transforms)
    else:
        raise NotImplementedError()
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader

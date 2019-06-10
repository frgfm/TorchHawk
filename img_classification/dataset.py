#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Data loader builders for image classification
'''

import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def get_dataloaders(dataset, batch_size, workers):

    if dataset == 'mnist':
        # Normalize tensors (MNIST mean & std)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # Train & test sets
        train_set = MNIST(root='./data', train=True, download=True, transform=transform)
        val_set = MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError()
    # Data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader

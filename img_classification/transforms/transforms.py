#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Data loader builders for image classification
'''

import torchvision.transforms as transforms


def get_transforms(dataset):
    """Create dataset transforms for train and test sets

    Args:
        dataset (str): dataset being used

    Returns:
        train_transforms (torchvision.transforms.transforms.Compose): training transformations
        test_transforms (torchvision.transforms.transforms.Compose): test transformations
    """

    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
    else:
        raise NotImplementedError()

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])


    return train_transforms, test_transforms
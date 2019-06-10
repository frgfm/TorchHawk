#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utilities
'''

__author__ = 'François-Guillaume Fernandez'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'François-Guillaume Fernandez'
__status__ = 'Development'


import os
import torch
import shutil


def save_training_state(net, optimizer, epoch, training_name, folder='checkpoint', best_state=False):

    state_path = os.path.join(folder, f"{training_name}_epoch{epoch}.pth")
    state_dict = dict(state_dict=net.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch)

    torch.save(state_dict, state_path)

    if best_state:
        shutil.copyfile(state_path, os.path.join(folder, f"{training_name}_best.pth"))


def load_training_state(net, optimizer, file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Unable to locate {file_path}")

    state_checkpoint = torch.load(file_path)
    net.load_state_dict(state_checkpoint.get('state_dict'))
    optimizer.load_state_dict(state_checkpoint.get('optimizer'))
    epoch = state_checkpoint.get('epoch')

    return net, optimizer, epoch


def normal_initialization(m, mean=0, std=0.2):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    # Apply initial weights to convolutional and linear layers
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, mean, std)


def enumerate_parameters(net):

    num_params = sum(param.numel() for param in net.parameters() if param.requires_grad)

    return num_params

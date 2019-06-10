#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
PyTorch implementation of LeNet5
'''

import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self, in_channels, activation=None, pooling=None, drop_rate=0):
        super(LeNet5, self).__init__()

        if activation is None:
            activation = nn.ReLU(inplace=True)
        if pooling is None:
            pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.features = nn.Sequential(nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),
                                      activation,
                                      pooling,
                                      nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                      activation,
                                      pooling)

        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                        activation,
                                        nn.Linear(120, 84),
                                        activation,
                                        nn.Linear(84, 10),
                                        nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Training functions
'''

__author__ = 'François-Guillaume Fernandez'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'François-Guillaume Fernandez'
__status__ = 'Development'

import torch
from fastprogress import progress_bar


def train(net, train_loader, optimizer, criterion, master_bar, logger=None, log_freq=100, epoch=0):
    # Training
    net.train()
    loader_iter = iter(train_loader)
    for batch_idx in progress_bar(range(len(train_loader)), parent=master_bar):

        optimizer.zero_grad()

        x, target = loader_iter.next()
        if torch.cuda.is_available():
            x, target = x.cuda(), target.cuda()

        # Forward
        outputs = net(x)

        # Backprop
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        master_bar.child.comment = f"Training loss: {loss.item()}"

        if logger is not None and (batch_idx + 1) % log_freq == 0:
            current_iter = epoch * len(train_loader) + batch_idx + 1
            logger.add_scalar(f"Training loss", loss.item(), current_iter)
            # # Histograms of parameters value and gradients
            # for name, param in D.named_parameters():
            #     if param.requires_grad and "bias" not in name:
            #         tag = {name.replace('.', '/')}
            #         logger.add_histogram(f"{tag}/value", param.cpu(), current_iter)
            #         logger.add_histogram(f"{tag}/grad", param.grad.cpu(), current_iter)

    return loss.item()


def evaluate(net, test_loader, criterion):
    net.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for x, target in test_loader:
            # Work with tensors on GPU
            if torch.cuda.is_available():
                x, target = x.cuda(), target.cuda()

            # Forward + Backward & optimize
            outputs = net.forward(x)
            val_loss += criterion(outputs, target).item()
            # Index of max log-probability
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(test_loader)
    batch_size = x.size()[0]
    accuracy = correct / float(batch_size * len(test_loader))

    return val_loss, accuracy

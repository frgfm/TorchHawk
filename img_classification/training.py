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


def train_batch(model, x, target, optimizer, criterion):
    """Train a model for one iteration

    Args:
        model (torch.nn.Module): model to train
        loader_iter (iter(torch.utils.data.DataLoader)): training dataloader iterator
        optimizer (torch.optim.Optimizer): parameter optimizer
        criterion (torch.nn.Module): criterion object

    Returns:
        batch_loss (float): training loss
        correct (int): number of correct top1 prediction on batch
    """

    # Forward
    outputs = model(x)

    # Loss computation
    batch_loss = criterion(outputs, target)
    pred = outputs.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum()

    # Backprop
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss, correct


def train_epoch(model, train_loader, optimizer, criterion, master_bar,
                tb_logger=None, log_freq=100, epoch=0, log_weight_histo=False):
    """Train a model for one epoch

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training dataloader
        optimizer (torch.optim.Optimizer): parameter optimizer
        criterion (torch.nn.Module): criterion object
        master_bar (fastprogress.MasterBar): master bar of training progress
        tb_logger (torch.utils.tensorboard.SummaryWriter): tensorboard logger
        log_freq (int): number of iterations between each log
        epoch (int): current epoch index
    """

    # Training
    model.train()
    loader_iter = iter(train_loader)
    running_loss = 0
    for batch_idx in progress_bar(range(len(train_loader)), parent=master_bar):

        x, target = next(loader_iter)
        if torch.cuda.is_available():
            x, target = x.cuda(non_blocking=True), target.cuda(non_blocking=True)

        batch_loss, correct = train_batch(model, x, target, optimizer, criterion)

        master_bar.child.comment = f"Training loss: {batch_loss.item():.4} (Acc@1: {correct.item()/target.size(0):.2%})"

        # Tensorboard logs
        running_loss += batch_loss.item()
        if tb_logger is not None and ((epoch == 0 and batch_idx == 0) or (batch_idx + 1) % log_freq == 0):
            current_iter = epoch * len(train_loader) + batch_idx + 1
            if batch_idx > 0:
                running_loss /= log_freq
            tb_logger.add_scalar(f"Training loss", running_loss, current_iter)
            if batch_idx > 0:
                running_loss = 0
            if log_weight_histo:
                # Histograms of parameters value and gradients
                for name, param in model.named_parameters():
                    if param.requires_grad and "bias" not in name:
                        tag = name.replace('.', '/')
                        tb_logger.add_histogram(f"{tag}/value", param.cpu(), current_iter)
                        tb_logger.add_histogram(f"{tag}/grad", param.grad.cpu(), current_iter)
            tb_logger.flush()


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

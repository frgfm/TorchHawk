#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Benchmark of deep learning architectures for image classification
'''


import os
import argparse
import math
from shutil import rmtree
import torch
import torch.nn as nn
import torch.optim as optim
from fastprogress import master_bar
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import LeNet5
from training import train_epoch, evaluate
from utils import normal_initialization, set_seed
from dataset import get_dataloaders


SEED = 42


def main(args):

    set_seed(SEED)

    # Get the dataloaders
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, args.workers)

    # Architecture
    if args.dataset == 'mnist':
        in_channels = 1
    else:
        raise NotImplementedError()
    if args.activation == 'relu':
        activation = nn.ReLU(inplace=True)
    else:
        raise NotImplementedError()
    if args.pooling == 'max':
        pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    else:
        raise NotImplementedError()
    drop_rate = args.drop_rate

    # Build model
    model = LeNet5(in_channels, activation, pooling, drop_rate)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    # Weight normal initialization
    if args.init_weights:
        model.apply(normal_initialization)

    start_epoch = 0
    if args.resume is not None:
        model, optimizer, start_epoch = load_training_state(model, optimizer, args.resume)

    # Loss function & optimizer
    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()
    if args.optimizer == 'sgd':
        # Issue
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0, threshold=1e-2, verbose=True)

    # Output folder
    output_folder = os.path.join(args.output_folder, args.training_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_path = os.path.join(args.output_folder, 'logs', args.training_name)
    if os.path.exists(log_path):
        rmtree(log_path)
    logger = SummaryWriter(log_path)

    # Train
    best_loss = math.inf 
    mb = master_bar(range(args.nb_epochs))
    for epoch_idx in mb:
        # Training
        train_epoch(model, train_loader, optimizer, criterion, mb, tb_logger=logger, epoch=start_epoch + epoch_idx)

        # Evaluation
        val_loss, accuracy = evaluate(model, test_loader, criterion)

        mb.first_bar.comment = f"Epoch {start_epoch+epoch_idx+1}/{start_epoch+args.nb_epochs}"
        mb.write(f'Epoch {start_epoch+epoch_idx+1}/{start_epoch+args.nb_epochs} - Validation loss: {val_loss:.4} (Acc@1: {accuracy:.2%})')

        # State saving
        if val_loss < best_loss:
            print(f"Validation loss decreased {best_loss:.4} --> {val_loss:.4}: saving state...")
            best_loss = val_loss
            torch.save(dict(epoch=start_epoch + epoch_idx,
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            val_loss=val_loss),
                       os.path.join(output_folder, "training_state.pth"))

        if logger is not None:
            current_iter = (start_epoch + epoch_idx + 1) * len(train_loader)
            logger.add_scalar(f"Validation loss", val_loss, current_iter)
            logger.add_scalar(f"Error rate", 1 - accuracy, current_iter)
            logger.flush()
        scheduler.step(val_loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("training_name", type=str, help="The name of your training")
    # Dataset
    parser.add_argument("--dataset", type=str, default='mnist', help="Dataset to be used (default: mnist)")
    # Hardware
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID you wish to use (default: 0)")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers used for data loading (default: 2)")
    # Loader
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size (default: 4)")
    # Architecture
    parser.add_argument("--activation", type=str, default='relu', help="Activation function to be used (default: relu)")
    parser.add_argument("--pooling", type=str, default='max', help="Pooling layer to be used (default: maxpooling)")
    parser.add_argument("--drop_rate", type=float, default=0., help="Drop rate of FC layers' neurons (default: 0)")
    parser.add_argument('--init_weights', action='store_true', help="Should the weights be initialized (default: False)")
    # Hyper parameters
    parser.add_argument("--criterion", type=str, default='ce', help="Loss criterion (default: cross entropy)")
    parser.add_argument("--optimizer", type=str, default='sgd', help="Parameter optimizer (default: SGD)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--momentum", type=float, default=0., help="SGD Momentum (default: 0)")
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay (default: 0)")
    parser.add_argument("--nesterov", action='store_true', help="Nesterov momentum (default: False)")
    parser.add_argument("--nb_epochs", "-n", type=int, default=10, help="Number of epochs to train (default: 10)")
    # Session management
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file to resume (default: None)")
    parser.add_argument("--output_folder", type=str, default='.', help="Output folder for log and states (default: '.')")
    args = parser.parse_args()

    main(args)

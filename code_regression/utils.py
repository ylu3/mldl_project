#!/usr/bin/env python3
import argparse
import functools
import os
import time
from datetime import datetime
from typing import Sequence, Text, Union
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

data_root = '../datasets/ROD-synROD'

def weights_init(m):
    """
    This is the same weight initializer used for the paper
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def entropy_loss(logits):
    """
    Domain Adaptation-specific Regularization loss
    :param logits:
    :return:
    """
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))

def rotation_loss(radians, labels):
    """
    Ratation loss based on sinus.
    :param radians:
    :return:
    """
    print(radians)
    radians = torch.reshape(radians, (-1,))
    return torch.sin(abs(radians-labels)/2).sum() / radians.size(0)

class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


class EvaluationManager:
    def __init__(self, nets):
        self.nets = nets

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        for net in self.nets:
            net.eval()

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        for net in self.nets:
            net.train()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class IteratorWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            items = self.iterator.next()
        except:
            self.__iter__()
            items = self.iterator.next()
        return items


def add_base_args(parser: argparse.ArgumentParser):
    """
    Add arguments which are not specific for the DA method. If you implement several versions of train.py you can
    import this function
    :param parser:
    :return:
    """
    # Dataset arguments
    parser.add_argument("--data_root")

    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for each DataLoader")
    parser.add_argument("--logdir", default="experiments", help="Directory for checkpoints and TensorBoard logs")
    parser.add_argument('--gpu', default=0, help="Which CUDA device to use")
    parser.add_argument('--suffix', type=str, default=None, help="Suffix for your run name")

    # hyper-params
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--lr_mult", default=1.0, type=float, help="Learning rate multiplier for non-pretrained layers")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay regularization")
    parser.add_argument("--dropout_p", default=0.5, help="Dropout (not for the backbone!)")

    parser.add_argument('--test_batches', default=100, type=int,
                        help="Number of batches to be considered at test time for source classification and the" 
                             " rotation task. Note that the evaluation on target is always done on all batches")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint if it exists")
    parser.add_argument('--smallset', action='store_true', default=False, help="Train on small set")
    parser.add_argument('--sanitycheck', action='store_true', default=False, help="Check sanity set")
    parser.add_argument("--projection_dim", default=100, type=int, help="Projection dimension in NetRot")
    parser.add_argument('--tanh', action='store_true', default=False, help="Try tanh as activation")

def make_paths(root, smallset=False):
    data_root_source = os.path.join(root, 'synROD')
    data_root_target = os.path.join(root, 'ROD')
    if smallset:
        split_source_train = os.path.join(data_root_source, 'smallsynROD_train.txt')
        split_source_test = os.path.join(data_root_source, 'smallsynROD_test.txt')
        split_target = os.path.join(data_root_target, 'smallROD.txt')
    else:
        split_source_train = os.path.join(data_root_source, 'synARID_50k-split_sync_train1.txt')
        split_source_test = os.path.join(data_root_source, 'synARID_50k-split_sync_test1.txt')
        split_target = os.path.join(data_root_target, 'wrgbd_40k-split_sync.txt')
    
    return data_root_source, data_root_target, split_source_train, split_source_test, split_target


def map_to_device(device: torch.device, t: Sequence):
    """
    Just a simple util function to move several tensors to a device in a single line
    :param device:
        The device
    :param t:
        A tuple or iterable of tensors (or any other moveable thing)
    :return:
        Moved tuple
    """
    return tuple(map(lambda x: x.to(device), t))


def save_checkpoint(path: Text,
                    epoch: int,
                    modules: Union[nn.Module, Sequence[nn.Module]],
                    optimizers: Union[opt.Optimizer, Sequence[opt.Optimizer]],
                    safe_replacement: bool = True):
    """
    Save a checkpoint of the current state of the training, so it can be resumed.
    This checkpointing function assumes that there are no learning rate schedulers or gradient scalers for automatic
    mixed precision.
    :param path:
        Path for your checkpoint file
    :param epoch:
        Current (completed) epoch
    :param modules:
        nn.Module containing the model or a list of nn.Module objects
    :param optimizers:
        Optimizer or list of optimizers
    :param safe_replacement:
        Keep old checkpoint until the new one has been completed
    :return:
    """

    # This function can be called both as
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, my_module, my_opt)
    # or
    # save_checkpoint('/my/checkpoint/path.pth', my_epoch, [my_module1, my_module2], [my_opt1, my_opt2])
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, opt.Optimizer):
        optimizers = [optimizers]
 
    # Data dictionary to be saved
    data = {
        'epoch': epoch,
        # Current time (UNIX timestamp)
        'time': time.time(),
        # State dict for all the modules
        'modules': [m.state_dict() for m in modules],
        # State dict for all the optimizers
        'optimizers': [o.state_dict() for o in optimizers]
    }

    # Safe replacement of old checkpoint
    temp_file = None
    if os.path.exists(path) and safe_replacement:
        # There's an old checkpoint. Rename it!
        temp_file = path + '.old'
        os.rename(path, temp_file)

    # Save the new checkpoint
    with open(path, 'wb') as fp:
        torch.save(data, fp)
        # Flush and sync the FS
        fp.flush()
        os.fsync(fp.fileno())

    # Remove the old checkpoint
    if temp_file is not None:
        os.unlink(path + '.old')


def load_checkpoint(path: Text,
                    default_epoch: int,
                    modules: Union[nn.Module, Sequence[nn.Module]],
                    optimizers: Union[opt.Optimizer, Sequence[opt.Optimizer]],
                    verbose: bool = True):
    """
    Try to load a checkpoint to resume the training.
    :param path:
        Path for your checkpoint file
    :param default_epoch:
        Initial value for "epoch" (in case there are not snapshots)
    :param modules:
        nn.Module containing the model or a list of nn.Module objects. They are assumed to stay on the same device
    :param optimizers:
        Optimizer or list of optimizers
    :param verbose:
        Verbose mode
    :return:
        Next epoch
    """
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(optimizers, opt.Optimizer):
        optimizers = [optimizers]

    # If there's a checkpoint
    if os.path.exists(path):
        # Load data
        data = torch.load(path, map_location=next(modules[0].parameters()).device)

        # Inform the user that we are loading the checkpoint
        if verbose:
            print(f"Loaded checkpoint saved at {datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S')}. "
                  f"Resuming from epoch {data['epoch']}")

        # Load state for all the modules
        for i, m in enumerate(modules):
            modules[i].load_state_dict(data['modules'][i])

        # Load state for all the optimizers
        for i, o in enumerate(optimizers):
            optimizers[i].load_state_dict(data['optimizers'][i])

        # Next epoch
        return data['epoch'] + 1
    else:
        return default_epoch

def extract_dataset(topK, firstC, input, ouput):
    input = os.path.join(data_root, input)
    df_rod = pd.read_csv(input, sep=" ", header=None)
    df_rod.columns = ["path", "category"]
    df_rod_extracted = df_rod.groupby('category').head(topK).sort_values('category')
    df_rod_extracted.loc[df_rod_extracted['category'] < firstC].to_csv(ouput, header=None, index=None, sep=' ')

def extract_small_dataset():
    extract_dataset(200, 5, 'ROD/wrgbd_40k-split_sync.txt', 'smalldatasets/smallROD.txt')
    extract_dataset(200, 5, 'synROD/synARID_50k-split_sync_train1.txt', 'smalldatasets/smallsynROD_train.txt')
    extract_dataset(200, 5, 'synROD/synARID_50k-split_sync_test1.txt', 'smalldatasets/smallsynROD_test.txt')

extract_small_dataset()

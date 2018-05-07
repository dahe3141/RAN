import argparse
import os
from train import Trainer
from models import RAN

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from dataset import MOT16_train_dataset, pad_packed_collate


def main():

    opt = lambda: None
    opt.dataroot = '/scratch0/MOT/MOT16'
    opt.detroot = '/scratch0/MOT/MOT16/external'
    opt.outf = '../results'
    opt.batch_size = 64
    opt.nepoch = 100
    opt.use_cuda = True

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'visualization'))
    except OSError:
        pass
    try:
        os.makedirs(os.path.join(opt.outf, 'models'))
    except OSError:
        pass

    train_dataset = MOT16_train_dataset(opt.dataroot, opt.detroot)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=pad_packed_collate)
    # TODO: add a get_dim method to dataset
    trainer = Trainer(opt, 4, 32, train_data_loader)
    trainer.train()


if __name__ == '__main__':
    main()

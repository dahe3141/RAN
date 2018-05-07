import os
import torch
from train import Trainer
from torch.utils.data import DataLoader
from dataset import MOT16_train_dataset, pad_packed_collate


def main():

    opt = lambda: None
    opt.dataroot = '/scratch0/MOT/MOT16'
    opt.detroot = '/scratch0/MOT/MOT16/external'
    opt.outf = '../results'
    opt.history_size = 10
    opt.batch_size = 64
    opt.nepoch = 100
    opt.use_cuda = torch.cuda.is_available()

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

    trainer = Trainer(opt, train_dataset.motion_dim, train_dataset.feat_dim, train_data_loader)
    trainer.train()


if __name__ == '__main__':
    main()

import argparse
import os
from train import trainIters
from models import RAN, save_model, load_model

import torch as t
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from dataset import MOT16_train_dataset, pad_packed_collate
from utils import show_track

# change the mot17_root_dir to the "MOT16" dir and run
def main(args):
    # if args.data_set == 1:
    #     seq_train_fn = "MOT16-train.txt"
    #     seq_test_fn = "MOT16-test.txt"
    #     data_set = "MOT16"
    # elif args.data_set == 2:
    #     seq_train_fn = "MOT16-train.txt"
    #     seq_test_fn = "MOT16-test.txt"
    #     data_set = "MOT16"
    #     print('not implemented')
    #     exit()
    # else:
    #     seq_train_fn = "MOT16-train.txt"
    #     seq_test_fn = "MOT16-test.txt"
    #     data_set = "MOT16"
    #     print('not implemented')
    #     exit()

    mot16_root_dir = '/scratch0/MOT/MOT16'

    train_dataset = MOT16_train_dataset(mot16_root_dir,
                                        val_id=7,
                                        trans_func=None)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=64,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=pad_packed_collate)

    model_prefix = '/scratch0/RAN/trained_model/ran'
    m = RAN(input_size=4, hidden_size=32, history_size=10, drop_rate=0.5, save_prefix=model_prefix)
    trainIters(m, train_data_loader, n_epoch=100)

    # load_model(m)
    save_model(m)



def sanity(idx):
    # define saved_path:
    # mot16_root_dir = "~/Projects/Datasets/MOT16"
    # processed_folder = 'processed'
    # training_file = 'train.pt'
    # saved_path = os.path.join(processed_folder, training_file)

    mot16_root_dir = os.path.join('Data', 'MOT16')
    train_dataset = MOT16_train_dataset(mot16_root_dir,
                                        val_id=7,
                                        trans_func=None,
                                        overwrite=False)

    show_track(idx, train_dataset)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="to be filled")
    # parser.add_argument('-d', '--data_set', type=int, default=1,
    #                     help="1: MOT16; 2: MOT17")
    # parser.add_argument('-p', '--data_root', type=str, default=os.path.join(os.path.pardir, "MOT16"),
    #                     help="path to the sequence directory")
    #
    # args = parser.parse_args()
    # main(args)

    # sanity(100)

    main("none")

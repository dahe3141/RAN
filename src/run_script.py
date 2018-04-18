import argparse
import os
from train import trainIters
from models import RAN

import torch as t
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from data_loader import MOT16_train_dataset, pad_packed_collate
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

    mot16_root_dir = os.path.abspath(os.path.join(os.path.pardir, "Data", "MOT16"))
    train_dataset = MOT16_train_dataset(mot16_root_dir,
                                        saved_path='saved_data',
                                        val_id=7,
                                        trans_func=None)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=1,
                                   collate_fn=pad_packed_collate)
    m = RAN(input_size=4,
            hidden_size=32,
            history_size=10,
            drop_rate=0.5)

    trainIters(m, train_data_loader, n_epoch=100)



    # not NLL. need to unroll
    # criterion = loss_fn

    # a = get_batch(train_samples)


    # Note: not sure how exactly training data is sampled.
    # My understanding is that it takes a fixed number of
    # consecutive frames from randomly selected 64 trajectories.

    # 399 total trajs, average 152 frames, median 80, max 1050, min 10


    # testing =============================================================================
    # for idx_det, seq in enumerate(mot17_root_dir):
    #     seq_root = os.path.join(args.data_root, "train", seq)
    #     res_path = os.path.join(os.path.pardir, "res", data_set, seq+".txt")
    #
    #     # load data
    #     det, gt, img_fns = load_mot16_train(seq_root)
    #
    #     training_trajs = generate_training_samples(det, gt)




def sanity(idx):

    mot16_root_dir = os.path.abspath(
        os.path.join(os.path.pardir, "Data", "MOT16"))

    train_dataset = MOT16_train_dataset(mot16_root_dir,
                                        saved_path='saved_data',
                                        val_id=7,
                                        trans_func=None)
    gt = train_dataset.gt
    sample = train_dataset.train_samples
    img_id = train_dataset.img_id
    mot_train_seq = train_dataset.img_id

    show_track(idx, train_dataset)

    pass








if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="to be filled")
    # parser.add_argument('-d', '--data_set', type=int, default=1,
    #                     help="1: MOT16; 2: MOT17")
    # parser.add_argument('-p', '--data_root', type=str, default=os.path.join(os.path.pardir, "MOT16"),
    #                     help="path to the sequence directory")
    #
    # args = parser.parse_args()
    # main(args)

    sanity(100)

    # main("none")

import argparse
import os
from utils import load_mot16_gt, load_mot16_det, generate_training_samples, get_batch
from train import loss_fn, train, trainIters
from models import RAN

import torch as t
import torch.nn as nn
import pickle



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

    # load training data
    saved_path = os.path.join(mot16_root_dir, 'train_samples')
    if os.path.exists(saved_path):
        with open(saved_path, 'rb') as f:
            train_samples, mot_train_seq = pickle.load(f)
    else:
        gt, mot_train_seq = load_mot16_gt(mot16_root_dir)
        det = load_mot16_det(mot16_root_dir, mot_train_seq)
        train_samples = generate_training_samples(det, gt, mot_train_seq)

        with open(saved_path, 'wb+') as f:
            pickle.dump((train_samples, mot_train_seq), f)

    use_cuda = t.cuda.is_available()

    m = RAN(input_size=4,
            hidden_size=32,
            history_size=10,
            drop_rate=0.5)


    trainIters(m, train_samples, n_iters=5)



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










if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description="to be filled")
    # parser.add_argument('-d', '--data_set', type=int, default=1,
    #                     help="1: MOT16; 2: MOT17")
    # parser.add_argument('-p', '--data_root', type=str, default=os.path.join(os.path.pardir, "MOT16"),
    #                     help="path to the sequence directory")
    #
    # args = parser.parse_args()
    # main(args)
    main("none")

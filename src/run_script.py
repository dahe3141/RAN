import argparse
import os
from utils import load_mot16_train
from train import generate_training_samples
from models import RAN

def main(args):
    if args.data_set == 1:
        seq_train_fn = "MOT16-train.txt"
        seq_test_fn = "MOT16-test.txt"
        data_set = "MOT16"
    elif args.data_set == 2:
        seq_train_fn = "MOT16-train.txt"
        seq_test_fn = "MOT16-test.txt"
        data_set = "MOT16"
        print('not implemented')
        exit()
    else:
        seq_train_fn = "MOT16-train.txt"
        seq_test_fn = "MOT16-test.txt"
        data_set = "MOT16"
        print('not implemented')
        exit()

    seq_train_fn = os.path.join(os.path.pardir, "seqmaps", seq_train_fn)
    seq_test_fn = os.path.join(os.path.pardir, "seqmaps", seq_test_fn)

    # read in video sequence names
    with open(seq_train_fn) as fd:
        seqs_train = [line.rstrip('\n') for line in fd]

    for idx_det, seq in enumerate(seqs_train):
        seq_root = os.path.join(args.data_root, "train", seq)
        res_path = os.path.join(os.path.pardir, "res", data_set, seq+".txt")

        # load data
        det, gt, img_fns = load_mot16_train(seq_root)

        training_trajs = generate_training_samples(det, gt)

        m = RAN(input_size=4,
                hidden_size=32,
                history_size=10)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="to be filled")
    parser.add_argument('-d', '--data_set', type=int, default=1,
                        help="1: MOT16; 2: MOT17")
    parser.add_argument('-p', '--data_root', type=str, default=os.path.join(os.path.pardir, "MOT16"),
                        help="path to the sequence directory")

    args = parser.parse_args()
    main(args)


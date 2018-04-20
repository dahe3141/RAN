import os
import torch as t
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from utils import load_mot16_det, load_mot16_gt, generate_training_samples, generate_img_fn
import pickle
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.autograd import Variable


class MOT16_train_dataset(Dataset):
    """ MOT16 dataset.

    Args:
        root (string): root path to MOT16 data dir.
        saved_path (string): Directory to save data on disk.
        trans_func (callable, optional): Optional transform to be applied
            on a sample.
        val_id (int): indicate which video is used for validation.
            training samples will not be generated on that video.
            to use validation data, call data.det[data.val_id - 1] and
            data.gt[data.val_id - 1]

    """
    processed_folder = 'processed'
    training_file = 'train.pt'

    def __init__(self, root, val_id=7,
                 trans_func=None, overwrite=False):

        self.root = os.path.expanduser(root)
        self.val_id = val_id
        self.trans_func = trans_func
        saved_path = os.path.join(self.root, self.processed_folder, self.training_file)

        if overwrite and os.path.exists(saved_path):
            os.remove(saved_path)

        if os.path.exists(saved_path):
            print("loading saved data from {}".format(saved_path))
            with open(saved_path, 'rb') as f:
                self.train_samples, self.img_id, self.mot_train_seq, self.gt,\
                    self.det = pickle.load(f)
        else:
            print('generating data and saving to {}'.format(saved_path))
            if not os.path.exists(os.path.join(self.root, self.processed_folder)):
                os.mkdir(os.path.join(self.root, self.processed_folder))

            self.gt, self.mot_train_seq = load_mot16_gt(self.root)
            self.det = load_mot16_det(self.root, self.mot_train_seq)
            self.train_samples, self.img_id = \
                generate_training_samples(self.det, self.gt, val_id=val_id)

            with open(saved_path, 'wb+') as f:
                pickle.dump((self.train_samples, self.img_id,
                             self.mot_train_seq, self.gt, self.det), f)

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        """
        Currently just return a trajectory
        Args:
            idx (int):
        Returns:
            sample (ndarray): (n, 4)
        """
        # img_name = generate_img_fn(self.root, self.mot_train_seq, self.img_id[idx])
        # # img_ic = io.imread_collection(img_name, conserve_memory=True, plugin=None)
        #
        # sample = (img_name, self.train_samples[idx])
        #
        # if self.trans_func:
        #     sample = self.trans_func(sample)

        return self.train_samples[idx]





# collate_fn([dataset[i] for i in batch_indices])

def pad_packed_collate(batch, hist_size=10):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
            batch (list): list of ndarray (n, feat), length is batch size
       Output:
            packed_batch (PackedSequence): (T, B, F)
            packed_ext (PackedSequence): (T, B, F, H)
            length (list): list of seq_len
        Note: the dataloader does not know history size which is an attribute of
            The model. It may not be appropriate to generate external here.
    """
    # pad sequence as TxBx*
    # T is length[0] longest seq, B is batch, * is feature
    # length and padded is sorted in descending order
    if len(batch) == 1:
        sorted_batch = batch
        padded_batch = batch[0][:, None, :]  # add batch dimension
        lengths = [padded_batch.shape[0]]
    else:
        # sort
        sorted_batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)
        lengths = [s.shape[0] for s in sorted_batch]

        # pad
        max_len, n_feats = sorted_batch[0].shape
        padded_batch = \
            [np.concatenate((s, np.zeros((max_len - s.shape[0], n_feats),
                                         dtype=np.float32)), axis=0)
             if s.shape[0] != max_len else s for s in sorted_batch]

        # stack
        padded_batch = np.stack(padded_batch, axis=1)

    # pack
    packed_batch = pack(Variable(t.from_numpy(padded_batch)), lengths,
                        batch_first=False)

    return packed_batch





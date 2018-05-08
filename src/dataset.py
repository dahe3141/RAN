import os
import torch as t
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from utils import load_mot16_det, load_mot16_gt, generate_trainset, match_detections, ind_select
import pickle
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import collections


class MOT16_train_dataset(Dataset):
    """ MOT16 dataset.

    Args:
        root (string): root path to MOT16 data dir.
        trans_func (callable, optional): Optional transform to be applied
            on a sample.
        val_id (int): indicate which video is used for validation.
            training samples will not be generated on that video.
            to use validation data, call data.det[data.val_id - 1] and
            data.gt[data.val_id - 1]

    """
    processed_folder = 'processed'
    training_file = 'train.pt'
    sequence = ['MOT16-13', 'MOT16-11', 'MOT16-10',
                'MOT16-09', 'MOT16-05', 'MOT16-04', 'MOT16-02']

    def __init__(self, root, det_dir):

        self.root = os.path.expanduser(root)
        self.det_dir = os.path.expanduser(det_dir)
        self.saved_path = os.path.join(self.root, self.processed_folder, self.training_file)
        self.refined_det_path = os.path.join(self.root, self.processed_folder)

        if os.path.exists(self.saved_path):
            print("Loading data from {}".format(self.saved_path))
            self._load(self.saved_path)

        else:
            if not os.path.exists(os.path.dirname(self.saved_path)):
                os.makedirs(os.path.dirname(self.saved_path))
            print('Generating data and saving to {}'.format(self.saved_path))
            self._process()
            self._load(self.saved_path)

    def __len__(self):
        return len(self.motion)

    def __getitem__(self, idx):
        return self.motion[idx], self.appearance[idx]

    def _process(self):
        # gt annotation
        gt_all, image_filenames = load_mot16_gt(self.root)

        # detection files
        det_files = [os.path.join(self.det_dir, '{}_det.txt'.format(x)) for x in self.sequence]
        feat_files = [os.path.join(self.det_dir, '{}_feat.txt'.format(x)) for x in self.sequence]

        det_all = load_mot16_det(det_files, feat_files)
        det_refined = match_detections(gt_all, det_all)
        bbox, motion, appearance, video_id, frame_num = generate_trainset(det_refined)

        self._write_det_refined(det_refined)

        with open(self.saved_path, 'wb+') as f:
            pickle.dump((bbox, motion, appearance, video_id, frame_num, image_filenames), f)

    def _load(self, saved_path):
        with open(saved_path, 'rb') as f:
            self.bbox, self.motion, self.appearance, self.video_id, self.frame_num, \
                self.image_filenames = pickle.load(f)

        self.motion_dim = self.motion[0].shape[1]
        self.feat_dim = self.appearance[0].shape[1]

    def _write_det_refined(self, seq_info):

        for i, seq in enumerate(seq_info):
            seq_name = self.sequence[i]
            bbox = seq['bbox'].copy()
            bbox[:, 2:4] += bbox[:, 0:2]

            out = np.hstack((seq['frame_num'][:, None], seq['track_id'][:, None], bbox, seq['feat']))
            np.savetxt('{}/{}.txt'.format(self.refined_det_path, seq_name), out, delimiter=',', fmt='%.6f')


# collate_fn([dataset[i] for i in batch_indices])
def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
            batch (list): list of ndarray (n, feat)
                len(batch) is the size of the batch
                n is the number of features
       Output:
            packed_batch (PackedSequence): (T, B, F)
            packed_ext (PackedSequence): (T, B, F, H)
            length (list): list of seq_len
        Note: the dataloader does not know history size which is an attribute of
            The model. It may not be appropriate to generate external here.
    """
    if isinstance(batch[0], np.ndarray):
        pass
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [pad_packed_collate(samples) for samples in transposed]

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
    packed_batch = pack_padded_sequence(Variable(t.from_numpy(padded_batch)), lengths,
                        batch_first=False)

    return packed_batch


if __name__ == '__main__':
    dataset = MOT16_train_dataset(os.path.expanduser('~/Projects/Datasets/MOT16'),
                                  os.path.expanduser('~/Projects/Datasets/MOT16/external'))





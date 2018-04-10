import numpy as np
import torch as t
from utils import iou
import time

def generate_training_samples(det, gt):
    """Generate training trajectory for a video

    Args:
        det (list): list of detections for each frame. Indexed by zero-based frame number. Represented with np array: one detection per row. [bbox, score]
        gt (list): list of trajectories. It is not indexed by the trajectory id. One frame per row. [frame_num, track_id, x, y, w, h]

    Return:
        training_trajs (list): a list of generated training trajectories. Represented with np array. One frame per row. track_id is discarded. [frame_num, bbox]

    """
    num_traj = len(gt)
    training_trajs = []
    # one training sample for each traj and each frame

    for t in range(0, num_traj):
        frame_idx = gt[t][:, 0].astype(int)
        traj_train = np.empty((0, 5))
        for i, f in enumerate(frame_idx):

            gt_bbox = gt[t][i, 2:6]
            det_bboxs = det[f-1][:, 0:4] # all bbox in frame
            iou_idx = iou(gt_bbox, det_bboxs)
            # no detected bbox can be associated to gt
            if len(iou_idx) == 0:
                continue
            bbox_sel = det_bboxs[np.random.permutation(iou_idx)[0], :]
            traj_train = np.append(traj_train, np.append(f, bbox_sel))

            traj_train = traj_train.reshape((-1, 5))

        training_trajs.append(traj_train)

    return training_trajs

def train(model, criterion, sample):
    """In progress"""
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden()
    for batch, i in enumerate(sample):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        t.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()







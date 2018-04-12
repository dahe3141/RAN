import numpy as np
import os
import csv


def load_mot16_train(data_root):
    """Parse MOT16 training data

    Args:
        data_root (str): path MOT16 folder

    Returns:
        det (list): list of detections. One structured np array for each video.
            one detection per row. [frame_num | x, y, w, h | score]
        gt (list): list of trajectories. One np array for each video.
            One frame per row. [frame_num \ track_id \ x, y, w, h \ class \ vis]
        mot_train_seq (list): video names as indexed by vid_id - 1

    expecting files in the following format:
    for ground truth:
        <frame_num>,<track_id>,<x>,<y>,<w>,<h>,<conf>,<class>,<vis>
    for detection:
        <frame_num>,<-1>,<x>,<y>,<w>,<h>,<conf>,<-1>,<-1>,<-1>

    Note:
        The x,y coord returned for bbox is the center pixel location.
    """
    mot_train_seq = os.listdir(os.path.join(data_root, 'train'))
    mot_train_seq_gt_fn = [os.path.join(data_root, 'train', seq, 'gt',
                                    'gt.txt') for seq in mot_train_seq]
    gt = []
    for i, gt_fn in enumerate(mot_train_seq_gt_fn):
        raw_gt = np.genfromtxt(gt_fn, delimiter=',',
                               dtype=[('frame_num', 'i4'),
                                      ('track_id', 'i4'),
                                      ('x', 'f4'),
                                      ('y', 'f4'),
                                      ('w', 'f4'),
                                      ('h', 'f4'),
                                      ('conf', 'i4'),
                                      ('class', 'i4'),
                                      ('vis', 'f4')])
        # we should consider what to be included for training
        # raw_gt['class'] include distractors
        # raw_gt['vis'] can be used to eliminate heavy occlusion
        raw_gt = raw_gt[raw_gt['conf'] == 1]
        # bbox x, y is the center pixel location
        raw_gt['x'] += raw_gt['w'] / 2
        raw_gt['y'] += raw_gt['h'] / 2
        raw_gt = _remove_field_name(raw_gt, ['conf'])
        gt.append(raw_gt)

    mot_train_seq_det_fn = [os.path.join(data_root, 'train', seq, 'det',
                                 'det.txt') for seq in mot_train_seq]
    det = []
    mask_col = np.ones(10, dtype=bool)
    mask_col[[1, 7, 8, 9]] = False
    for i, det_fn in enumerate(mot_train_seq_det_fn):
        raw_det = np.genfromtxt(det_fn, delimiter=',',
                                dtype=[('frame_num', 'i4'),
                                       ('d1', 'i4'),
                                       ('x', 'f4'),
                                       ('y', 'f4'),
                                       ('w', 'f4'),
                                       ('h', 'f4'),
                                       ('score', 'f4'),
                                       ('d2', 'i4'),
                                       ('d3', 'i4'),
                                       ('d4', 'i4')])
        raw_det = _remove_field_name(raw_det, ['d1', 'd2', 'd3', 'd4'])
        raw_det['x'] += raw_det['w'] / 2
        raw_det['y'] += raw_det['h'] / 2
        # raw_det = np.append([[i + 1]] * raw_det.shape[0],
        #                     raw_det, axis=1)
        det.append(raw_det)
    return det, gt, mot_train_seq


def _remove_field_name(a, name):
    names = [n for n in list(a.dtype.names) if not (n in name)]
    return a[names]

def _print_train_sample_len(train_samples, mot_train_seq):
    for name, vid in zip(mot_train_seq, train_samples):
        print(name, [len(a) for a in vid])



def iou(gt_bbox, det_bboxs):
    """compute IOU between groud truth bbox and detected bboxes

    Args:
        gt_bbox (numpy.array, (4,)): bounding box in format center x, y, w, h
        det_bboxs (numpy.array, (4,n)): bboxes in detection in the same format.
    Returns:
        bbox_sel (np array): randomly selected bbox with iou larger than 0.5. Or empty array if non valid bbox exist.
    """

    # input are some structured array. Convert to regular ndarray
    gt_bbox = np.array(gt_bbox.tolist())
    det_bboxs = det_bboxs.view(det_bboxs.dtype[0]).reshape((-1, 4))

    gt_tl = gt_bbox[0:2] - gt_bbox[2:4]/2
    gt_br = gt_bbox[0:2] + gt_bbox[2:4]/2

    det_tl = det_bboxs[:, 0:2] - det_bboxs[:, 2:4] / 2
    det_br = det_bboxs[:, 0:2] + det_bboxs[:, 2:4] / 2

    # get the overlap rectangle
    overlap_br = np.minimum(gt_br, det_br)
    overlap_tl = np.maximum(gt_tl, det_tl)

    overlap_mask = np.all((overlap_br - overlap_tl) > 0, axis=1)
    if not np.any(overlap_mask):
        return np.empty((0, 4), dtype=np.float32)

    overlap_br = overlap_br[overlap_mask, :]
    overlap_tl = overlap_tl[overlap_mask, :]
    det_br = det_br[overlap_mask, :]
    det_tl = det_tl[overlap_mask, :]

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_all = np.prod(det_br - det_tl, axis=1) + np.prod((gt_br - gt_tl))
    size_intersection = np.prod((overlap_br - overlap_tl), axis=1)
    size_union = size_all - size_intersection

    size_iou = size_intersection / size_union
    iou_mask = size_iou > 0.5
    if not np.any(iou_mask):
        return np.empty((0, 4), dtype=np.float32)

    iou_idx = np.where(overlap_mask)[0][iou_mask]
    bbox_sel = det_bboxs[np.random.permutation(iou_idx)[0], :]

    return bbox_sel


def generate_training_samples(det_all, gt_all, mot_train_seq, min_len=20):
    """Generate training trajectory for a video

    Args:
        det (list): list of detections for each frame. Indexed by zero-based frame number.
            Represented with np array: one detection per row. [bbox, score]
        gt (list): list of trajectories. It is not indexed by the trajectory id.
            One frame per row. [frame_num, track_id, x, y, w, h]

    Return:
        training_trajs (list): a list of generated training trajectories.
            indexed by video first then tracks. Each track is represented by a ndarray of (n, 5) [vid_num frame_num bbox]

    """
    # for each vid:
    #     for each traj in vid:
    #         for each frame in traj:
    #             sample a det

    # note, here we can run Hungarian method for min assignment cost (IOU).
    c = []
    train_samples = []
    for i, gt, det in zip(range(len(gt_all)), gt_all, det_all):
        list_traj_id = np.unique(gt['track_id'])
        vid_train = []
        cc = 0
        for t in list_traj_id:
            gt_traj = gt[gt['track_id'] == t]
            traj_train = np.empty((0, 6))
            for i, f in enumerate(gt_traj):
                gt_bbox = f[['x', 'y', 'w', 'h']]
                det_mask = det['frame_num'] == f['frame_num']
                # There are frames without any detection
                if not np.any(det_mask):
                    cc += 1
                    continue
                det_bboxs = det[det_mask][['x', 'y', 'w', 'h']]  # all bbox in frame
                bbox_sel = iou(gt_bbox, det_bboxs)
                # no detected bbox can be associated to gt
                if len(bbox_sel) == 0:
                    bbox_sel = np.array(gt_bbox.tolist())

                traj_train = np.vstack((traj_train,
                                    np.hstack((i+1, f[0], bbox_sel))))

            if len(traj_train) >= min_len:
                vid_train.append(traj_train)
        c.append(cc)
        train_samples.append(vid_train)

    # print(c)
    # a = np.array([11450, 17833, 6818, 47557, 12318, 9174, 5257])
    # print(c/a)
    _print_train_sample_len(train_samples, mot_train_seq)

    # Note: I decide to drop the vid index and mix all tracks
    train_samples = [traj for vid in train_samples for traj in vid]
    return train_samples


def get_batch(samples, n_traj=64, n_frame=20):
    """
    return a (10, 64, 4) matrix, (n_traj, n_frame, bbox)
        and a (10, 64, 2) matrix for vid and frame indexing (to generate filename)

    :param samples:
    :param n_traj:
    :param n_frame:
    :return:
    """
    idx = np.random.choice(len(samples), size=n_traj)
    trajs = [samples[i] for i in idx]
    ret = np.empty((0, n_frame, 6), dtype=np.float32)
    for t in trajs:
        i = np.random.choice(len(t) - n_frame + 1)
        sel = t[i:i + n_frame, :]
        ret = np.concatenate((ret, sel[None, :, :]))
    ret = np.swapaxes(ret, 0, 1)
    return ret[:, :, 2:6], ret[:, :, 0:2]







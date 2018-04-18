import numpy as np
import os
import pylab as pl
import matplotlib.patches as patches
from videofig import videofig
from scipy.misc import imread
from matplotlib.pyplot import Rectangle, Text


def load_mot16_gt(data_root):
    """Parse MOT16 ground truth data
    Args:
        data_root (str): path MOT16 folder

    Returns:
        gt (list): list of trajectories. One np array for each video.
            One frame per row. [frame_num \ track_id \ x, y, w, h \ class \ vis]
        mot_train_seq (list): video names as indexed by vid_id - 1

    expecting files in the following format:
        <frame_num>,<track_id>,<x>,<y>,<w>,<h>,<conf>,<class>,<vis>
    Note:
        The x,y coord returned for bbox is the center pixel location.
    """
    mot_train_seq = os.listdir(os.path.join(data_root, 'train'))
    mot_train_seq = [f for f in mot_train_seq if not f.startswith('.')]
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

    return gt, mot_train_seq


def load_mot16_det(data_root, mot_train_seq):
    """Parse MOT16 detection data
    Args:
        data_root (str): path MOT16 folder
        mot_train_seq (list): list of video files. This is used to process video
            in the same order as ground truth
    Returns:
        det (list): list of detections. One structured np array for each video.
            one detection per row. [frame_num | x, y, w, h | score]

    expecting files in the following format:
        <frame_num>,<-1>,<x>,<y>,<w>,<h>,<conf>,<-1>,<-1>,<-1>

    Note:
        The x,y coord returned for bbox is the center pixel location.
    """
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
    return det


def _remove_field_name(a, name):
    names = [n for n in list(a.dtype.names) if not (n in name)]
    return a[names]


def _print_train_sample_len(train_samples, mot_train_seq):
    for name, vid in zip(mot_train_seq, train_samples):
        print(name, [len(a) for a in vid])


def iou(gt_bbox, det_bboxs):
    """compute IOU between groud truth bbox and detected bboxes

    Args:
        gt_bbox (numpy.void, (4,)): bounding box in format center x, y, w, h
        det_bboxs (structured numpy.array, (4,n)):
            bboxes in detection in the same format.
    Returns:
        candidate bboxs (np array (n, 4)): all candidate bboxs with iou larger
            than 0.5. Or empty array if non valid bbox exist.
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

    return det_bboxs[iou_idx]


def generate_training_samples(det_all, gt_all, val_id=7, min_len=20):
    """Generate training trajectories for all videos

    Args:
        det (list): list of detections for each video.
            each vid is represented with np array: one detection per row.
            (n, 5) [frame_num, bbox, score]
        gt (list): list of trajectories for each video.
            each trajectory is represented by a np array.
            (n, 6) [frame_num, track_id, x, y, w, h]
        val_id (int): indicate which video is used for validation. one-indexing.
            max is 7.

    Return:
        train_samples (list): a list of generated training trajectories.
            Each traj is represented by a ndarray of (n, 4) [bbox]
            Note: the bbox contain change in center pixel not the absolute location
        img_id_samples (list): same as train_samples but contain image id info
            Each track is represented by a ndarray of
            (n, 3) [vid_num track_id frame_num]

    """
    # currently 458 training trajectories
    # for each vid:
    #     for each traj in vid:
    #         for each frame in traj:
    #             sample a det

    # note, here we can run Hungarian method for min assignment cost (IOU)
    # between all detections and all ground truth in a frame.
    c = []
    train_samples = []
    img_id_train_samples = []
    for i, gt, det in zip(range(len(gt_all)), gt_all, det_all):
        list_traj_id = np.unique(gt['track_id'])
        if i+1 == val_id:
            continue
        cc = 0
        for t in list_traj_id:
            gt_traj = gt[gt['track_id'] == t]
            traj_train = np.empty((0, 4))
            for f in gt_traj:
                #gt_bbox = f[['x', 'y', 'w', 'h']]
                gt_bbox = np.array([f['x'], f['y'], f['w'], f['h']])
                det_mask = det['frame_num'] == f['frame_num']
                # There are frames without any detection
                if not np.any(det_mask):
                    cc += 1
                    bbox_sel = np.array(gt_bbox.tolist())
                else:
                    det_bboxs = det[det_mask][['x', 'y', 'w', 'h']]  # all bbox in frame
                    bbox_candidates = iou(gt_bbox, det_bboxs)
                    bbox_sel = uniform_sample_bbox(bbox_candidates)
                # no detected bbox can be associated to gt
                if len(bbox_sel) == 0:
                    bbox_sel = np.array(gt_bbox.tolist())

                traj_train = np.vstack((traj_train, bbox_sel))
                                    # np.hstack((i+1, f[0], ))))
            if len(traj_train) >= min_len:
                # compute change in center pixel
                traj_train[1:, 0:2] -= traj_train[0:-1, 0:2]
                traj_train[0, 0:2] = 0
                img_id = np.vstack(((i+1) * np.ones(gt_traj['frame_num'].shape),
                                    gt_traj['track_id'],
                                    gt_traj['frame_num'])).astype(np.int)

                train_samples.append(traj_train.astype(np.float32))
                img_id_train_samples.append(img_id.transpose())

        c.append(cc)

    # print(c)
    # a = np.array([11450, 17833, 6818, 47557, 12318, 9174, 5257])
    # print(c/a)
    # _print_train_sample_len(train_samples, mot_train_seq)

    # Note: I decide to drop the vid index and mix all tracks
    return train_samples, img_id_train_samples


def uniform_sample_bbox(bboxs):
    """ Select a bbox from input bboxs. Return an empty array if bboxs is empty.

    Args:
        bboxs (numpy array): (n, 4) where n can be 0

    Returns:
        bbox_sel (numpy array): (1, 4)

    """
    if bboxs.shape[0] == 0:
        return bboxs
    else:
        return bboxs[np.random.choice(bboxs.shape[0]), :]


def generate_external(padded_batch,  lengths, hist_size):
    """
    Helper function for pad_packed_collate
    generate external memory of the given a padded batch for training.
    Args:
        padded_batch(np.array (T, B, F)): T is maximum seq_len in batch.
            B is batch size, F is feature size.
        lengths (list): list of sequence length.
        hist_size (int): external memory history size
    Returns:
        external(np.array (T, B, F, H)): padded external memory.
            used to compute loss

    """
    n_frame, n_batch, n_feature = padded_batch.shape
    ext = np.zeros((n_frame, n_batch, n_feature, hist_size), dtype=np.float32)

    # Get the batch into shape 1, B, F, T)
    padded_batch = np.transpose(padded_batch, (1, 2, 0))[None, :, :, :]
    lengths = np.array(lengths)
    for i in range(n_frame):
        if i < hist_size:
            ext[i, :, :, :i+1] = padded_batch[:, :, :, i::-1]
        else:
            mask = i < lengths
            ext[i, mask, :, :] = padded_batch[:, mask, :, i:i-hist_size:-1]
    return ext


def generate_img_fn(root, mot_train_seq, img_id, train=True):
    """
    Generate all filenames of a given sequence
    Args:
        root (str): data root dir
        mot_train_seq (list): list of video names
        img_id (ndarray): (n, 2)  <vid_num frame_num>
        Train (bool): vid in train dir or test dir

    Returns: (list) list of path to videos. same order as in img_id
    """
    ret = []
    if train:
        sub = 'train'
    else:
        sub = 'test'
    img_id = img_id.tolist()
    ret = [os.path.join(root, sub, mot_train_seq[i[0]-1],
                 'img1', '{0:06d}.jpg'.format(i[2])) for i in img_id]
    return ret


def show_track(idx, data_set):
    """
    visualize a track for sanity check
    Args:
        idx(int): track index 434 total tracks currently
        data_set(MOT16_train_dataset):

    Returns:
        you will see
    """
    root = data_set.root
    gt = data_set.gt
    sample = data_set.train_samples
    img_id = data_set.img_id
    mot_train_seq = data_set.mot_train_seq
    if idx + 1 > len(img_id):
        idx = len(img_id)
    elif idx < 0:
        idx = len(img_id) + idx

    # working with samples
    img_id = img_id[idx]
    vid_id = img_id[0][0] - 1
    track_id = img_id[0][1]
    frame_num = img_id[:, 2]
    sample = sample[idx]

    # working with gt
    gt = gt[vid_id]
    mask_track = gt['track_id'] == track_id
    track_gt = gt[mask_track][['frame_num', 'x', 'y', 'w', 'h']]

    # find intersection. it is a pain in the ass working with structured array!
    frame_num_gt = track_gt['frame_num']
    mask_intersect = np.isin(frame_num_gt, frame_num, assume_unique=True)
    track_gt = track_gt[mask_intersect][['x', 'y', 'w', 'h']]

    # align initial center
    sample[0][0] = track_gt[0][0]
    sample[0][1] = track_gt[0][1]
    # cumsum
    sample[:, 0:2] = np.cumsum(sample[:, 0:2], axis=0)

    print("playing vid {}".format(mot_train_seq[vid_id]))
    img_files = generate_img_fn(root, mot_train_seq, img_id)

    # track_gt[0]
    # tuple(sample[0].tolist())
    def redraw_fn(f, axes):
        img = imread(img_files[f])

        x1, y1, w1, h1 = track_gt[f]
        x2, y2, w2, h2 = tuple(sample[f].tolist())
        x1 = int(x1 - w1/2)
        x2 = int(x2 - w2/2)
        y1 = int(y1 - h1/2)
        y2 = int(y2 - h2/2)
        if not redraw_fn.initialized:
            im = axes.imshow(img, animated=True)
            bb1 = Rectangle((x1, y1), w1, h1,
                           fill=False,  # remove background
                           edgecolor="red",
                            gid='gt')
            bb2 = Rectangle((x2, y2), w2, h2,
                            fill=False,  # remove background
                            edgecolor="blue",
                            gid='sample')

            t1 = axes.text(0, 0, '[{} {}]'.format(x1, y1),
                           bbox=dict(facecolor='red', alpha=0.5))
            t2 = axes.text(300, 0, '[{} {}]'.format(x2, y2),
                           bbox=dict(facecolor='blue', alpha=0.5))

            axes.add_patch(bb1)
            axes.add_patch(bb2)
            redraw_fn.im = im
            redraw_fn.bb1 = bb1
            redraw_fn.bb2 = bb2
            redraw_fn.t1 = t1
            redraw_fn.t2 = t2
            redraw_fn.initialized = True
        else:
            redraw_fn.im.set_array(img)
            redraw_fn.bb1.set_xy((x1, y1))
            redraw_fn.bb1.set_width(w1)
            redraw_fn.bb1.set_height(h1)
            redraw_fn.t1.set_text('[{} {}]'.format(x1, y1))
            redraw_fn.bb2.set_xy((x2, y2))
            redraw_fn.bb2.set_width(w2)
            redraw_fn.bb2.set_height(h2)
            redraw_fn.t2.set_text('[{} {}]'.format(x2, y2))
            # redraw_fn.t2.set_y(y2)

    redraw_fn.initialized = False
    videofig(len(img_files), redraw_fn, play_fps=30)


    # # for f in frame_num:
    # img = None
    # for f in files:
    #     im = pl.imread(f)
    #     if img is None:
    #         img = pl.imshow(im)
    #     else:
    #         img.set_data(im)
    #
    #     img.add_patch(
    #         patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False,
    #                           lw=1, ec=colours[bbox[4] % 32, :]))
    #     ax1.set_adjustable('box-forced')
    #     pl.pause(.1)
    #     pl.draw()

# def generate_external(traj, hist_size):
#     """
#     generate external memory for training. the first (hist_size) frames are
#     padded with zeros.
#     Args:
#         traj(np.array (n_fram, n_traj, n_feature)):
#
#     Returns:
#         external(np.array (n_fram, n_traj, n_feature, hist_size))
#
#     """
#     n_frame, n_batch, n_feature = traj.shape
#     traj = np.rollaxis(traj, 0, 3)  # (64, 4, 20)
#     ret = np.empty((0, n_batch, n_feature, hist_size), dtype=np.float32)
#     for i in range(n_frame):
#         curr = np.zeros((1, n_batch, n_feature, hist_size), dtype=np.float32)
#         if i < hist_size:
#             curr[:, :, :, list(range(i+1))] = traj[None, :, :, list(range(i+1))]
#         else:
#             curr = traj[None, :, :, i-hist_size:i]
#         ret = np.concatenate((ret, curr), axis=0)
#
#     return ret

# def get_batch(samples, n_traj=64, n_frame=20):
#     """
#     return a (10, 64, 4) matrix, (n_frame, n_traj, bbox)
#         and a (10, 64, 2) matrix for vid and frame indexing (to generate filename)
#         This method need to be fixed. It generate sample with replacement.
#
#     Args:
#         samples(list[traj]): return value from generate_training_samples()
#         n_traj: number of traj for training batch (batch number)
#         n_frame: number of time step.
#
#     Returns:
#         training_sample(np.array (n_frame, n_traj, feature_size)):
#         idx_vid_frame (np.array (n_frame, n_traj, 2)): the last dimension contain
#             video and frame number.
#     """
#     idx = np.random.choice(len(samples), size=n_traj)
#     trajs = [samples[i] for i in idx]
#     ret = np.empty((0, n_frame, 6), dtype=np.float32)
#     for t in trajs:
#         i = np.random.choice(len(t) - n_frame + 1)
#         sel = t[i:i + n_frame, :]
#         ret = np.concatenate((ret, sel[None, :, :]))
#     ret = np.swapaxes(ret, 0, 1)
#
#     return ret[:, :, 2:6], ret[:, :, 0:2]




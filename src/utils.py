import numpy as np
import os
import pylab as pl
import matplotlib.patches as patches
from videofig import videofig
from scipy.misc import imread
from matplotlib.pyplot import Rectangle, Text
from sklearn.utils.linear_assignment_ import linear_assignment
import cv2

mot16_train_seq = ['MOT16-13', 'MOT16-11', 'MOT16-10',
                 'MOT16-09', 'MOT16-05', 'MOT16-04', 'MOT16-02']


def load_mot16_gt(data_root):
    """Parse MOT16 ground truth data
    Args:
        data_root (str): path MOT16 folder

    Returns:
        gt (list): list of trajectories. One np array for each video.
            One frame per row. [frame_num \ track_id \ x, y, w, h \ class \ vis]
        mot_train_seq (list): video names

    expecting files in the following format:
        <frame_num>,<track_id>,<x>,<y>,<w>,<h>,<conf>,<class>,<vis>
    Note:
        The x,y coord returned for bbox is the center pixel location.
    """
    # pre-defined mot_train_seq
    mot_train_seq = mot16_train_seq
    sequence_dir_list = [os.path.join(data_root, 'train', seq) for seq in mot_train_seq]
    gt = []
    image_info = []
    for i, sequence_dir in enumerate(sequence_dir_list):
        gt_fn = os.path.join(sequence_dir, 'gt/gt.txt')
        image_dir = os.path.join(sequence_dir, 'img1')
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        raw_gt = np.genfromtxt(gt_fn, delimiter=',', dtype=np.float32)
        raw_gt = raw_gt[raw_gt[:, 6] == 1]
        raw_gt = raw_gt[raw_gt[:, 8] > 0.5]
        seq_info = {
            'frame_num': raw_gt[:, 0],
            'track_id': raw_gt[:, 1],
            'bbox': raw_gt[:, 2:6],
            'vis': raw_gt[:, 8]
        }

        gt.append(seq_info)
        image_info.append(image_filenames)

    return gt, image_info


def load_mot16_det(det_file_list, feat_file_list=None):
    """Parse MOT16 detection data
    Args:
        det_file_list (str)
    Returns:
        det (list): list of detections. One structured np array for each video.
            one detection per row. [frame_num | x, y, w, h | score]

    expecting files in the following format:
        <frame_num>,<-1>,<x>,<y>,<w>,<h>,<conf>,<-1>,<-1>,<-1>

    Note:
        The x,y coord returned for bbox is the center pixel location.
    """

    det = []
    for i, det_fn in enumerate(det_file_list):
        raw_det = np.genfromtxt(det_fn, delimiter=',', dtype=np.float32)
        mask = raw_det[:, 6] > 0.7
        raw_det = raw_det[mask]

        if feat_file_list is not None:
            raw_feat = np.genfromtxt(feat_file_list[i], delimiter=',', dtype=np.float32)
            raw_feat = raw_feat[mask]
        else:
            raw_feat = np.empty((len(raw_det), 4)) * np.nan

        seq_info = {
            'frame_num': raw_det[:, 0],
            'bbox': raw_det[:, 2:6],
            'score': raw_det[:, 6],
            'feat': raw_feat
        }

        det.append(seq_info)

    return det


def iou_vec(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def hungarian_match(gt_bboxes, det_bboxes):
    iou = iou_vec(gt_bboxes, det_bboxes)
    matched_indices = linear_assignment(-iou)
    matches, unmatched_gts, unmatched_dets = [], [], []
    gt_matched, det_matched = [], []

    for g, _ in enumerate(gt_bboxes):
        if g not in matched_indices[:, 0]:
            unmatched_gts.append(g)

    for d, _ in enumerate(det_bboxes):
        if d not in matched_indices[:, 1]:
            unmatched_dets.append(d)

    for g, d in matched_indices:
        if iou[g, d] < 0.2:
            unmatched_gts.append(g)
            unmatched_dets.append(d)
        else:
            matches.append((g, d))
            gt_matched.append(g)
            det_matched.append(d)

    return matches, np.array(gt_matched), np.array(det_matched)


def match_detections(gt_all, det_all):
    """
    Parameters:
        gt_all (list): A list of ndarray groundtruth annotation
        det_all (list): A list of ndarray detections

    Return:
        gt_indices_all (list)
        det_indices_all (list): A list of ndarray indices
    """

    refined_detection = []

    for gt, det in zip(gt_all, det_all):
        frame_num_list = np.unique(gt['frame_num'])

        frame_num_refined = []
        track_id_refined = []
        bbox_refined = []
        score_refined = []
        feat_refined = []

        for t in frame_num_list:
            gt_select = np.where(gt['frame_num'] == t)[0]
            det_select = np.where(det['frame_num'] == t)[0]

            gt_bboxes = gt['bbox'][gt_select].copy()
            det_bboxes = det['bbox'][det_select].copy()
            # convert to (x1 y1 x2 y2)
            gt_bboxes[:, 2:4] += gt_bboxes[:, 0:2]
            det_bboxes[:, 2:4] += det_bboxes[:, 0:2]

            matches, gt_matched, det_matched = hungarian_match(gt_bboxes, det_bboxes)

            if len(gt_matched) > 0:
                gt_indices = gt_select[gt_matched]
                det_indices = det_select[det_matched]
                frame_num_refined.append(gt['frame_num'][gt_indices])
                track_id_refined.append(gt['track_id'][gt_indices])
                bbox_refined.append(det['bbox'][det_indices])
                score_refined.append(det['score'][det_indices])
                feat_refined.append(det['feat'][det_indices])

        seq_info = {
            'frame_num': np.concatenate(frame_num_refined),
            'track_id': np.concatenate(track_id_refined),
            'bbox': np.concatenate(bbox_refined),
            'score': np.concatenate(score_refined),
            'feat': np.concatenate(feat_refined)
        }

        refined_detection.append(seq_info)

    return refined_detection


def ind_select(source, indices, field_name):
    """
    Parameters:
        source: A list of dictionary
        indices: A list of ndarray indicating which rows to select
        field_name: selected field_name of the dictionary
    """

    result = []
    for src, ind in zip(source, indices):
        if src[field_name] is None:
            result.append(None)
        else:
            result.append(src[field_name][ind].copy())

    return result


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


def generate_trainset(det_all, min_len=20):

    motion = []
    appearance = []
    video_id = []
    frame_num = []

    for i, det in enumerate(det_all):
        unique_id = np.unique(det['track_id'])

        for t in unique_id:
            select = np.where(det['track_id'] == t)

            if len(select) > min_len:
                bb = det['bbox'][select].copy()
                # x,y,w,h to cx,cy,w,h
                bb[:, 0:2] += bb[:, 2:4] / 2.0
                motion.append(bb)

                if det['feat'] is not None:
                    f = det['feat'][select].copy()
                    appearance.append(f)

                video_id.append([i] * len(select))
                frame_num.append(det['frame_num'][select])

    return motion, appearance, video_id, frame_num


def generate_training_samples(det_all, gt_all, min_len=20, gt_only=True):
    """Generate training trajectories for all videos

    Args:
        det_all (list): list of detections for each video.
            each vid is represented with np array: one detection per row.
            (n, 5) [frame_num, bbox, score]
        gt_all (list): list of trajectories for each video.
            each trajectory is represented by a np array.
            (n, 6) [frame_num, track_id, x, y, w, h]

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

    # iterate over videos
    for i, gt, det in zip(range(len(gt_all)), gt_all, det_all):
        list_traj_id = np.unique(gt['track_id'])

        cc = 0
        for t in list_traj_id:
            gt_traj = gt[gt['track_id'] == t]
            traj_train = np.empty((0, 4))
            for f in gt_traj:
                #gt_bbox = f[['x', 'y', 'w', 'h']]
                gt_bbox = np.array([f['x'], f['y'], f['w'], f['h']])

                if gt_only:
                    bbox_sel = np.array(gt_bbox.tolist())

                else:
                    det_mask = det['frame_num'] == f['frame_num']
                    # There are frames without any detection
                    if not np.any(det_mask):
                        cc += 1
                        bbox_sel = np.array(gt_bbox.tolist())
                    else:
                        det_bboxs = det[det_mask][['x', 'y', 'w', 'h']]  # all bbox in frame
                        bbox_candidates = iou(gt_bbox, det_bboxs)
                        bbox_sel = uniform_sample_bbox(bbox_candidates)
                    # use gt bbox if no association can be found
                    if len(bbox_sel) == 0:
                        bbox_sel = np.array(gt_bbox.tolist())

                traj_train = np.vstack((traj_train, bbox_sel))
                                    # np.hstack((i+1, f[0], ))))

            # compute bbox displacement
            if len(traj_train) >= min_len:

                #traj_train[1:, 0:2] -= traj_train[0:-1, 0:2]
                #traj_train[0, 0:2] = 0
                traj_train[1:, :] -= traj_train[0:-1, :]
                traj_train[0, :] = 0
                img_id = np.vstack(((i+1) * np.ones(gt_traj['frame_num'].shape),
                                    gt_traj['track_id'],
                                    gt_traj['frame_num'])).astype(np.int)

                train_samples.append(traj_train.astype(np.float32))
                img_id_train_samples.append(img_id.transpose())

        c.append(cc)

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


def generate_external(padded_batch, lengths, hist_size):
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


colours = np.random.rand(32, 3) * 512
colours = [(int(c[0]), int(c[1]), int(c[2])) for c in colours]


def save_to_video(video_handle, img_file, img_size, bboxes, id_list):
    """ Write a given frame to the video handle.
    BBoxes should be in the format (x, y, w, h).

    """

    # automatically handles the case when bboxes = []
    frame = cv2.imread(img_file)
    selected_colors = [colours[int(i) % 32] for i in id_list]

    for (bbox, color, track_id) in zip(bboxes, selected_colors, id_list):
        x, y, w, h = bbox
        x1 = int(x)
        y1 = int(y)
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(track_id), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    frame = cv2.resize(frame, img_size)
    video_handle.write(frame)

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




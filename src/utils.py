import numpy as np
import os
import csv


def load_mot16_train(seq_root):
    """Parse MOT16 training data

    Args:
        seq_root (str): sequence root path e.g. "MOT16/train/MOT16-02/"

    Returns:
        det (list): list of detections for each frame represented with np array. one detection per row. [bbox, score]
        gt (list): list of trajectories. It is not indexed by the trajectory id. One frame per row. [frame_num, track_id, x, y, w, h]
        img_fns (list): file names of images

    expecting files in the following format:
    for ground truth:
        <frame_num>,<track_id>,<x>,<y>,<w>,<h>,<conf>,<class>,<vis>
    for detection:
        <frame_num>,<-1>,<x>,<y>,<w>,<h>,<conf>,<-1>,<-1>,<-1>

    Note:
        The x,y coord returned for bbox is the center pixel location.
        Other conf, class and vis information are removed.
    """

    gt_path = os.path.join(seq_root, "gt", "gt.txt")
    raw_gt = np.genfromtxt(gt_path, delimiter=',', dtype=np.float32)
    gt = []

    # ground truth info is ordered in track id
    max_track = int(np.max(raw_gt[:, 1]))
    consider_gt = raw_gt[:, 6] == 1
    for i in range(1, max_track+1):

        track_id_mask = raw_gt[:, 1] == i

        # remove not considered frames because they are skipped during training
        track_id_mask = track_id_mask & consider_gt
        if not any(track_id_mask):
            continue

        gts = raw_gt[track_id_mask, 0:6]
        gts[:, 2:4] += gts[:, 4:6]/2
        gt.append(gts)

    # detection info are ordered in frame number
    det_path = os.path.join(seq_root, "det", "det.txt")
    img_dir = os.path.join(seq_root, "img1")
    raw_det = np.genfromtxt(det_path, delimiter=',', dtype=np.float32)
    det = []
    img_fns = []

    end_frame = int(np.max(raw_det[:, 0]))

    for i in range(1, end_frame + 1):
        idx_det = raw_det[:, 0] == i

        dets = raw_det[idx_det, :]
        dets[:, 2:4] += dets[:, 4:6] / 2
        #<frame_num>,<-1>,<x>,<y>,<w>,<h>,<conf>,<-1>,<-1>,<-1>
        dets = np.delete(dets, [0,1,7,8,9], axis=1)
        det.append(dets)

        img_fns.append(os.path.join(img_dir, format(i, '06d') + '.jpg'))

    return det, gt, img_fns


def iou(gt_bbox, det_bboxs):
    """compute IOU between groud truth bbox and detected bboxes

    Args:
        gt_bbox (numpy.array, (4,)): bounding box in format center x, y, w, h
        det_bboxs (numpy.array, (4,n)): bboxes in detection in the same format.
    Returns:
        iou_idx (np array): index into detected bboxes with iou larger than 0.5
    """

    gt_tl = gt_bbox[0:2] - gt_bbox[2:4]/2
    gt_br = gt_bbox[0:2] + gt_bbox[2:4]/2

    det_tl = det_bboxs[:, 0:2] - det_bboxs[:, 2:4] / 2
    det_br = det_bboxs[:, 0:2] + det_bboxs[:, 2:4] / 2

    # get the overlap rectangle
    overlap_br = np.minimum(gt_br, det_br)
    overlap_tl = np.maximum(gt_tl, det_tl)

    overlap_mask = np.all((overlap_br - overlap_tl) > 0, axis=1)

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
    iou_idx = np.where(overlap_mask)[0][iou_mask]

    return iou_idx








#=========================================================
def _load_mot(seq_root, train_flag):
    """Backup

    Args:
        seq_root (str): sequence root path e.g. "MOT16/train/MOT16-02/"

    Returns:
        det (list): data[frame][track]{'bbox_det','score'} in np arrays
        gt (list): data[frame][track]{'bbox_det','score'} in np arrays
        img_fns (list):

    expecting files in the following format:
    for ground truth:
        <frame_num>,<track_id>,<x>,<y>,<w>,<h>,<conf>,<class>,<vis>
    for detection:
        <frame_num>,<-1>,<x>,<y>,<w>,<h>,<conf>,<-1>,<-1>,<-1>
    """

    # max_frame is read from seqinfo.ini
    # we may read in other info such as resolution
    max_frame = 0
    with open(os.path.join(seq_root, "seqinfo.ini")) as fd:
        for line in fd:
            if "seqLength" in line:
                max_frame = int(line.rstrip("\n").split("=")[1])

    det_path = os.path.join(seq_root, "det", "det.txt")
    img_dir = os.path.join(seq_root, "img1")
    raw_det = np.genfromtxt(det_path, delimiter=',', dtype=np.float32)
    det = []
    img_fns = []

    if train_flag:
        gt_path = os.path.join(seq_root, "gt", "gt.txt")
        raw_gt = np.genfromtxt(gt_path, delimiter=',', dtype=np.float32)
    gt = []

    # either use end_frame of max_frame
    end_frame = int(np.max(raw_det[:, 0]))
    # sanity check
    assert (end_frame == max_frame), "end_frame, max_frame conflict"

    for i in range(1, end_frame+1):

        idx_det = raw_det[:, 0] == i
        bbox_det = raw_det[idx_det, 2:6]
        bbox_det[:, 0:2] += bbox_det[:, 2:4]/2
        scores_det = raw_det[idx_det, 6]

        dets = []
        for bb, s in zip(bbox_det, scores_det):
            dets.append({'bbox_det': (bb[0], bb[1], bb[2], bb[3]), 'score': s})
        det.append(dets)

        if train_flag:
            idx_gt = raw_gt[:, 0] == i
            bbox_gt = raw_gt[idx_gt, 2:6]
            bbox_gt[:, 0:2] += bbox_gt[:, 2:4]/2
            id_gt = raw_gt[idx_gt, 1]
            consider_gt = raw_gt[idx_gt, 6] == 1
            class_gt = raw_gt[idx_gt, 7]
            vis_gt = raw_gt[idx_gt, 8]
            gts = []

            for bb, id_, flag, c, vis in zip(bbox_gt, id_gt, consider_gt, class_gt, vis_gt):
                gts.append({'bbox_gt': (bb[0], bb[1], bb[2], bb[3]), 'id': int(id_), 'flag': flag, 'class': int(c), 'vis': vis})
            gt.append(gts)

        img_fns.append(os.path.join(img_dir, format(i, '06d')+'.jpg'))

    return det, gt, img_fns

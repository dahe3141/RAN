import os
import numpy as np
import cv2
import torch
from utils import save_to_video
from models import RAN
from trackers import RANTracker


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information
        (1) image filenames
        (2) ground truth annotation (if available)
        (3) detections (if available)
        (4) features (if available)

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.
    Returns
    -------
    Dict
        A dictionary of the following sequence information:
        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.
    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections, groundtruth = None, None
    if detection_file is not None:
        detections = np.loadtxt(detection_file, delimiter=',')

    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')
        groundtruth = groundtruth[groundtruth[:, 6] == 1]

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        fps = int(info_dict["frameRate"])
    else:
        fps = 30

    feature_dim = None
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "fps": fps
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.
    BBox is in the format <center_x, center_y, w, h>

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.
    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    bbox_list = []
    conf_list = []
    for row in detection_mat[mask]:
        bbox, confidence = row[2:6], row[6]
        bbox[0:2] += bbox[2:4] / 2.0
        if bbox[3] < min_height:
            continue
        bbox_list.append(bbox.astype(np.float32))
        conf_list.append(confidence)

    return bbox_list, conf_list


if __name__ == '__main__':

    seq_info = gather_sequence_info('/scratch0/MOT/MOT16/train/MOT16-02', '/scratch0/MOT/MOT16/external/MOT16-02_det.txt')

    video = cv2.VideoWriter('../results/video_gt.avi', cv2.VideoWriter_fourcc(*"MJPG"), seq_info['fps'], (640, 480))

    model_path = "../results/models/RAN.pth"
    # load model
    checkpoint = torch.load(model_path)
    RAN_motion = RAN(input_size=4, hidden_size=32, history_size=10, drop_rate=0.5)
    RAN_feat = RAN(input_size=4, hidden_size=32, history_size=10, drop_rate=0.5)

    RAN_motion.load_state_dict(checkpoint['RAN_motion'])
    RAN_feat.load_state_dict(checkpoint['RAN_feat'])
    RAN_motion = RAN_motion.cuda()
    RAN_feat = RAN_feat.cuda()

    RAN_motion.eval()
    RAN_feat.eval()
    tracker = RANTracker(RAN_motion, feat_model=None)

    for frame_idx in seq_info['image_filenames'].keys():
        bboxes, confs = create_detections(seq_info['groundtruth'], frame_idx)
        #bboxes, confs = create_detections(seq_info['detections'], frame_idx)

        # filter detections
        bboxes = [bbox for (bbox, conf) in zip(bboxes, confs) if conf > 0.7]

        # run tracker
        tracker.predict()
        tracker.update(bboxes)

        track_list, id_list = tracker.get_tracks()

        # write to video
        filename = seq_info['image_filenames'][frame_idx]
        save_to_video(video, filename, (640, 480), track_list, id_list)

    video.release()
import cv2
from utils import load_mot16_gt, load_mot16_det, save_to_video, match_detections, ind_select
import os
import numpy as np
import time
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import load_mot16_gt

file_path = '/scratch0/MOT/MOT16/'


def write_video(video_fn, image_filenames, frame_num, bboxes, track_id=None):
    video_handle = cv2.VideoWriter(video_fn,
                                   cv2.VideoWriter_fourcc(*"MJPG"),
                                   30,
                                   (640, 480))
    unique_frame_num = np.unique(frame_num)

    if track_id is None:
        track_id = np.zeros(len(bboxes), dtype=int)

    for n in unique_frame_num:
        bb_list = bboxes[frame_num == n]
        id_list = track_id[frame_num == n]
        save_to_video(video_handle, image_filenames[n], (640, 480), bb_list, id_list)

    video_handle.release()


mot16_train_seq = ['MOT16-13', 'MOT16-11', 'MOT16-10',
                 'MOT16-09', 'MOT16-05', 'MOT16-04', 'MOT16-02']
det_seq = [os.path.expanduser('~/Projects/Datasets/MOT16/external/{}_det.txt'.format(x))
           for x in mot16_train_seq]
feat_seq = [os.path.expanduser('~/Projects/Datasets/MOT16/external/{}_feat.txt'.format(x))
           for x in mot16_train_seq]
gt, image_info = load_mot16_gt(os.path.expanduser('~/Projects/Datasets/MOT16'))
det = load_mot16_det(det_seq, feat_seq)
r = match_detections(gt, det)

#write_video('../results/video_gt.avi', image_info[-1], gt[-1]['frame_num'], gt[-1]['bbox'], gt[-1]['track_id'])
#write_video('../results/video_det.avi', image_info[-1], det[-1]['frame_num'], det[-1]['bbox'])
write_video('../results/video_refine.avi', image_info[-1], r[-1]['frame_num'], r[-1]['bbox'], r[-1]['track_id'])

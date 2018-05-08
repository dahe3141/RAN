import os
import torch
import numpy as np
import cv2
from models import RAN
from collections import deque
from torch.autograd import Variable

from dataset import MOT16_train_dataset
from utils import save_to_video, show_track
from sort import convert_bbox_to_z, convert_x_to_bbox, KalmanBoxTracker

use_cuda = torch.cuda.is_available()


def to_var(array):
    tensor = torch.from_numpy(array)
    var = Variable(tensor)
    if use_cuda:
        var = var.cuda()
    return var


def to_np(var):
    tensor = var.data.cpu()
    return tensor.numpy()


def sanity(idx):
    mot16_root_dir = "/scratch0/MOT/MOT16"

    train_dataset = MOT16_train_dataset(mot16_root_dir)

    show_track(idx, train_dataset)


def test_Kalman(idx):
    dataroot = os.path.expanduser('~/Projects/Datasets/MOT16')
    detroot = os.path.expanduser('~/Projects/Datasets/MOT16/external')
    video_path = '../results/visualization/sample_track_Kalman.avi'

    video_handle = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))

    dataset = MOT16_train_dataset(dataroot, detroot)

    # sample a track from training data
    bbox_data = dataset.bbox[idx]
    frame_num = dataset.frame_num[idx]

    video_id = dataset.video_id[idx][0]
    image_names = dataset.image_filenames[video_id]

    bbox_gt = bbox_data.copy()
    bbox_gt[:, 0:2] -= bbox_gt[:, 2:4] / 2.0

    count = 0
    for f_num in np.arange(frame_num.min(), frame_num.max()+1):
        if np.any(frame_num == f_num):
            idx = np.where(frame_num == f_num)[0][0]
            gt = bbox_gt[idx].copy()
            bbox = bbox_gt[idx].copy()
            bbox[2:4] += bbox[0:2]
            if count == 0:
                track = KalmanBoxTracker(bbox)
                count += 1
            else:
                track.update(bbox)

                bbox_pred = track.predict()[0]
                bbox_pred[2:4] -= bbox[0:2]

                save_to_video(video_handle,
                              image_names[f_num],
                              (640, 480), [gt, bbox_pred], [(0, 255, 0), (0, 0, 255)])

        else:
            save_to_video(video_handle, image_names[f_num], (640, 480), [], [])

    video_handle.release()


def test(idx):
    dataroot = '/scratch0/MOT/MOT16'
    detroot = '/scratch0/MOT/MOT16/external'
    model_path = '../results/models/RAN.pth'
    video_path = '../results/visualization/sample_track.avi'

    video_handle = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))

    dataset = MOT16_train_dataset(dataroot, detroot)

    # load model
    checkpoint = torch.load(model_path)
    RAN_motion = RAN(input_size=dataset.motion_dim, hidden_size=32, history_size=10, drop_rate=0.5)
    RAN_feat = RAN(input_size=dataset.feat_dim, hidden_size=32, history_size=10, drop_rate=0.5)

    RAN_motion.load_state_dict(checkpoint['RAN_motion'])
    RAN_feat.load_state_dict(checkpoint['RAN_feat'])
    RAN_motion = RAN_motion.cuda()
    RAN_feat = RAN_feat.cuda()

    RAN_motion.eval()
    RAN_feat.eval()

    memory_size = 10
    input_size = 4

    hidden = RAN_motion.init_hidden(batch_size=1)
    external = deque([np.zeros(input_size, dtype=np.float32) for _ in range(memory_size)], maxlen=memory_size)

    # sample a track from training data
    bbox_data = dataset.bbox[idx]
    bbox_motion = dataset.motion[idx]
    frame_num = dataset.frame_num[idx]

    video_id = dataset.video_id[idx][0]
    image_names = dataset.image_filenames[video_id]

    bbox_gt = bbox_data.copy()
    bbox_gt[:, 0:2] -= bbox_gt[:, 2:4] / 2.0

    for f_num in np.arange(frame_num.min(), frame_num.max() + 1):
        if np.any(frame_num == f_num):
            idx = np.where(frame_num == f_num)[0][0]
            gt = bbox_gt[idx].copy()
            motion = bbox_motion[idx].copy()
            bbox = bbox_data[idx].copy()

            external.appendleft(motion)
            motion_var = to_var(motion).view(1, 1, -1)
            alpha, sigma, hidden = RAN_motion(motion_var, hidden)
            # linear combination of history
            alpha_np = to_np(alpha.squeeze())
            motion_pred = np.matmul(alpha_np, np.array(external))

            bbox_pred = bbox + motion_pred
            bbox_pred[0:2] -= bbox_pred[2:4] / 2.0

            save_to_video(video_handle, image_names[f_num], (640, 480), [gt, bbox_pred], [(0, 255, 0), (0, 0, 255)])

        else:
            save_to_video(video_handle, image_names[f_num], (640, 480), [], [])

    # for bbox, motion, gt, frame in zip(bbox_data, bbox_motion, bbox_gt, frame_num):
    #
    #     external.appendleft(motion)
    #     motion_var = to_var(motion).view(1, 1, -1)
    #     alpha, sigma, hidden = RAN_motion(motion_var, hidden)
    #     # linear combination of history
    #     alpha_np = to_np(alpha.squeeze())
    #     motion_pred = np.matmul(alpha_np, np.array(external))
    #
    #     bbox_pred = bbox + motion_pred
    #     bbox_pred[0:2] -= bbox_pred[2:4] / 2.0
    #
    #     save_to_video(video_handle, image_names[frame], (640, 480), [gt, bbox_pred], [(0,255,0), (0,0,255)])

    video_handle.release()


if __name__ == '__main__':
    test(300)
    #test_Kalman(300)
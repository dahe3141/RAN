import os
import numpy as np
import time
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import load_mot16_gt

file_path = '/scratch0/MOT/MOT16/'

gt, train_seq = load_mot16_gt(file_path)

colours = np.random.rand(32, 3)
plt.ion()
fig = plt.figure(figsize=(16, 12))


for num in range(0, 50):
    fn = os.path.join(file_path, 'train', 'MOT16-13', 'img1', '{:06d}.jpg'.format(num+1))

    im = io.imread(fn)
    ax1 = fig.add_subplot(111, aspect='equal')
    ax1.imshow(im)

    tracks = gt[0][gt[0]['frame_num'] == num]
    mask = tracks['track_id'] == 8
    bboxes = tracks[['x', 'y', 'w', 'h', 'track_id', 'class']][mask]

    # select_bboxes = bboxes[bboxes[:, 4] == 7]


    for bbox in bboxes:
        bbox = np.array(bbox.tolist(), dtype=np.int32)
        bbox[0:2] -= bbox[2:4] // 2
        ax1.add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, lw=1, ec=colours[bbox[4]%32,:]))
        ax1.set_adjustable('box-forced')
        print(bbox[5])


    fig.canvas.flush_events()
    plt.draw()
    ax1.cla()
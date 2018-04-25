from PIL import Image
import os
import numpy as np
from videofig import videofig
from scipy.misc import imread
from matplotlib.pyplot import Rectangle
from torch.utils.data import Dataset
import configparser

class MOT16_Dataset_vgg(Dataset):
    def __init__(self, data_root, train=True, transform=None,
                 th_vis=0.9, th_track=20, consider_only=False):
        self.root = os.path.expanduser(data_root)
        self.train = train  # no use for now
        self.transform = transform
        self.vid_seq = ['MOT16-13', 'MOT16-11', 'MOT16-10',
                        'MOT16-09', 'MOT16-05', 'MOT16-04', 'MOT16-02']
        # headers_in = ['f', 't', 'x', 'y', 'w', 'h', 'c', 'class', 'vis']
        f, t, x, y, w, h, csd, cls, vis, vid =\
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        label_current = 1
        label_count = [0]
        data = np.empty((0, 10))
        label = np.empty(0, dtype=int)
        for i in range(len(self.vid_seq)):
            gt_dir = os.path.join(self.root, 'train',
                                  self.vid_seq[i], 'gt', 'gt.txt')
            gt = np.loadtxt(gt_dir, delimiter=',')
            seq_info_dir = os.path.join(self.root, 'train',
                                  self.vid_seq[i], 'seqinfo.ini')
            config = configparser.ConfigParser()
            config.read(seq_info_dir)
            max_w = int(config['Sequence']["imWidth"])
            max_h = int(config['Sequence']["imHeight"])
            # vis threshold
            gt = gt[gt[:, vis] > th_vis, :]
            # consider threshold
            if consider_only:
                gt = gt[gt[:, csd] > 0]
            # video indexing
            vid_column = np.ones(gt.shape[0]) * (i + 1)
            # create labels
            label_column = np.zeros(gt.shape[0], dtype=int)
            tracks = np.unique(gt[:, t])
            for ti in tracks:
                mask_t = gt[:, t] == ti
                if sum(mask_t) < th_track:
                    continue
                else:
                    label_column[mask_t] = label_current
                    label_current += 1
            # remove all invalid labels
            mask_t = label_column != 0
            temp = np.concatenate((gt, vid_column[:, None]), axis=1)
            temp = temp[mask_t, :]
            label_column = label_column[mask_t]

            # make w, h to be the lower right point, and filter out invalid bbox
            temp[:, [w, h]] += temp[:, [x, y]]
            temp[temp[:, x] < 0, x] = 0
            temp[temp[:, y] < 0, y] = 0
            temp[temp[:, w] > max_w, w] = max_w
            temp[temp[:, h] > max_h, h] = max_h
            # accumulate data
            data = np.concatenate((data, temp), axis=0)
            label = np.concatenate((label, label_column), axis=0)
            label_count.append(label_current - label_count[i])

        # generate filenames
        sub = 'train'
        fn = [os.path.join(self.root, sub, self.vid_seq[row[1] - 1], 'img1',
                           '{0:06d}.jpg'.format(row[0]))
              for row in data[:, [f, vid]].astype(np.int)]

        # save some relevant info
        self.data = data
        self.label = label
        self.fn = np.array(fn)
        self.len = self.data.shape[0]
        print("Dataset created with {} samples, {} tracks"
              .format(self.len, label_current))
        # mean, std = self.compute_mean_std()
        # print(mean, std)
        # self.show_track(50)
        pass
        # select a track
        # mask_track = self.data[self.headers_label] == 50
        # mask_track = mask_track.squeeze()
        # img_files = self.fn[mask_track]
        # bbox = self.data[self.headers_bbox].loc[
        #     mask_track, ['x', 'y', 'w', 'h']]
        #

        # for i, f in enumerate(self.fn):
        #     bbox = tuple(self.data.iloc[i, 4:8])
        #     img = Image.open(f).resize((224, 224), Image.LANCZOS, box=bbox)
        #
        #     pass

        # keep aspect ratio
        # w, h = im.size
        # ar = w / h
        # h_new = math.sqrt(224 * 224 / ar)
        # w_new = h_new * ar

    def __getitem__(self, idx):
        fn = self.fn[idx]
        bbox = tuple(self.data[idx, 2:6])
        img = Image.open(fn).resize((224, 224), Image.LANCZOS, box=bbox)
        label = self.label[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.len

    def compute_mean_std(self):
        per = int(self.len / 100)
        r1 = r2 = np.zeros((1, 3))
        for idx in range(self.len):
            img, _ = self.__getitem__(idx)
            if idx % per == 0:
                print(idx/self.len)
            img = np.array(img).reshape((-1, 3)) / 255
            r1 += np.mean(img, axis=0)
            r2 += np.mean(img**2, axis=0)
        mean = r1 / self.len
        std = np.sqrt(r2 / self.len - mean**2)
        return mean, std

    # def _generate_fn(self):
    #     sub = 'train'
    #     ret = [os.path.join(root, sub, self.vid_seq[row['v']-1],
    #                  'img1', '{0:06d}.jpg'.format(row['f']))
    #            for idx, row in self.data[:, []].iterrows()]
    #     return np.array(ret)

    def show_track(self, idx):
        mask_track = self.label == idx
        img_files = self.fn[mask_track]
        bbox = self.data[mask_track, 2:6]
        frames = self.data[mask_track, 0]
        viss = self.data[mask_track, 8]
        def redraw_fn(f, axes):
            img = imread(img_files[f])
            x, y, w, h = tuple(bbox[f, :])
            w = w - x
            h = h - y
            frame = frames[f]
            vis = viss[f]
            if not redraw_fn.initialized:
                im = axes.imshow(img, animated=True)
                bb = Rectangle((x, y), w, h,
                               fill=False,  # remove background
                               edgecolor="red")

                t1 = axes.text(0, 0, '[f:{} vis:{:1.2f}]'.format(frame, vis),
                               bbox=dict(facecolor='red', alpha=0.5))

                axes.add_patch(bb)
                redraw_fn.im = im
                redraw_fn.bb1 = bb
                redraw_fn.t1 = t1
                redraw_fn.initialized = True
            else:
                redraw_fn.im.set_array(img)
                redraw_fn.bb1.set_xy((x, y))
                redraw_fn.bb1.set_width(w)
                redraw_fn.bb1.set_height(h)
                redraw_fn.t1.set_text('[f:{} vis:{:1.2f}]'.format(frame, vis))

        redraw_fn.initialized = False
        videofig(len(img_files), redraw_fn, play_fps=30)


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.pardir, "Data", "MOT16"))
    a = MOT16_Dataset_vgg(root)
    im, label = a.__getitem__(100)
    pass


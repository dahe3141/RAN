import torch
from torch.autograd import Variable
import numpy as np
import math
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from collections import deque

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


class TrackState(object):
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Detection(object):
    def __init__(self, bbox=None, feat=None):
        """
            bbox in (x, y, w, h) format
        """
        self.bbox = self._centering(bbox)
        self.feat = feat

    def _centering(self, bbox):
        if bbox is not None:
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            w = bbox[2]
            h = bbox[3]
            return np.array([cx, cy, w, h])
        else:
            return None


class _RANTrackObject(object):
    def __init__(self, state, ran_model):
        self.inactive = False

        if ran_model is None:
            self.inactive = True
            return

        memory_size = ran_model.history_size

        state_dim = ran_model.input_size
        self.ran_model = ran_model
        self.hidden = ran_model.init_hidden(batch_size=1)

        # external memory
        self.external = deque([np.zeros(state_dim, dtype=np.float32) for _ in range(memory_size)],
                                     maxlen=memory_size)
        self.update(state)

    def inactivate(self):
        self.inactive = True

    def update(self, state):
        if self.inactive:
            return
        self.external.appendleft(state)
        self.state = to_var(state).view(1, 1, -1)

    def predict(self):
        if self.inactive:
            return
        alpha, var, self.hidden = self.ran_model(self.state, self.hidden)
        alpha = to_np(alpha.squeeze())
        self.mu = np.matmul(alpha, np.array(self.external))
        self.var = to_np(var.squeeze())

    def similarity(self, state):
        if self.inactive:
            return 0

        diff2 = (state - self.mu) ** 2
        M = (diff2 / self.var).sum()
        log_scale = np.log(self.var).sum()
        constant = math.log(2 * math.pi) * len(state)

        return -0.5 * (constant + log_scale + M)


class RANTrack(object):
    def __init__(self, track_id, detection, motion_model=None, feat_model=None, n_init=3, max_age=10):
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.track_status = TrackState.Tentative

        self._max_age = max_age
        self._n_init = n_init

        # detection.bbox is always not None
        self.state_bbox = detection.bbox

        self.track_motion = _RANTrackObject(detection.bbox - self.state_bbox, motion_model)
        self.track_feat = _RANTrackObject(detection.feat, feat_model)

    def update(self, detection):
        self.hits += 1
        self.time_since_update = 0

        if self.track_status == TrackState.Tentative and self.hits >= self._n_init:
            self.track_status = TrackState.Confirmed

        bbox_diff = detection.bbox - self.state_bbox
        # save associated bbox
        self.state_bbox = detection.bbox

        self.track_motion.update(bbox_diff)
        self.track_feat.update(detection.feat)

    def predict(self):
        self.age += 1
        self.time_since_update += 1

        self.track_motion.predict()
        self.track_feat.predict()

    def mark_missed(self):
        if self.track_status == TrackState.Tentative:
            self.track_status = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.track_status = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.track_status == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.track_status == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.track_status == TrackState.Deleted

    def similarity(self, detection):
        """
        Computes similarity between the RANTrack and a detection
        """
        bbox_diff = detection.bbox - self.state_bbox

        return self.track_motion.similarity(bbox_diff) + self.track_feat.similarity(detection.feat)


class RANTracker(object):
    def __init__(self, motion_model, feat_model, max_age=10):
        """
        Parameters:
            motion_model
            feat_model
            max_age

        Note:
            Inactivate motion or appearance model by passing 'None'
        """
        self.motion_model = motion_model
        self.feat_model = feat_model

        self.max_age = max_age
        self.min_similarity = -2e4

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Update alpha, sigma, and hidden states for all the tracks
        using the RAN model
        """
        for track in self.tracks:
            track.predict()

    def update(self, detections):

        # run matching
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # update tracks
        for detection_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])

        # mark missed tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # initiate new tracks
        for detection_idx in unmatched_detections:
            self._init_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def get_tracks(self):
        bbox_list = []
        id_list = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox_list.append(track.state_bbox)
            id_list.append(track.track_id)

        return bbox_list, id_list

    def _init_track(self, detection):
        self.tracks.append(RANTrack(self._next_id,
                                    detection,
                                    self.motion_model,
                                    self.feat_model,
                                    max_age=self.max_age))
        self._next_id += 1

    def _match(self, detections):
        """


        Returns
        ------
        * [List] Indices of matched tracks and detections
        * [List] Indices of unmatched tracks
        * [List] Indices of unmatched detections
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], np.arange(len(self.tracks)), np.arange(len(detections))

        sim_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                sim_matrix[d, t] = trk.similarity(det)

        matched_indices = linear_assignment(-sim_matrix)

        matches, unmatched_tracks, unmatched_detections = [], [], []

        for d, _ in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        for t, _ in enumerate(self.tracks):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)

        for d, t in matched_indices:
            if sim_matrix[d, t] < self.min_similarity:
                unmatched_tracks.append(t)
                unmatched_detections.append(d)
            else:
                matches.append((d, t))
        return matches, unmatched_tracks, unmatched_detections


if __name__ == '__main__':
    from models import RAN

    model_save_prefix = "/scratch0/RAN/trained_model/ran"

    # load model
    ran = RAN(input_size=4,
              hidden_size=32,
              history_size=10,
              drop_rate=0.5)
    ran = ran.cuda()
    ran.eval()

    bbox1_1 = np.array([500, 500, 40, 50], dtype=np.float32)
    bbox1_2 = np.array([100, 200, 60, 60], dtype=np.float32)
    bbox1_3 = np.array([400, 300, 70, 70], dtype=np.float32)
    bbox1_4 = np.array([200, 100, 80, 80], dtype=np.float32)

    bbox2_1 = np.array([512, 490, 40, 50], dtype=np.float32)
    bbox2_2 = np.array([400, 330, 70, 75], dtype=np.float32)
    bbox2_3 = np.array([110, 198, 65, 65], dtype=np.float32)
    bbox2_4 = np.array([200, 120, 85, 85], dtype=np.float32)
    bbox2_5 = np.array([100, 100, 45, 45], dtype=np.float32)

    # gt for matching:
    # 0->0, 1->2, 2->1, 3->3, []->4

    tracker = RANTracker(ran)
    tracker.predict()
    tracker.update([bbox1_1, bbox1_2, bbox1_3, bbox1_4])
    tracker.predict()
    tracker.update([bbox2_1, bbox2_2, bbox2_3, bbox2_4, bbox2_5])
    print('Hi')
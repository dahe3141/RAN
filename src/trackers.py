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


class RANTrack(object):
    def __init__(self, bbox, track_id, ran_model, feature=None, n_init=3, max_age=10):
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative

        self._max_age = max_age
        self._n_init = n_init

        self.prev_bbox = bbox

        # RAN include:
        #     (1)RNN hidden states
        #     (2)alpha, sigma
        #     (3)external memory
        #
        # mean of the AR model will be estimated through linear combination

        memory_size = ran_model.history_size
        input_size = ran_model.input_size

        self.model = ran_model
        self.h_motion = ran_model.init_hidden(batch_size=1)
        self.h_feature = 0

        # RAN outputs
        self.alpha_motion = np.zeros(memory_size, dtype=np.float32)
        self.sigma_motion = np.ones(input_size, dtype=np.float32)

        # predicted mean vector from the AR model
        self.mu_motion = np.zeros(input_size, dtype=np.float32)

        # external memory
        self.external_motion = deque([np.zeros(input_size, dtype=np.float32) for _ in range(memory_size)], maxlen=memory_size)
        self.external_feature = deque([np.zeros(input_size, dtype=np.float32) for _ in range(memory_size)], maxlen=memory_size)

        self.update(bbox, feature)

    def update(self, bbox, feature=None):
        """
        compute bbox_diff and external memory using the associated detection
        """
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # compute bbox_diff
        bbox_diff = bbox - self.prev_bbox
        self.prev_bbox = bbox

        # add bbox_diff and feature to external memory
        self.external_motion.appendleft(bbox_diff)
        self.external_feature.appendleft(feature)

        self.bbox_diff = to_var(bbox_diff).view(1, 1, -1)

        if feature is not None:
            self.feature = to_var(feature).view(1, 1, -1)
        else:
            self.feature = None

    def predict(self):
        self.age += 1
        self.time_since_update += 1

        # obtain h, alpha, sigma using RAN
        alpha_motion, sigma_motion, self.h_motion = self.model(self.bbox_diff, self.h_motion)

        # obtain mu using alpha and external memory
        alpha_motion = to_np(alpha_motion.squeeze())
        self.mu_motion = np.matmul(alpha_motion, np.array(self.external_motion))

        self.sigma_motion = to_np(sigma_motion.squeeze())

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def similarity(self, bbox, feature=None):
        """
        Computes similarity between the RANTrack and a detection
        """
        # TODO: similarity should handle feature
        # compute log probability
        diff2 = (self.mu_motion - (bbox - self.prev_bbox)) ** 2
        M = (diff2 / self.sigma_motion).sum()
        log_scale = np.log(self.sigma_motion).sum()

        constant = math.log(2 * math.pi) * len(bbox)

        return -0.5 * (constant + log_scale + M)


class RANTracker(object):
    def __init__(self, ran_model, max_age=30, memory_size=10):
        self.ran_model = ran_model

        self.max_age = max_age
        self.memory_size = memory_size
        self.min_similarity = -500

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Update alpha, sigma, and hidden states for all the tracks
        using the RAN model
        """
        for track in self.tracks:
            track.predict()

    def update(self, bboxes, features=None):

        # run matching
        matches, unmatched_tracks, unmatched_detections = self._match(bboxes)

        # update tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(bboxes[detection_idx])

        # mark missed tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # initiate new tracks
        for detection_idx in unmatched_detections:
            self._init_track(bboxes[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _init_track(self, bbox, feature=None):
        self.tracks.append(RANTrack(bbox, self._next_id, self.ran_model, feature))
        self._next_id += 1

    def _match(self, detections, features=None):
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
    from models import RAN, load_model

    model_save_prefix = "/scratch0/RAN/trained_model/ran"

    # load model
    ran = RAN(input_size=4,
              hidden_size=32,
              history_size=10,
              drop_rate=0.5,
              save_path=model_save_prefix)
    load_model(ran)
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
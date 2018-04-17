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
    def __init__(self, bbox, track_id, n_init, max_age, ran_model, feature=None):
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative

        self._max_age = max_age
        self._n_init = n_init

        # TODO: save observed bbox and estimate bbox offset

        # RAN include:
        #     (1)RNN hidden states
        #     (2)alpha, sigma
        #     (3)external memory
        #
        # mean of the AR model will be estimated through linear combination

        memory_size = ran_model.history_size
        input_size = ran_model.input_size

        self.model = ran_model
        self.h_bbox = ran_model.init_hidden(batch_size=1)
        self.h_feature = 0

        # RAN outputs
        self.alpha_bbox = np.zeros(memory_size, dtype=np.float32)
        self.sigma_bbox = np.ones(input_size, dtype=np.float32)

        # predicted mean vector from the AR model
        self.mu_bbox = np.zeros(input_size, dtype=np.float32)

        # external memory
        self.external_bbox = deque([np.zeros(input_size, dtype=np.float32) for _ in range(memory_size)], maxlen=memory_size)
        self.external_feature = deque([np.zeros(input_size, dtype=np.float32) for _ in range(memory_size)], maxlen=memory_size)

        self.update(bbox, feature)

    def update(self, bbox, feature=None):
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # add associated bbox and feature to external memory
        self.external_bbox.append(bbox)
        self.external_feature.append(feature)

        self.bbox = to_var(bbox).view(1, 1, -1)

        if feature is not None:
            self.feature = to_var(feature).view(1, 1, -1)
        else:
            self.feature = None

    def predict(self):
        self.age += 1
        self.time_since_update += 1

        # obtain h, alpha, sigma using RAN
        alpha_bbox, sigma_bbox, self.h_bbox = self.model(self.bbox, self.h_bbox)

        # obtain mu using alpha and external memory
        alpha_bbox = to_np(alpha_bbox.squeeze())
        self.mu_bbox = np.matmul(alpha_bbox, np.array(self.external_bbox))

        self.sigma_bbox = to_np(sigma_bbox.squeeze())

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
        Computes similarity between the RANTrack and the new detection
        """

        # compute log probability
        diff2 = torch.pow(self.mu_bbox - bbox, 2)
        M = (diff2 / self.sigma_bbox).sum()
        log_scale = self.sigma_bbox.log().sum()

        constant = math.log(2 * math.pi) * len(bbox)

        return -0.5 * (constant + log_scale + M)


class RANTracker(object):
    def __init__(self, max_age=30, memory_size=10):
        self.max_age = max_age
        self.memory_size = memory_size
        self.min_similarity = 0

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
        # update tracks
        # mark missed tracks
        # initiate new tracks
        pass

    def _init_track(self, bbox, feature=None):
        self.tracks.append(RANTrack(bbox, self._next_id, feature))
        self._next_id += 1

    def _match(self, detections, features=None):
        # TODO: similarity should handle feature
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4))

        sim_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)

        for d, det in detections:
            for t, trk in self.tracks:
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
    ran = RAN(input_size=4, hidden_size=32, history_size=10, drop_rate=0.5).cuda()

    bbox = np.array([2, 3, 40, 50], dtype=np.float32)
    track_id = 300
    n = 3
    max_age = 30
    track = RANTrack(bbox, track_id, n, max_age, ran)
    track.predict()
    print(track.mu_bbox)
    print(track.sigma_bbox)
    track.predict()
    print(track.mu_bbox)
    print(track.sigma_bbox)
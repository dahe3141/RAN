import torch
from torch.autograd import Variable
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from collections import deque
use_cuda = torch.cuda.is_available()


class RANTrack(object):
    def __init__(self, bbox, track_id, mem_size=10, feature=None):
        self.bbox = bbox
        self.feature = feature
        self.track_id = track_id

        # RNN state vectors
        self.h_bbox = 0
        self.h_feature = 0

        # RAN outputs
        self.alpha_bbox = np.zeros(mem_size, dtype=np.float32)
        self.sigma_bbox = np.ones(4, dtype=np.float32)

        self.mu_bbox = np.zeros(4, dtype=np.float32)

        # initialize external memory
        self.external_bbox = deque([np.zeros(4, dtype=np.float32) for _ in range(mem_size)], maxlen=mem_size)
        self.external_feature = deque([np.zeros(4, dtype=np.float32) for _ in range(mem_size)], maxlen=mem_size)

    def update(self, bbox, feature=None):
        # add bbox, feature to external memory
        self.external_bbox.append(bbox)
        self.external_feature.append(feature)

        # TODO: consider transform into Variable here
        self.bbox = bbox
        self.feature = feature

    def predict(self, model):
        bbox = Variable(torch.from_numpy(self.bbox)).cuda()

        # obtain h, alpha, sigma using RAN
        alpha_bbox, sigma_bbox, self.h_bbox = model(bbox, self.h_bbox)

        # obtain mu using alpha and external memory
        alpha_bbox = alpha_bbox.data.cpu().to_numpy()
        self.mu_bbox = alpha_bbox * np.array(self.external_bbox)

        self.sigma_bbox = sigma_bbox.data.cpu().to_numpy()

    def similarity(self, bbox, feature=None):
        """
        Computes similarity between the new detection and this track
        """

        # linear combination of instances in the external memory using alpha
        pass


class RANTracker(object):
    def __init__(self, max_age=30, mem_size=10):
        self.max_age = max_age



        self.tracks = []
        self._next_id = 1

    def linear_combination(self):
        return np.matmul(self.alpha_bbox, np.array(self.externel_bbox))

    def predict(self, model):
        """Update alpha, sigma, and hidden states for all the tracks
        using the RAN model
        """
        for track in self.tracks:
            track.predict(model)

    def update(self, bboxes, features=None):


        # run matching
        # update tracks
        # mark missed tracks
        # initiate new tracks
        pass

    def _init_track(self, bbox, feature=None):
        self.tracks.append(RANTrack(bbox, self._next_id, feature))
        self._next_id += 1



def associate_detections_to_trackers(detections, trackers, min_similarity=0):

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4))

    sim_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in detections:
        for t, trk in trackers:
            sim_matrix[d, t] = trk.similarity(det)

    matched_indices = linear_assignment(-sim_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    for d, _ in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    for t, _ in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    for d, t in matched_indices:
        if sim_matrix[d, t] < min_similarity:
            unmatched_tracks.append(t)
            unmatched_detections.append(d)
        else:
            matches.append((d, t))
    return matches, unmatched_tracks, unmatched_detections



if __name__ == '__main__':
    mem_size = 10
    externel = deque([np.zeros(4, dtype=np.float32) for _ in range(mem_size)], maxlen=mem_size)
    alpha = np.zeros(mem_size)
    print(np.matmul(alpha, np.array(externel)))
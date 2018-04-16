import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class RANTracker(object):
    def __init__(self, bbox, feature=None):
        self.bbox = bbox
        self.feature = feature
        self.h_bbox = 0
        self.h_feature = 0

        # use np array for this
        self.external_bbox = 0
        self.externel_feature = 0

    def update(self, bbox, feature=None):
        self.bbox = bbox
        self.feature = feature

    def predict(self, model):
        bbox = Variable(torch.from_numpy(self.bbox)).cuda()
        h_bbox = Variable(torch.from_numpy(self.h_bbox)).cuda()
        #update h, alpha, sigma using RAN
        alpha_bbox, sigma_bbox, self.h_bbox = model(bbox, h_bbox)

        self.alpha_bbox = alpha_bbox.data.cpu().to_numpy()
        self.sigma_bbox = sigma_bbox.data.cpu().to_numpy()

    def similarity(self, bbox, feature=None):
        """
        Computes similarity between the new detection and this target
        """
import math
import torch as t
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter


class RAN(nn.Module):
    def __init__(self, input_size, hidden_size, history_size, drop_rate):
        super(RAN, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bias=False,
                          dropout=drop_rate)
        self.linear_alpha = nn.Linear(in_features=hidden_size,
                                out_features=history_size,
                                bias=False)
        self.linear_sigma = nn.Linear(in_features=hidden_size,
                                out_features=input_size,
                                bias=False)
        self.softmax = nn.Softmax()  # for batch training, need dim info.
        self.drop = nn.Dropout(drop_rate)

    def encode(self, x):
        return self.linear(self.gru(x))

    def init_hidden(self):
        pass

    def init_weight(self):
        pass



    def forward(self, x, hidden, external):
        # RNNdropout is not implemented
        _, hidden = self.gru(x, hidden)
        alpha = self.softmax(self.linear_alpha(hidden))
        sigma = t.exp(self.linear_sigma(hidden))
        mu = alpha @ external
        prob = log_prob(x, mu, t.diag(sigma))

        return prob

def log_prob(value, mu, cov):
    diff = value - mu
    M = _batch_mahalanobis(cov, diff)
    log_det = _batch_diag(cov).abs().log().sum(-1)
    return -0.5 * (M + mu.size(-1) * math.log(2 * math.pi)) - log_det


def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L.unsqueeze(0).reshape((-1,) + L.shape[-2:])
    L_inv = t.stack([t.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)

def _batch_diag(bmat):
    r"""
    Returns the diagonals of a batch of square matrices.
    """
    return bmat.reshape(bmat.shape[:-2] + (-1,))[..., ::bmat.size(-1) + 1]


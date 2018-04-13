import math
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.nn.parameter import Parameter

use_cuda = t.cuda.is_available()
class RAN(nn.Module):
    def __init__(self, input_size, hidden_size, history_size, drop_rate):
        super(RAN, self).__init__()
        self.hidden_size = hidden_size
        self.inpu_size = input_size
        self.history_size = history_size
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
        self.softmax = nn.Softmax(dim=-1)  # for batch training, need dim info.
        # self.drop = nn.Dropout(drop_rate)
        self.init_weight()

    def encode(self, x):
        return self.linear(self.gru(x))

    def init_hidden(self, batch_size):
        result = Variable(t.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def init_weight(self):
        xavier_uniform(self.linear_alpha.weight.data)
        xavier_uniform(self.linear_sigma.weight.data)



    def forward(self, x, hidden, external):
        # RNNdropout is not implemented
        # x (seq_len, batch, input_size)
        # hidden (1, batch, hidden_size)
        # output (seq_len, batch, hidden_size)
        # h_n (1, batch, hidden_size)
        # external (seq_len, batch, history, feature)
        # return (


        # x (20, 64, 4)
        # hidden (1, 64, 32)
        # external [20, 64, 10, 4]
        # output (20, 64, 32)
        output, _ = self.gru(x, hidden)
        a = self.linear_alpha(hidden)  # (1, 64, 10)
        alpha = self.softmax(a)  # (1, 64, 10)
        sigma = t.exp(self.linear_sigma(hidden))  # (1, 64, 4)
        if self.training:
            mu = t.matmul(external,
                          alpha.permute([1, 2, 0])).squeeze() # (20, 64, 4, 1)
        else:
            raise NotImplementedError
        diff = t.pow(x - mu, 2)
        M = t.matmul(diff.unsqueeze(2),
                     1 / sigma.unsqueeze(-1)).squeeze()  # (20, 64)
        M = 0.5 * t.sum(M)
        log_det = t.prod(sigma, dim=-1).abs().log().sum()
        c = math.log(2 * math.pi)
        # mu.size(-1) * math.log(2 * math.pi))
        # prob = log_prob(x, mu, t.diag(sigma))
        # need to normalize
        prob = M
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


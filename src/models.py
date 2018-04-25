import math
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import os
import time
from torch.nn.parameter import Parameter

use_cuda = t.cuda.is_available()
class RAN(nn.Module):
    def __init__(self, input_size, hidden_size, history_size, drop_rate,
                 save_path='ran'):
        super(RAN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
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
        self.save_path = save_path

    def init_hidden(self, batch_size):
        result = Variable(t.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def init_weight(self):
        xavier_uniform(self.linear_alpha.weight.data)
        xavier_uniform(self.linear_sigma.weight.data)

    def forward(self, x, hidden):
        """
        Compute autoregressive parameter and variance.
        T is the longest sequence in the batch
        B is batch size / number of trajectories
        F is feature size
        H is history size
        Args:
            x (PackedSequence or Variable): (T, B, F) (20, 64, 4)
            hidden (Variable): (1, B, hidden_size) (1, 64, 32)

        Returns:
            alpha (Variable): (T, B, H)
            sigma (Variable): (T, B, F)
            h_n (Variable): (1, B, hidden_size)
        """
        # RNNdropout is not implemented

        output, h_n = self.gru(x, hidden)  # output (20, 64, 32)
        # unpack output and pad with zeros
        if isinstance(output, PackedSequence):
            output, _ = pad_packed_sequence(output)

        alpha = self.softmax(self.linear_alpha(output))  # (T, 64, 10)
        sigma = t.exp(self.linear_sigma(output))  # (T, 64, 4)
        # trim off padding
        # alpha = [alpha[0:l, i, :] for i, l in enumerate(lengths)]
        # sigma = [sigma[0:l, i, :] for i, l in enumerate(lengths)]
        return alpha, sigma, h_n


def save_model(model, p=None):
    """This functions cannot be inside the model"""
    if p is None:
        p = model.save_path + '.pt'

    t.save(model.state_dict(), p)


def load_model(m, p=None):
    """
    You have to build the exact model first
    """
    if p is not None:
        print("loading from {}".format(p))
        m.load_state_dict(t.load(p))
    else:
        print('loading model from default path')
        m.load_state_dict(t.load(m.save_path + '.pt'))





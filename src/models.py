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

    def init_weight(self):

    def forward(self, x, hidden):
        _, hidden = self.gru(x, hidden)
        alpha = self.softmax(self.linear_alpha(hidden))
        sigma = t.exp(self.linear_sigma(hidden))
        return alpha, sigma, hidden


import numpy as np
import torch
from utils import generate_external
from torch.autograd import Variable
import time
from torch import optim
from models import RAN
import math
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
import progressbar


class Trainer(object):
    def __init__(self, opt, motion_dim, feat_dim, dataloader):

        self.opt = opt
        self.RAN_motion = RAN(input_size=motion_dim, hidden_size=32, history_size=10, drop_rate=0.5)
        self.RAN_feat = RAN(input_size=feat_dim, hidden_size=32, history_size=10, drop_rate=0.5)

        self.dataloader = dataloader

        if self.opt.use_cuda:
            self.RAN_motion.cuda()
            self.RAN_feat.cuda()

        self.optimizer_motion = optim.Adam(self.RAN_motion.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.optimizer_feat = optim.Adam(self.RAN_feat.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def train(self):
        self.RAN_motion.train()
        self.RAN_feat.train()

        total_loss = []
        curr_iters = 0

        for epoch in range(self.opt.nepoch):

            for i, (motion_data, feat_data) in enumerate(self.dataloader):
                curr_iters += 1

                ########
                # Train motion model
                ########
                self.RAN_motion.zero_grad()
                padded_batch, lengths, packed_input, ext = self.prepare_data(motion_data)

                hidden = self.RAN_motion.init_hidden(len(lengths))  # (1, B, hidden)

                alpha, sigma, h_n = self.RAN_motion(packed_input, hidden)

                loss = loss_fn(alpha, sigma, padded_batch[1:], ext, lengths)
                loss.backward()
                self.optimizer_motion.step()

                if i == 1:
                    print('Epoch: {}, M Loss: {}'.format(epoch, loss.cpu().data.numpy()))

                ########
                # Train appearance model
                ########
                self.RAN_feat.zero_grad()
                padded_batch, lengths, packed_input, ext = self.prepare_data(feat_data)

                hidden = self.RAN_feat.init_hidden(len(lengths))  # (1, B, hidden)

                alpha, sigma, h_n = self.RAN_feat(packed_input, hidden)

                loss = loss_fn(alpha, sigma, padded_batch[1:], ext, lengths)
                loss.backward()
                self.optimizer_feat.step()

                if i == 1:
                    print('Epoch: {}, A Loss: {}'.format(epoch, loss.cpu().data.numpy()))

        torch.save({'RAN_motion': self.RAN_motion.state_dict(),
                    'RAN_feat': self.RAN_feat.state_dict()},
                   '{}/models/RAN.pth'.format(self.opt.outf))

    def prepare_data(self, batch_data):
        # obtain a tensor (max_length, batch_size, feat_dim) and lengths for sequences
        padded_batch, lengths = pad_packed_sequence(batch_data)
        lengths = [l - 1 for l in lengths]

        ext = generate_external(padded_batch.data.numpy()[:-1],
                                lengths, self.opt.history_size)
        ext = Variable(torch.from_numpy(ext), requires_grad=False)

        # generate input from t=0 to t=L-2
        packed_input = pack_padded_sequence(padded_batch[:-1], lengths)

        if self.opt.use_cuda:
            ext = ext.cuda()
            padded_batch = padded_batch.cuda()
            packed_input = PackedSequence(packed_input.data.cuda(),
                                          packed_input.batch_sizes)

        return padded_batch, lengths, packed_input, ext


def loss_fn(alpha, sigma, x, ext, lengths):
    """
    compute negative log likelihood probability of new observation x
    Args:
        alpha (Variable): (T, B, H) padded with 1/H
        sigma (Variable): (T, B, F) padded with 1
        x (Variable): (T, B, F) padded with 0
        external (Variable): (seq_len, batch, feature, history) (20, 64, 4, 10)
        lengths (list): a list of seq_len, one for each traj. Can be a list of
            ones whith size of the batch during testing.
    Returns:
        negative log likelihood of oberservation. scalar tensor.
    """
    # H = alpha.data.shape[-1]  # history
    # S = lengths[-1]  # shortest
    # assert (alpha.data[S, -1, :] == 1/ H).prod() == 1
    # assert (sigma.data[S, -1, :] == 1).prod() == 1
    # assert x.data[S, -1, :].sum() == 0
    # assert ext.data[S, -1, :, :].sum() == 0

    mu = torch.matmul(ext, alpha.unsqueeze(-1)).squeeze()
    diff = torch.pow(x - mu, 2)
    M = torch.matmul(diff.unsqueeze(2),
                 1 / sigma.unsqueeze(-1)).sum()  # Mahalanobis distance
    log_det = sigma.log().sum()
    # sum over all sequence of all batch of all features
    n = np.sum(np.array(lengths)) * x.shape[-1]
    c = n * math.log(2 * math.pi)
    # mu.size(-1) * math.log(2 * math.pi))
    return 0.5 * (log_det + M + c)


use_cuda = True

def trainIters(model, dataloader, n_epoch, lr=0.001, betas=(0.9, 0.99), eps=1e-8):
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    total_loss = []
    i = 0
    # progressbar setting
    widgets = ['Training epoch ', progressbar.Counter(), progressbar.Percentage(),
               ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=n_epoch)

    for e in bar(range(n_epoch)):

        for sample, _ in dataloader:
            i += 1
            model.train()
            optimizer.zero_grad()

            # unpack sample
            packed_batch = sample  # PackedSequence

            # obtain a tensor (T, B, *) and lengths for sequences
            padded_batch, lengths = pad_packed_sequence(packed_batch)
            lengths = [l-1 for l in lengths]

            # generate two copies, one from 0 to L-2, the other from 1 to L-1
            packed_input = pack_padded_sequence(padded_batch[:-1], lengths)
            #packed_output = pack_padded_sequence(padded_batch[1:], lengths)

            ext = generate_external(padded_batch.data.numpy()[:-1],
                                    lengths, model.history_size)

            # pytorch 0.31 does not support cuda() call for PackedSequence.
            # support will be added for pytorch 0.4
            if use_cuda:
                packed_input = PackedSequence(packed_input.data.cuda(),
                                              packed_input.batch_sizes)

            hidden = model.init_hidden(len(lengths))  # (1, B, hidden)

            alpha, sigma, h_n = model(packed_input, hidden)

            ext = Variable(torch.from_numpy(ext), requires_grad=False)

            if use_cuda:
                ext, padded_batch = ext.cuda(), padded_batch.cuda()

            loss = loss_fn(alpha, sigma, padded_batch[1:], ext, lengths)
            loss.backward()
            optimizer.step()

            total_loss += list(loss.data)
    print(i, ' iterations')
    print('min loss ', np.array(total_loss).min())
    plt.plot(total_loss)
    plt.ylabel('Negative Log-likelihood')
    plt.xlabel('Batches')
    plt.show()



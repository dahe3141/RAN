import numpy as np
import torch as t
from utils import generate_external
from torch.autograd import Variable
import time
from torch import optim
import math
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import matplotlib.pyplot as plt
import progressbar

use_cuda = t.cuda.is_available()


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

    mu = t.matmul(ext, alpha.unsqueeze(-1)).squeeze()
    diff = t.pow(x - mu, 2)
    M = t.matmul(diff.unsqueeze(2),
                 1 / sigma.unsqueeze(-1)).sum()  # Mahalanobis distance
    log_det = sigma.log().sum()
    # sum over all sequence of all batch of all features
    n = np.sum(np.array(lengths)) * x.shape[-1]
    c = n * math.log(2 * math.pi)
    # mu.size(-1) * math.log(2 * math.pi))
    return 0.5 * (log_det + M + c)


def trainIters(model, dataloader, n_epoch, lr=0.001, betas=(0.9, 0.99), eps=1e-8):
    if use_cuda: model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    total_loss = []
    i=0
    # progressbar setting
    widgets = ['Training epoch ', progressbar.Counter(), progressbar.Percentage(),
               ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=n_epoch)
    for e in bar(range(n_epoch)):

        for sample in dataloader:
            i += 1
            model.train()
            optimizer.zero_grad()

            # unpack sample
            packed_batch = sample  # PackedSequence

            # pad batch to get lengths and generate external memory
            padded_batch, lengths = pad_packed_sequence(packed_batch)
            ext = generate_external(padded_batch.data.numpy(),
                                    lengths, model.history_size)

            # pytorch 0.31 does not support cuda() call for PackedSequence.
            # support will be added for pytorch 0.4
            if use_cuda: packed_batch = PackedSequence(packed_batch.data.cuda(),
                                                       packed_batch.batch_sizes)

            hidden = model.init_hidden(len(lengths))  # (1, B, hidden)

            alpha, sigma, h_n = model(packed_batch, hidden)

            ext = Variable(t.from_numpy(ext), requires_grad=False)
            if use_cuda: ext, padded_batch = ext.cuda(), padded_batch.cuda()

            loss = loss_fn(alpha, sigma, padded_batch, ext, lengths)
            loss.backward()
            optimizer.step()

            total_loss += list(loss.data)
    print(i, ' iterations')
    print('min loss ', np.array(total_loss).min())
    plt.plot(total_loss)
    plt.ylabel('Negative Log-likelihood')
    plt.xlabel('Batches')
    plt.show()



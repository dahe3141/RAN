import numpy as np
import torch as t
from utils import get_batch, generate_external
from torch.autograd import Variable
import time
from torch import optim
import math

use_cuda = t.cuda.is_available()

def loss_fn(alpha, sigma,x, external, n):
    """
    compute negative log likelihood probability of new observation x
    Args:
        alpha: (seq_len, batch, history) (20, 64, 10)
        sigma: (seq_len, batch, feature) (20, 64, 4)
        x: (seq_len, batch, feature) (20, 64, 4)
        external: (seq_len, batch, feature, history) (20, 64, 4, 10)
        n: (n_frame * batch_size * feature_size) used to compute the constant c

    Returns:
        negative log likelihood of oberservation. scalar tensor.
    """
    mu = t.matmul(external,  # (20, 64, 4, 10) * (20, 64, 10, 1) = (20, 64, 4, 1)
                  alpha.unsqueeze(-1)).squeeze()
    diff = t.pow(x - mu, 2)
    M = t.matmul(diff.unsqueeze(2),
                 1 / sigma.unsqueeze(-1)).sum()  # Mahalanobis distance
    log_det = sigma.log().sum()
    c = n * math.log(2 * math.pi)
    # mu.size(-1) * math.log(2 * math.pi))
    return 0.5 * (log_det + M + c)

def train(model, optimizer, traj, n_frame, batch_size):
    """In progress"""
    model.train()
    optimizer.zero_grad()
    hidden = model.init_hidden(batch_size)  # (1, 1, 32)
    external = generate_external(traj, model.history_size)
    external = Variable(t.from_numpy(external.astype(np.float32)))

    if use_cuda: external = external.cuda()
    x = Variable(t.from_numpy(traj.astype(np.float32)), requires_grad=False)
    if use_cuda: x = x.cuda()

    alpha, sigma, h_n = model(x, hidden)
    loss = loss_fn(alpha, sigma, x, external,
                    n_frame * batch_size * model.input_size)
    loss.backward()
    optimizer.step()

    return loss, h_n


def trainIters(model, samples, n_iters, n_frame=20, batch_size=64, lr=0.001,
               betas=(0.9, 0.99), eps=1e-8):
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    total_loss = []
    for iter in range(1, n_iters + 1):

        traj, idx = get_batch(samples, n_traj=batch_size, n_frame=n_frame)
        loss, hidden = train(model, optimizer, traj, n_frame, batch_size)
        total_loss += list(loss.data)

    print(total_loss)




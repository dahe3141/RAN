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
        alpha: (1, batch, history) (1, 64, 10)
        sigma: (1, batch, feature) (1, 64, 4)
        x: (seq_len, batch, input_size) (20, 64, 4)
        external: (seq_len, batch, feature, history) (20, 64, 4, 10)
        n: (n_frame * batch_size * feature_size) used to compute the constant c

    Returns:
        negative log likelihood of oberservation. scalar tensor.
    """
    mu = t.matmul(external,  # (20, 64, 4, 10) * (20, 64, 10, 1) = (20, 64, 4, 1)
                  alpha.permute([1, 2, 0])).squeeze()
    diff = t.pow(x - mu, 2)
    M = t.matmul(diff.unsqueeze(2),
                 1 / sigma.unsqueeze(-1)).squeeze()  # (20, 64)
    M = t.sum(M)  # Mahalanobis distance
    log_det = t.prod(sigma, dim=-1).abs().log().sum()
    c = n * math.log(2 * math.pi)
    # mu.size(-1) * math.log(2 * math.pi))
    # prob = log_prob(x, mu, t.diag(sigma))
    # need to normalize
    return 0.5 * (log_det + M + c)

def train(model, optimizer, traj, n_frame, batch_size):
    """In progress"""
    model.train()
    optimizer.zero_grad()
    hidden = model.init_hidden(batch_size)  # (1, 1, 32)
    # external = t.zeros(model.history_size, 4)
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
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    # t.nn.utils.clip_grad_norm(model.parameters(), args.clip)

    return loss, h_n

        # if batch % args.log_interval == 0 and batch > 0:
        #     cur_loss = total_loss[0] / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
        #             'loss {:5.2f} | ppl {:8.2f}'.format(
        #         epoch, batch, len(train_data) // args.bptt, lr,
        #         elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()


def trainIters(model, samples, n_iters, n_frame=20, batch_size=64, lr=0.001,
               betas=(0.9, 0.99), eps=1e-8):
    if use_cuda:
        model = model.cuda()
    # start = time.time()
    # plot_losses = []
    # print_loss_total = 0  # Reset every print_every
    # plot_loss_total = 0  # Reset every plot_every
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    # training_pairs = [variablesFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    # criterion = nn.NLLLoss()
    total_loss = []
    for iter in range(1, n_iters + 1):

        traj, idx = get_batch(samples, n_traj=batch_size, n_frame=n_frame)
        # loss = train(input_variable, target_variable, encoder,
        #              decoder, encoder_optimizer, decoder_optimizer, criterion)

        loss, hidden = train(model, optimizer, traj, n_frame, batch_size)
        total_loss += list(loss.data)
        # print_loss_total += loss
        # plot_loss_total += loss
        #
        # if iter % print_every == 0:
        #     print_loss_avg = print_loss_total / print_every
        #     print_loss_total = 0
        #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
        #                                  iter, iter / n_iters * 100, print_loss_avg))
        #
        # if iter % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total = 0
    print(total_loss)
    # showPlot(plot_losses)




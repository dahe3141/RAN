import numpy as np
import torch as t
from utils import get_batch, generate_external
from torch.autograd import Variable
import time
from torch import optim

use_cuda = t.cuda.is_available()

def loss_fn(x):
    return -t.cumsum(x, dim=0)

def train(model, optimizer, traj, n_frame, batch_size):
    """In progress"""
    model.train()
    total_loss = 0
    # init hidden currently in
    hidden = model.init_hidden(batch_size) # (1, 1, 32)
    for i in range(1):
        optimizer.zero_grad()
        loss = 0
        # external = t.zeros(model.history_size, 4)
        external = generate_external(traj, model.history_size)
        external = Variable(t.from_numpy(external.astype(np.float32)))
        if use_cuda: external = external.cuda()
        x = Variable(t.from_numpy(traj.astype(np.float32)), requires_grad=False)
        if use_cuda: x = x.cuda()
        output, hidden = model(x, hidden, external)
        loss += output
        loss = criterion()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        t.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        total_loss += loss.data

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

    for iter in range(1, n_iters + 1):

        traj, idx = get_batch(samples, n_traj=batch_size, n_frame=n_frame)
        # loss = train(input_variable, target_variable, encoder,
        #              decoder, encoder_optimizer, decoder_optimizer, criterion)

        train(model, optimizer, traj, n_frame, batch_size)
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

    # showPlot(plot_losses)




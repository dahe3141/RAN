import numpy as np
import torch as t
from utils import get_batch
import time

def loss_fn(x):
    return -t.cumsum(x, dim=0)

def train(model, criterion, sample):
    """In progress"""
    model.train()
    total_loss = 0
    hidden = model.init_hidden()
    for i in range(1):
        traj, idx = get_batch(sample)
        model.zero_grad()
        loss = 0
        for j in range(traj.shape[0]): # for each time step
            output, hidden = model(traj[j, :, :], hidden, external)
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







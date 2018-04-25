import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
from MOT16_Dataset_vgg import MOT16_Dataset_vgg
from torchvision.models import vgg16_bn
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import shutil

# adopted from:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb.js

# dataset params:
data_root = os.path.expanduser(os.path.abspath(os.path.join(os.path.pardir,
                                                            "Data", "MOT16")))
processed_folder = 'processed'

th_vis = 0.9
consider_only = False
valid_size = 0.1
random_seed = 1

# opt params:
lr = 0.01
weight_decay = 1e-4
momentum = 0.9

# training params
batch_size = 32  # can only run this small batch
epochs = 300
workers = 4
pin_memory = True
half = False  # convert to half tensor. not working correctly now
augment = False  # data augmentation

# other
resume_path = None
# os.path.join(data_root, processed_folder, 'checkpoint.pth.tar')
# 'model_best.pth.tar'
# 'checkpoint.pth.tar'
start_epoch = 0
print_freq = 200

use_cuda = torch.cuda.is_available()
best_prec1 = 0


# TODO: Currently using pre-trained VGG. We may not need such a large network.
# TODO: average aspect ratio is 1 by 3. Should use 32 by 96 instead of 224 by 224
# TODO: Better way to organize hyperparameters?
def main():
    global start_epoch, best_prec1

    # create model:
    model = vgg16_bn(pretrained=True)
    # model.features = torch.nn.DataParallel(model.features)
    if use_cuda:
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    if half:
        model.half()
        criterion.half()

    # optionally resume from a checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))


    # benchmark mode is good whenever your input sizes for your network do not vary.
    # This way, cudnn will look for the optimal set of algorithms for that
    # particular configuration (which takes some time). This usually leads to
    # faster runtime.
    # But if your input sizes changes at each iteration, then cudnn will benchmark
    # every time a new size appears, possibly leading to worse runtime performances.
    cudnn.benchmark = True

    # Data loading code
    # computed from dataset
    normalize = transforms.Normalize(mean=[0.71954213, 0.69802504, 0.67027679],
                                     std=[0.44922294, 0.45911445, 0.47011256])
    # given from VGG
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # data augmentation
    if augment:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = MOT16_Dataset_vgg(data_root, train=True,
                                      th_vis=th_vis,
                                      consider_only=consider_only,
                                      transform=transform)

    # create subset sampler to separate training and validation data.
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              shuffle=False,
                              num_workers=workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=valid_sampler,
                            shuffle=False,
                            num_workers=workers,
                            pin_memory=pin_memory)

    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        if half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda(async=True)
            input = input.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        if half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    filename = os.path.join(os.path.expanduser(data_root), processed_folder,
                            filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(data_root, processed_folder,
                                               'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

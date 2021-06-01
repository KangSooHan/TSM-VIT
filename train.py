#-*- coding: utf-8 -*-#
import os
import time
import argparse
import logging
import shutil
import numpy as np
from tqdm import tqdm

import torch

from hparams import Hparams
from utils import *
from ops.data_load import return_dataset
from ops.dataset import DataSet
from ops.transforms import *
from ops.temporal_shift import make_temporal_shift
#from model import *

from models.model import VisionTransformer, CONFIGS

from tensorboardX import SummaryWriter


def main():
    global hp
    logging.basicConfig(level = logging.INFO)

    logging.info("# Hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    logging.info("# Prepare Dataset")
    num_class, hp.train_list, hp.val_list, hp.root_path, prefix = return_dataset(hp.dataset, hp.modality)

    save_dir = hp.save_dir

    config = CONFIGS[hp.model_type]

    model = VisionTransformer(config, hp.img_size, zero_head=True, num_classes=101, modality=hp.modality, num_segments=hp.num_segments)

    emb = model.transformer.embeddings
    scale_size = model.scale_size
    crop_size = model.crop_size
    train_augmentation = model.get_augmentation()
    model.load_from(np.load(os.path.join(hp.pretrained_dir, hp.model_type+'.npz')))

    from ops.temporal_shift import make_temporal_shift
    make_temporal_shift(model, hp.num_segments)
    print(model)

    policies = model.get_optim_policies()

    num_params = count_parameters(model)
    logging.info("{}".format(config))
    logging.info("Training Parameters %s", hp)
    logging.info("Total Parameters: \t%2.1fM" % num_params)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(policies,
                        hp.lr,
                        #momentum=hp.momentum,
                        weight_decay=hp.weight_decay)


    # Data loading code
    if hp.modality != 'RGBDiff':
        normalize = GroupNormalize(
            [.485, .456, .406],
            [.229, .224, .225])

    else:
        normalize = IdentityTransform()

    if hp.modality == 'RGB':
        data_length = 1
    elif hp.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        DataSet(hp.root_path, hp.train_list, num_segments=hp.num_frames,
                   new_length=data_length,
                   modality=hp.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ]), dense_sample=hp.dense_sample),
        batch_size=hp.batch_size, shuffle=True,
        num_workers=hp.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        DataSet(hp.root_path, hp.val_list, num_segments=hp.num_frames,
                   new_length=data_length,
                   modality=hp.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ]), dense_sample=hp.dense_sample),
        batch_size=hp.batch_size, shuffle=False,
        num_workers=hp.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if hp.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    hp.store_name = "test"

    tf_writer = SummaryWriter(log_dir=os.path.join(hp.root_log, hp.store_name))
    best_prec1=0
    for epoch in range(0, hp.epochs):
        adjust_learning_rate(optimizer, epoch, hp.lr_type, hp.lr_steps)

        # train for one epoch
        train(train_loader, model, emb, criterion, optimizer, epoch, tf_writer)

        # evaluate on validation set
        if (epoch + 1) % hp.eval_freq == 0 or epoch == hp.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)

def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = hp.lr * decay
        decay = hp.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * hp.lr * (1 + math.cos(math.pi * epoch / hp.epochs))
        decay = hp.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def train(train_loader, model, emb, criterion, optimizer, epoch, tf_writer):
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

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

#        img_emb = emb(input_var.view((-1, 3) + input_var.size()[-2:]).cuda())
#        nt, c, h = img_emb.size()
#        img_emb = img_emb.view(-1, hp.num_frames, c*h)
#        sim = torch.zeros(img_emb.size()[:2])
#        for t in range(hp.num_frames-1):
#            sim[:,t] = torch.nn.CosineSimilarity()(img_emb[:,t], img_emb[:, t+1])
#
#        idx,_ = torch.sort(torch.topk(sim, hp.num_segments, 1).indices)
#        input_var = input_var.view((-1, hp.num_frames, 3) + input_var.size()[-2:])
#        inputs = batched_index_select(input_var, 1, idx).view((-1, hp.num_segments*3)+input_var.size()[-2:])
#
        # compute output
        output, _ = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if hp.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), hp.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % hp.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output,_ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % hp.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg



if __name__ == '__main__':
    main()

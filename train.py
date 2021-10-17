# -*- coding: utf-8 -*-
"""
@author    : GiantPandaSR
@data      : 2021-02-09
@describe  : Training with DDP or DataParallel
"""
from __future__ import print_function

from optim.optimizer import Optimizer, adjust_learning_rate_for_cosine_decay
from config.Config import Config
from loss.generator_loss import L1_Charbonnier, MS_SSIM, SSIM
from data.utils import PSNR, Metric_rank, BatchAug
from data.vanilar_dataset import NTIRE_Track2
from data.augments import *
from datetime import datetime

from model.NTIRE2020_Deblur_top.uniA.model_stage1 import AtrousNet
from model.NTIRE2021_Deblur.CARN.CARN import Net as CARNNet
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1 import AtrousNet as AtrousNetElu
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_Upsample_Deep import AtrousNet_billinear_Wide as AtrousNetEluUpWide
# wavelet with crop
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_deblur_remix import AtrousNet_wavlet_remix
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_SRCNN_remix import SRCNN
from utils.calc_psnr_for_val import get_psnr
# from model.NTIRE2020_Deblur_top.uniA.model_stage1 import AtrousNet>>>>>>> master
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as AMPDDP
from apex.parallel.LARC import LARC
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

from apex import amp
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import random
import math
import time
import cv2
import os

# system
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='SR training')
parser.add_argument('--config_file', type=str,
                    default="/data/jiangmingchao/data/SR_NTIRE2021/config/VDSR.yaml")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--world-size', type=int, default=1,
                    help="number of nodes for distributed training")
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', default=1, type=int,
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
parser.add_argument('--local_rank', default=1)
parser.add_argument('--dataparallel', default=1, type=int,
                    help="model data parallel")


# random seed
def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_ckpt(model, weights):
    state_dict = torch.load(weights, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    return model


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

# main func
def main_worker(gpu, ngpus_per_node, args):
    cfg = Config(args.config_file)()
    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu
    model_arch = "{}-{}".format("SR", "AtrousNet")
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print('rank: {} / {}'.format(args.rank, args.world_size))
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
    if args.rank == 0:
        if not os.path.exists(cfg.CHEKCPOINTS.CKPT_PATH):
            os.makedirs(cfg.CHEKCPOINTS.CKPT_PATH)

    # metric
    train_loss_metric = Metric_rank("train_loss")
    train_psnr_metric = Metric_rank("train_psnr")
    train_metric = {"loss": train_loss_metric,
                    "psnr": train_psnr_metric}

    val_loss_metric = Metric_rank("val_loss")
    val_psnr_metric = Metric_rank("val_psnr")
    val_metric = {"loss": val_loss_metric,
                  "psnr": val_psnr_metric}

    # psnr
    calculate_psnr = PSNR(cfg)

    # model
    if cfg.MODEL.BACKBONE.NAME == "CARN":
        model = CARNNet(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet":
        model = AtrousNet(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetElu":
        model = AtrousNetElu(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetEluUpWide":
        model = AtrousNetEluUpWide(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetEluUpWideWaveletRemix":
        model = AtrousNet_wavlet_remix(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "waveletSrcnn":
        model = SRCNN(3)


    if cfg.MODEL.PRETRAIN:
        model = load_ckpt(model, cfg.MODEL.WEIGHTS)

    # model = VDSR(in_channels=3, out_channels=3)
    if args.rank == 0:
        print("================{}=============".format(model_arch))
        print(model)
    if torch.cuda.is_available():
        model.cuda(args.gpu)

    # batch augments
    batch_aug = BatchAug(cfg)

    # optim
    Optim = Optimizer(cfg)
    optimizer = Optim.optimizer(model)

    if cfg.TRAIN.LARS:
        optimizer = LARC(optimizer)

    if cfg.TRAIN.MIX:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if cfg.TRAIN.FP16:
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
        model = model.half()

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        if args.dataparallel:
            model = DataParallel(model)
        else:
            model = model

    device = model.device

    if cfg.LOSS.MS_SSIM:
        ssim_criterion = MS_SSIM()
    elif cfg.LOSS.SSIM:
        ssim_criterion = SSIM()

    # loss
    if cfg.LOSS.L1_Charbonnier:
        criterion = L1_Charbonnier(eps=cfg.LOSS.EPS)
    else:
        criterion = torch.nn.L1Loss()

    # dataset
    train_dataset = NTIRE_Track2(cfg, train=True)
    validation_dataset = NTIRE_Track2(cfg, train=False)
    if args.rank == 0:
        print("Training dataset length: ", len(train_dataset))
        print("Validation dataset length: ", len(validation_dataset))
    # sampler
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        validation_sampler = DistributedSampler(validation_dataset)
    else:
        train_sampler = None
        validation_sampler = None

    # logs
    log_writer = SummaryWriter(cfg.CHEKCPOINTS.LOGS_PATH)

    if cfg.IPNUT.NORM:
        denormalize = DeNormalize(cfg.INPUT.MEAN, cfg.IPNUT.STD)
    else:
        denormalize = None

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=cfg.VALIDATION.BATCH_SIZE,
        shuffle=(validation_sampler is None),
        num_workers=cfg.VALIDATION.NUM_WORKERS,
        pin_memory=True,
        sampler=validation_sampler,
        drop_last=True
    )

    batch_iter = 0
    train_batch = math.ceil(len(train_dataset) / (cfg.TRAIN.BATCH_SIZE * ngpus_per_node))
    total_batch = train_batch * cfg.TRAIN.MAX_EPOCHS
    no_warmup_total_batch = int(cfg.TRAIN.MAX_EPOCHS - cfg.TRAIN.WARM_EPOCHS) * train_batch

    best_loss, best_psnr = 100.0, 0.0
    batch_count = 0
    batch_total_count = int(len(train_loader) * cfg.TRAIN.MAX_EPOCHS // cfg.VALIDATION.ITER)

    # training loop
    for epoch in range(1, cfg.TRAIN.MAX_EPOCHS + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for epoch
        batch_iter = train(cfg, train_loader, validation_loader, model, criterion, optimizer, batch_aug, epoch, args,
                            batch_iter, total_batch, train_batch, no_warmup_total_batch,
                            log_writer, calculate_psnr, train_metric, denormalize, ssim_criterion,
                            batch_count, batch_total_count)

        # control the validaiton calculate step
        if cfg.VALIDATION.USE and (epoch % cfg.VALIDATION.INTERVAL) == 0 and epoch >= cfg.VALIDATION.STEP :
            val_loss, val_psnr = val(cfg, validation_loader, model, criterion, epoch, cfg.TRAIN.MAX_EPOCHS, args, log_writer, calculate_psnr, denormalize=denormalize)

            if args.rank == 0:
                # save best loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state_dict, cfg.CHEKCPOINTS.CKPT_PATH + '/' + model_arch + f'_best_loss_{best_loss}' + '.pth')

                # save best psnr
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state_dict, cfg.CHEKCPOINTS.CKPT_PATH + '/' + cfg.MODEL.BACKBONE.NAME + f'_best_psnr_{best_psnr}' + '.pth')

        if (epoch + 1) % 10 == 0:
            if args.rank == 0:
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, cfg.CHEKCPOINTS.CKPT_PATH + '/' + cfg.MODEL.BACKBONE.NAME + '_epoch_{}'.format(epoch+1) + '.pth')


# epoch scalra record
def record_scalars(log_writer, mean_loss, mean_psnr, epoch, flag="train"):
    log_writer.add_scalar(f"{flag}/epoch_loss", mean_loss, epoch)
    log_writer.add_scalar(f"{flag}/epoch_psnr", mean_psnr, epoch)


# batch scalar record
def record_log(log_writer, losses, lr, batch_idx, batch_time):
    log_writer.add_scalar("train/l1_loss", losses.data.item(), batch_idx)
    log_writer.add_scalar("train/learning_rate", lr, batch_idx)
    log_writer.add_scalar("train/batch_time", batch_time, batch_idx)


# sr images record
def draw_images(cfg, epoch, log_writer, lr_images, sr_images, hr_images, denormalize, flag="train"):
    """record the sr images
    """
    max_numbers = cfg.DEBUG.MAX_NUMBERS
    if denormalize is not None:
        lr_images= denormalize(lr_images)
        sr_images= denormalize(sr_images)
        hr_images= denormalize(hr_images)

    assert isinstance(sr_images, torch.Tensor), "images must be a tensor with BXCXHXW or CXHXW"
    if max_numbers == 1 and max_numbers <= sr_images.shape[0]:
        indices = random.choice(list(range(sr_images.shape[0])))
        lr_image = lr_images[indices, :, :, :]
        sr_image = sr_images[indices, :, :, :]
        hr_image = hr_images[indices, :, :, :]
        if cfg.INPUT.NORM == 255:
            sr_image.clamp_(0, 255)
        else:
            sr_image.clamp_(0, 1)

    elif max_numbers > 1 and max_numbers <= sr_images.shape[0]:
        lr_image = lr_images[:max_numbers, :, :, :]
        sr_image = sr_images[:max_numbers, :, :, :]
        hr_image = hr_images[:max_numbers, :, :, :]

        lr_image = make_grid(lr_image, nrow=max_numbers, padding=cfg.DEBUG.PADDING)
        sr_image = make_grid(sr_image, nrow=max_numbers, padding=cfg.DEBUG.PADDING)
        hr_image = make_grid(hr_image, nrow=max_numbers, padding=cfg.DEBUG.PADDING)

        if cfg.INPUT.RANGE == 255:
            sr_image.clamp_(0, 255)
        else:
            sr_image.clamp_(0, 1)

        log_writer.add_images(f"{flag}/lr_image", lr_image, epoch, dataformats="CHW")
        log_writer.add_images(f"{flag}/sr_image", sr_image, epoch, dataformats="CHW")
        log_writer.add_images(f"{flag}/hr_image", hr_image, epoch, dataformats="CHW")


def train(cfg, train_loader, validation_loader, model, criterion, optimizer, batch_aug, epoch, args,
            batch_iter, total_batch, train_batch, no_warmup_total_batch, log_writer,
            calculate_psnr, train_metric, denormalize, ssim_criterion, batch_count, batch_total_count):

    model.train()
    device = model.device
    loader_length = len(train_loader)

    for batch_idx, data in enumerate(train_loader):
        batch_start = time.time()
        if cfg.TRAIN.COSINE:
            # cosine learning rate
            lr = adjust_learning_rate_for_cosine_decay(
                cfg, epoch, batch_idx+1, optimizer, train_batch, no_warmup_total_batch, loader_length
            )
        else:
            # step learning rate
            lr = adjust_learning_rate(
                cfg, epoch, batch_idx+1, optimizer, loader_length
                )
        # forward
        lr_images, hr_images = data[0], data[1]

        if cfg.TRAIN.FP16:
            lr_images = lr_images.half()
            hr_images = hr_images.half()

        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # batch augments
        if cfg.DATAAUG.MIXUP or cfg.DATAAUG.CUTMIX or cfg.DATAAUG.CUTBLUR:
            lr_images, hr_images = batch_aug(lr_images, hr_images)

        sr_images = model(lr_images)

        if cfg.LOSS.MS_SSIM:
            l1_losses = criterion(sr_images, hr_images)
            ssim = ssim_criterion(sr_images, hr_images)
            losses = (1 - cfg.LOSS.ALPHA) * l1_losses + cfg.LOSS.ALPHA *  (1 - ssim)
        elif cfg.LOSS.SSIM:
            l1_losses = criterion(sr_images, hr_images)
            ssim = ssim_criterion(sr_images, hr_images)
            losses = cfg.LOSS.ALPHA * l1_losses + cfg.LOSS.ALPHA * (1 - ssim)
        else:
            losses = criterion(sr_images, hr_images)

        # loss regularization
        if cfg.TRAIN.GRAD_ACCUMULATE and cfg.TRAIN.STEP:
            losses = losses / cfg.TRAIN.GRAD_ACCUMULATE_STEP

        batch_psnr = calculate_psnr(sr_images, hr_images)

        optimizer.zero_grad()
        if cfg.TRAIN.MIX:
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()

        if cfg.TRAIN.GRAD_ACCUMULATE:
            if (batch_idx + 1) % cfg.TRAIN.GRAD_ACCUMULATE_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        batch_time = time.time() - batch_start

        batch_iter += 1
        batch_idx += 1

        train_metric["loss"].update(losses.data.item())
        train_metric["psnr"].update(batch_psnr.data.item())
        if args.rank == 0:
            if cfg.LOSS.SSIM or cfg.LOSS.MS_SSIM:
                print("Training Epoch: [{}/{}] batchidx:[{}/{}] batchiter: [{}/{}] batch_losses: {:.4f} l1_loss: {:.4f} ssim: {:.4f} psnr: {:.4f} LearningRate: {:.10f} Batchtime: {:.4f}s Datetime: {}".format(
                    epoch,
                    cfg.TRAIN.MAX_EPOCHS,
                    batch_idx,
                    train_batch,
                    batch_iter,
                    total_batch,
                    losses.data.item(),
                    l1_losses.data.item(),
                    ssim.data.item(),
                    batch_psnr.data.item(),
                    lr,
                    batch_time,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            else:
                print("Training Epoch: [{}/{}] batchidx:[{}/{}] batchiter: [{}/{}] batch_losses: {:.4f} psnr: {:.4f} LearningRate: {:.10f} Batchtime: {:.4f}s Datetime: {}".format(
                    epoch,
                    cfg.TRAIN.MAX_EPOCHS,
                    batch_idx,
                    train_batch,
                    batch_iter,
                    total_batch,
                    losses.data.item(),
                    batch_psnr.data.item(),
                    lr,
                    batch_time,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        if args.rank == 0:
            # batch record
            record_log(log_writer, losses, lr, batch_iter, batch_time)
            if batch_idx == 1:
                if cfg.DEBUG.SHOW_SR_IMAGE:
                    draw_images(cfg, epoch, log_writer, lr_images, sr_images, hr_images, denormalize, flag="train")

        # validation for iter
        if cfg.VALIDATION.USE and (batch_iter % cfg.VALIDATION.ITER) == 0:
            batch_count += 1
            val_loss, val_psnr = val(cfg, validation_loader, model, criterion, batch_count, batch_total_count, args, log_writer, calculate_psnr, denormalize=denormalize)

            if args.rank == 0:
                best_loss = val_loss
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, cfg.CHEKCPOINTS.CKPT_PATH + '/' + cfg.MODEL.BACKBONE.NAME + f'_best_loss_{best_loss}' + '.pth')

                best_psnr = val_psnr
                state_dict = translate_state_dict(model.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state_dict, cfg.CHEKCPOINTS.CKPT_PATH + '/' + cfg.MODEL.BACKBONE.NAME + f'_best_psnr_{best_psnr}' + '.pth')

    # epoch record
    record_scalars(log_writer, train_metric["loss"].average, train_metric["psnr"].average, epoch, flag="train")

    return batch_iter


def val(cfg, val_loader, model, criterion, epoch, total_epoch, args, log_writer, calculate_psnr, denormalize):
    model.eval()
    device = model.device

    epoch_losses, epoch_psnr = 0.0, torch.tensor(0.0).cuda().float()
    iter_idx = torch.tensor(0.0).cuda().float()

    for batch_idx, data in enumerate(val_loader):
        # forward
        lr_images, hr_images = data[0], data[1]
        iter_idx += lr_images.shape[0]
        # print(f"val: {batch_idx}+{lr_images.shape[0]}")

        if cfg.TRAIN.FP16:
            lr_images = lr_images.half()
            hr_images = hr_images.half()

        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        with torch.no_grad():
            start_time = time.time()
            sr_images = model(lr_images)
            batch_time = time.time() - start_time

            # losses
            batch_losses = criterion(sr_images, hr_images)
            epoch_losses += batch_losses.data.item()

            if cfg.VALIDATION.KEEP_PSNR:
                batch_psnr = get_psnr(sr_images, hr_images)
                epoch_psnr += batch_psnr
            else:
                batch_psnr = calculate_psnr(sr_images, hr_images)
                epoch_psnr += batch_psnr.data.item()

        # show the last batch idx images
        if cfg.DEBUG.SHOW_SR_IMAGE:
            if args.rank == 0:
                if batch_idx == len(val_loader) - 1:
                    draw_images(cfg, epoch, log_writer, lr_images, sr_images, hr_images, denormalize, flag="val")

    model.train()

    epoch_losses = epoch_losses / float(batch_idx + 1)

    if cfg.VALIDATION.KEEP_PSNR:

        dist.all_reduce(epoch_psnr, op=dist.reduce_op.SUM)
        dist.all_reduce(iter_idx, op=dist.reduce_op.SUM)

        output_psnr = epoch_psnr.data.item() / iter_idx.data.item()
    else:
        output_psnr = epoch_psnr / float(batch_idx + 1)

    if args.rank == 0 :
        print(f"Validation Epoch: [{epoch}/{total_epoch}] mean_losses : {epoch_losses} mean_psnr: {output_psnr}")

        record_scalars(log_writer, epoch_losses, output_psnr, epoch, flag="val")

    return epoch_losses, output_psnr


def adjust_learning_rate(cfg, epoch, batch_idx, optimizer, loader_length):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    total_epochs = cfg.TRAIN.MAX_EPOCHS
    warm_epochs = cfg.TRAIN.WARM_EPOCHS
    if epoch < warm_epochs:
        epoch += float(batch_idx + 1) / loader_length
        lr_adj = 1. / ngpus_per_node * \
            (epoch * (ngpus_per_node - 1) / warm_epochs + 1)
    elif epoch < int(0.3 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1e-1
    else:
        lr_adj = 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.TRAIN.BASE_LR * lr_adj
    return cfg.TRAIN.BASE_LR * lr_adj


if __name__ == '__main__':
    # debug
    args = parser.parse_args()
    cfg = Config(args.config_file)()
    if cfg.SEED is not None:
        setup_seed(cfg.SEED)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        print("ngpus_per_node", ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("ngpus_per_node", ngpus_per_node)
        main_worker(args.gpu, ngpus_per_node, args)

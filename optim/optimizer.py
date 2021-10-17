"""
-*- coding:utf-8 -*-
@author:   GiantPandaSR
@date:     2021-02-11
@describe:
"""
import math 
import torch.optim as optim


class Optimizer(object):
    def __init__(self, cfg):
        super(Optimizer, self).__init__()
        self.cfg = cfg
        self.optimizer_type = self.cfg.TRAIN.OPTIMIZER
        self.base_lr = self.cfg.TRAIN.BASE_LR
        self.momentum = self.cfg.TRAIN.MOMENTUM
        self.weight_decay = self.cfg.TRAIN.WEIGHT_DECAY

    def optimizer(self, model):

        if self.optimizer_type == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.base_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "ADAMW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.base_lr,
                weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer


def adjust_learning_rate_for_cosine_decay(cfg, epoch, batch_idx, optimizer, total_train_sampler, total_sampler_batch, loader_length):
    """Sets the learning rate with  the cosine decay
    """
    total_epochs = cfg.TRAIN.MAX_EPOCHS
    warm_epochs = cfg.TRAIN.WARM_EPOCHS
    
    if epoch < warm_epochs:
        epoch += float(batch_idx + 1) / loader_length
        lr_adj = 1. / loader_length * (epoch * (loader_length - 1) / warm_epochs + 1)
    else:
        batch_sample = total_train_sampler * (epoch - warm_epochs) + batch_idx 
        lr_adj = 1/2 * (1 + math.cos(batch_sample * math.pi / total_sampler_batch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.TRAIN.BASE_LR * lr_adj
    
    return cfg.TRAIN.BASE_LR * lr_adj

        
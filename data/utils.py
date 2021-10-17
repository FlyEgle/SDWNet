"""
-*- coding:utf-8 -*-
@author  : GiantPandaSR
@date    : 2021-02-10
@describe: utils for sr model
"""
import os
import cv2
import math
import torch
import random
import numpy as np
import urllib.request as urt
import multiprocessing as mp
import torchvision.transforms as transforms

from io import BytesIO
from PIL import Image

from data.augments import *


class Metric_rank:
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

    @property
    def average(self):
        return self.sum / self.n


class CalculateSSIM:
    """SSIM calculate
    img1 and img2 have range [0, 255]
    """
    def __init__(self, cfg):
        pass

    def __call__(self, img1, img2):
        pass


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255] """

    def __init__(self, cfg):
        self.cfg = cfg
        self.name = "PSNR"
        self.norm = cfg.INPUT.NORM
        self.range = cfg.INPUT.RANGE
        self.mean = torch.tensor(cfg.INPUT.MEAN, dtype=torch.float32)
        self.std = torch.tensor(cfg.INPUT.STD, dtype=torch.float32)

    def __call__(self, img1, img2):
        image1 = img1.clone()
        image2 = img2.clone()
        if self.norm:
            device = img1.device
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            image1.mul_(self.std[None, :, None, None]).add_(self.mean[None, :, None, None])
            image2.mul_(self.std[None, :, None, None]).add_(self.mean[None, :, None, None])

        if self.range != 255:
            image1 *= 255
            image2 *= 255

        mse = torch.mean((image1 - image2) ** 2)
        psnr =  20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr

class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


def sliding_window(image, patch_size: tuple, step: int, show_debug: bool = False) -> list:
    """sliding the patch size window, crop from the whole images
    Args:
            image: PIL or ndarray.
            patch_size: a tuple for (128, 128).
            step: window step.
    Returns:
            crop_image_list: a list of crop image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if step == 0:
        h, w = image.shape[0], image.shape[1]  # 720, 1280
        w_iter, h_iter = w // patch_size[0], h // patch_size[1]
        crop_image_list = []
        for i in range(h_iter):
            for j in range(w_iter):
                bbox = (i*patch_size[0], j*patch_size[0],
                        (i+1)*patch_size[0], (j+1)*patch_size[0])
                crop_image = image[bbox[0]:bbox[2], bbox[1]: bbox[3]]
                if show_debug:
                    crop_image = Image.fromarray(crop_image)
                    crop_image.save(f"/data/jiangmingchao/patches/{i}.png")
                    cv2.rectangle(image,
                                  (i*patch_size[0], j*patch_size[0]),
                                  ((i+1)*patch_size[0], (j+1)*patch_size[0]),
                                  (255, 255, 0),
                                  2,
                                  )

                crop_image_list.append(Image.fromarray(crop_image))

        if show_debug:
            cv2.imwrite("1.jpg", image)

    else:
        h, w = image.shape[0], image.shape[1]
        step_w_iter, step_h_iter = (w - patch_size[0]) // step, (h - patch_size[0]) // step
        crop_image_list = []
        for i in range(step_h_iter):
            for j in range(step_w_iter):
                bbox = (i * step, j * step, patch_size[0] + i * step, patch_size[1] + j * step)
                crop_image = image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
                print(crop_image.shape)
                crop_image_list.append(Image.fromarray(crop_image))

    return crop_image_list


def get_patches(lr, hr, patch_size: tuple, step: int, max_nums: int) -> (list, list):

    lr_patches = sliding_window(lr, patch_size, step)
    hr_patches = sliding_window(hr, patch_size, step)

    if len(lr_patches) <= max_nums:
        return lr_patches, hr_patches

    else:
        indices = [i for i in range(len(lr_patches))]
        samples = random.sample(indices, max_nums)
        new_lr_patches, new_hr_patches = [], []
        for i in range(len(indices)):
            new_lr_patches.append(lr_patches[indices[i]])
            new_hr_patches.append(hr_patches[indices[i]])
        return new_lr_patches, new_hr_patches


class BatchAug(object):
    """Make the batch augments for lr, hr
    """
    def __init__(self, cfg):
        super(BatchAug, self).__init__()
        self.cfg = cfg
        self.prob = 1.0
        self.beta = 1.0
        self.mixup = MixUp(self.prob, self.beta)
        self.cutmix = CutMix(self.prob, self.beta)
        self.cutblur = CutBlur(self.prob, self.beta)

    def __call__(self, batch_lr, batch_hr):
        if self.cfg.DATAAUG.MIXUP:
            batch_lr, batch_hr = self.mixup(batch_lr, batch_hr)
        elif self.cfg.DATAAUG.CUTMIX:
            batch_lr, batch_hr = self.cutmix(batch_lr, batch_hr)
        elif self.cfg.DATAAUG.CUTBLUR:
            batch_lr, batch_hr = self.cutblur(batch_lr, batch_hr)

        return batch_lr, batch_hr


if __name__ == "__main__":
    # image = "/data/jiangmingchao/patches/00000000.jpg"
    # img = Image.open(image)
    # crop_image_list = sliding_window(
    #     img, (320, 320), step=32, show_debug=True)
    # print(len(crop_image_list))

    lr_path = "/data/jiangmingchao/data/SR_NTIRE2021/data/train/train_jpeg.log"
    hr_path = "/data/jiangmingchao/data/SR_NTIRE2021/data/train/train_gt.log"





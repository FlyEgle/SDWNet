"""
-*- coding:utf-8 -*-
@author  : GiantPandaSR
@date    : 2021-02-09
@describe: The basic data augments for SR work.
"""
from __future__ import division

import PIL
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr):
        for t in self.transforms:
            if t is not None:
                lr, hr = t(lr, hr)
        return lr, hr


class RandomCrop(object):
    def __init__(self, crop_size: tuple):
        self.crop_size = crop_size

    def __call__(self, lr, hr):
        w, h = lr.size[0], lr.size[1]
        x1 = random.choice(list(range(w - self.crop_size[0])))
        y1 = random.choice(list(range(h - self.crop_size[1])))
        x2 = x1 + self.crop_size[0]
        y2 = y1 + self.crop_size[1]

        bbox = (x1, y1, x2, y2)

        if isinstance(lr, PIL.Image.Image):
            lr = lr.crop(bbox)
            hr = hr.crop(bbox)
        elif isinstance(lr, np.ndarray):
            lr, hr = Image.fromarray(lr), Image.fromarray(hr)
            lr = lr.crop(bbox)
            hr = hr.crop(bbox)
        else:
            raise TypeError("data must be pil or array!!")

        return lr, hr


class CenterCrop(object):
    def __init__(self, crop_size:tuple):
        self.crop_size = crop_size

    def __call__(self, lr, hr):
        lr = F.center_crop(lr, self.crop_size)
        hr = F.center_crop(hr, self.crop_size)

        return lr, hr


class RandomHorizonFlip(object):
    def __init__(self, proba=0.5):
        self.proba = proba

    def __call__(self, lr, hr):
        if random.random() > self.proba:
            if isinstance(lr, PIL.Image.Image):
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(lr, np.ndarray):
                lr, hr = Image.fromarray(lr), Image.fromarray(hr)
                lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
                lr, hr = np.asarray(lr), np.asarray(hr)
            else:
                raise TypeError("data must be pil or array!!")
        return lr, hr


class RandomVerticalFlip(object):
    def __init__(self, proba=0.5):
        self.proba = proba

    def __call__(self, lr, hr):
        if random.random() > self.proba:
            if isinstance(lr, PIL.Image.Image):
                lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            elif isinstance(lr, np.ndarray):
                lr, hr = Image.fromarray(lr), Image.fromarray(hr)
                lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
                lr, hr = np.asarray(lr), np.asarray(hr)
            else:
                raise TypeError("data must be pil or array!!")
        return lr, hr


class RandomRotate(object):
    def __init__(self, prob=0.3):
        self.prob = prob
        self.ROTATE_ANGLE = [90, 180, 270]

    def __call__(self, lr, hr):
        if random.random() > self.prob:
            angle = random.choice(self.ROTATE_ANGLE)
            if isinstance(lr, PIL.Image.Image):
                lr = lr.rotate(angle)
                hr = hr.rotate(angle)
            elif isinstance(lr, np.ndarray):
                lr, hr = Image.fromarray(lr), Image.fromarray(hr)
                lr = lr.rotate(angle)
                hr = hr.rotate(angle)
                lr, hr = np.asarray(lr), np.asarray(hr)
            else:
                raise TypeError("data must be pil or array!!")
        return lr, hr


class RandomGaussianBlur(object):
    """PIL image
    """
    def __init__(self, prob=0.2, kernel_size=5):
        super(RandomGaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.gaussian_blur = transforms.GaussianBlur(kernel_size)
        self.prob = prob

    def __call__(self, lr, hr):
        if random.random() < self.prob:
            lr = self.gaussian_blur(lr)
            return lr, hr


class RandomGamma(object):
    """gamma
    """
    def __init__(self, prob=0.5):
        self.prob = prob
        self.gamma_prob = 1.0

    def __call__(self, lr, hr):
        if random.random() > self.prob:
            lr = transforms.functional.adjust_gamma(lr, self.gamma_prob)
            hr = transforms.functional.adjust_gamma(hr, self.gamma_prob)
        return lr, hr


class RandomSaturation(object):
    """Saturation
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, lr, hr):
        if random.random() > self.prob:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            lr = transforms.functional.adjust_gamma(lr, sat_factor)
            hr = transforms.functional.adjust_gamma(hr, sat_factor)
        return lr, hr


class ToTensor(object):
    """tensor range(0-1)
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, lr, hr):
        lr, hr = self.to_tensor(lr), self.to_tensor(hr)
        return lr, hr


class ToTensor2(object):
    """tensor range(0-255)
    """
    def __init__(self):
        super(ToTensor2, self).__init__()

    def __call__(self, lr, hr):

        if type(lr) == np.ndarray:
            lr_tensor = torch.as_tensor(np.ascontiguousarray(lr.transpose(2, 0, 1)), dtype=torch.float)
            hr_tensor = torch.as_tensor(np.ascontiguousarray(hr.transpose(2, 0, 1)), dtype=torch.float)
        else:
            lr_array = np.array(lr)
            lr_tensor = torch.as_tensor(np.ascontiguousarray(lr_array.transpose(2, 0, 1)), dtype=torch.float)

            if hr is not None:
                hr_array = np.array(hr)
                hr_tensor = torch.as_tensor(np.ascontiguousarray(hr_array.transpose(2, 0, 1)), dtype=torch.float)
                return lr_tensor, hr_tensor
            else:
                return lr_tensor, None


class RandomRGB(object):
    """Random permute R,G,B
    """
    def __init__(self, prob=0.2):
        self.prob = prob
        self.channel_range = [
            [0,2,1],
            [1,0,2],
            [1,2,0],
            [2,0,1],
            [2,1,0]
        ]

    def __call__(self, lr, hr):
        """Random Permute the R,G,B
        Args:
            lr: pil or ndarray
            hr: pil or ndarray
        Returns:
            PIL image.
        """
        if random.random() < self.prob:
            random_permute = random.choice(self.channel_range)

            if isinstance(lr, Image.Image):
                lr = np.array(lr)
                hr = np.array(hr)
            elif isinstance(lr, np.ndarray):
                lr = lr
                hr = hr
            else:
                raise TypeError("input must be PIL or np.ndarray format!!!")

            lr = lr[:, :, random_permute]
            hr = hr[:, :, random_permute]

            lr = Image.fromarray(lr)
            hr = Image.fromarray(hr)

        return lr, hr


class Blend(object):
    """Blend the image
    """
    def __init__(self, prob=0.5, alpha=0.6):
        super(Blend, self).__init__()
        self.prob = prob
        self.alpha = alpha

    def __call__(self, lr, hr):
        """
        Args:
            lr: torch.Tensor
            hr: torch.Tensor
        Returns:
            torch.Tensor
        """
        if np.random.random() < prob and self.alpha > 0:
            if not isinstance(lr, torch.Tensor):
                raise TypeError("Input must be the tensor!!!!")
            c = torch.empty((lr.shape[0], 3, 1, 1), device=lr.device).uniform_(0, 255)
            rlr = c.repeat((1, 1, lr.shape[2], lr.shape[3]))
            rhr = c.repeat((1, 1, hr.shape[2], hr.shape[3]))

            v = np.random.uniform(self.alpha, 1)
            lr = v * lr + (1 - v) * rlr
            hr = v * hr + (1 - v) * rhr

        return lr, hr


class MixUp(object):
    """Mixup for lr, hr tensor
    """
    def __init__(self, prob=0.5, beta=1.0):
        self.prob = prob
        self.beta = beta

    def __call__(self, lr, hr):
        """
        Args:
            lr: torch.tensor
            hr: torch.tensor
        Returns:
            torch.tensor
        """
        if np.random.random() < self.prob and self.beta > 0.0:
            if not isinstance(lr, torch.Tensor):
                raise TypeError("Input must be the tensor!!!!")

            v = np.random.beta(self.beta, self.beta)
            r_index = torch.randperm(lr.shape[0]).to(lr.device)

            lr = v * lr + (1 - v) * lr[r_index]
            hr = v * hr + (1 - v) * hr[r_index]
            return lr, hr


class CutBlur(object):
    """CutOut for lr, hr tensor
    """
    def __init__(self, prob=1.0, beta=1.0):
        super(CutBlur, self).__init__()
        self.prob = prob
        self.beta = beta

    def __call__(self, lr, hr):
        """
        Args:
            lr : torch.tensor
            hr : torch.tensor
        Returns:
            torch.tensor
        """
        if np.random.random() < self.prob and self.beta > 0.0:
            if not isinstance(lr, torch.Tensor):
                raise TypeError("Input must be the tensor!!!!")

            cut_ratio = np.random.randn() * 0.01 + self.beta
            cut_ratio = min(cut_ratio, 0.25)                   # fix the 25% crop size

            h, w = lr.shape[2], lr.shape[3]
            ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
            cy = np.random.randint(0, h - ch + 1)
            cx = np.random.randint(0, w - cw + 1)

            # insdie
            if np.random.random() > 0.5:
                hr[..., cy:cy + ch, cx:cx + cw] = lr[..., cy:cy + ch, cx:cx + cw]
            # outside
            else:
                hr_aug = lr.clone()
                hr_aug[..., cy:cy + ch, cx:cx + cw] = hr[..., cy:cy + ch, cx:cx + cw]
                hr = hr_aug

            return lr, hr


class CutMix(object):
    """CutMix for lr, hr tensor
    """
    def __init__(self, prob=0.5, beta=1.0):
        self.prob = prob
        self.beta = beta

    def _cutmix(self, image):
        cut_ratio = np.random.randn() * 0.01 + self.beta
        cut_ratio = min(cut_ratio, 0.25)                   # fix the 25% crop size

        h, w = image.shape[2], image.shape[3]
        ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)

        fcy = np.random.randint(0, h - ch + 1)
        fcx = np.random.randint(0, w - cw + 1)
        tcy, tcx = fcy, fcx
        rindex = torch.randperm(image.shape[0]).to(image.device)

        info = {
                "rindex": rindex,
                "ch": ch,
                "cw": cw,
                "tcy": tcy,
                "tcx": tcx,
                "fcy": fcy,
                "fcx": fcx,
                }

        return info

    def __call__(self, lr, hr):
        """
        Args:
            lr : torch.tensor
            hr : torch.tensor
        Returns:
            torch.tensor
        """
        if np.random.random() < self.prob and self.beta > 0.0:
            if not isinstance(lr, torch.Tensor):
                raise TypeError("Input must be the tensor!!!!")

            c = self._cutmix(lr)
            rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
            tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

            lr[..., tcy:tcy + ch, tcx:tcx + cw] = lr[rindex, :, fcy:fcy + ch, fcx:fcx + cw]
            hr[..., tcy:tcy + ch, tcx:tcx + cw] = hr[rindex, :, fcy:fcy + ch, fcx:fcx + cw]

            return lr, hr


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(self.mean, self.std)

    def __call__(self, lr, hr):
        lr, hr = self.normalize(lr), self.normalize(hr)
        return lr, hr


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _denormalize(self, input, inplace = False):

        if inplace:
            ret = input
        else:
            ret = input.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)

        if len(input.shape) == 4:
            ret.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])
        elif len(input.shape) == 3:
            ret = torch.add(torch.mul(ret,std[:, None, None]),mean[:, None, None])

            # ret.mut_(std[:, None, None]).add_(mean[:, None, None])

        return ret

    def __call__(self, images):
        images = self._denormalize(images)
        return images


if __name__ == "__main__":
    pd_file = "http://ai-train-datasets.oss-cn-zhangjiakou-internal.aliyuncs.com/jiangmingchao/super2021dataset/train/train_blur_bicubic/X4/000/00000000.png"
    gt_file = "http://ai-train-datasets.oss-cn-zhangjiakou-internal.aliyuncs.com/jiangmingchao/super2021dataset/train/train_sharp/000/00000000.png"

    import urllib.request as urt
    from io import BytesIO
    pd = Image.open(BytesIO(urt.urlopen(pd_file).read()))
    gt = Image.open(BytesIO(urt.urlopen(gt_file).read()))

    print(pd.size)
    print(gt.size)














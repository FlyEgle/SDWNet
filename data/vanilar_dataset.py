import os
import cv2
import time
import random
import json
import urllib
import numpy as np
from PIL import Image
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from tqdm import tqdm
from data.augments import *
from data.utils import get_patches
from config.Config import Config


class NTIRE_Track1(Dataset):
    def get_filelist(self, path, ext):
        if not isinstance(ext, list):
            ext_filter = [ext]
        else:
            ext_filter = ext

        result = []

        for maindir, subdir, filename_list in os.walk(path):
            for fname in filename_list:
                apath = os.path.join(maindir, fname).replace('//', '/')
                ext = os.path.splitext(apath)[1]
                if ext in ext_filter:
                    result.append(apath)
        return result

    def get_gt_path(self, GT_PATH, filename, ext):
        prob_fname = os.path.join(GT_PATH, filename + ext)
        return prob_fname

    def get_paired_list(self, LR_PATH, GT_PATH):
        lr_list = self.get_filelist(LR_PATH, ext='.png')
        gt_list = []
        for lr in lr_list:
            lr_fname = os.path.splitext(os.path.split(lr)[1])[0]
            gt_path = self.get_gt_path(GT_PATH, lr_fname, ext='.png')
            gt_list.append(gt_path)

        return [lr_list, gt_list]

    def __init__(self, LR_PATH, GT_PATH, input_size=[320, 320], patch=True,
                 base_transforms=None, augment_transforms=None, batch_transforms=None):
        super().__init__()
        self.lr_path = LR_PATH
        self.gt_path = GT_PATH
        self.paired_list = self.get_paired_list(self.lr_path, self.gt_path)
        self.base_transforms = base_transforms
        self.augment_transforms = augment_transforms
        self.batch_transforms = batch_transforms

    def __len__(self):
        return min(len(self.paired_list[0]), len(self.paired_list[1]))

    def __getitem__(self, index):
        lr_file_path = self.paired_list[0][index]
        gt_file_path = self.paired_list[1][index]

        # TODO: all the augmentations need to be implemented.
        lr_img = cv2.imread(lr_file_path)
        gt_img = cv2.imread(gt_file_path)

        return lr_img, gt_img


class NTIRE_Track2(Dataset):

    def __init__(self, cfg, train=True):
        super(NTIRE_Track2, self).__init__()
        self.cfg = cfg
        self.train = train
        self.range = self.cfg.INPUT.RANGE

        if self.train:
            self.lr_path = self.cfg.DATA.TRAIN.LR_PATH
            self.hr_path = self.cfg.DATA.TRAIN.HR_PATH
        else:
            self.lr_path = self.cfg.DATA.VALIDATION.LR_PATH
            self.hr_path = self.cfg.DATA.VALIDATION.HR_PATH

        self.total_pair= self._make_pair(self.lr_path, self.hr_path)
        if self.train:
            if self.cfg.TRAIN.SAMPLER:
                self.paired_list = random.sample(self.total_pair, int(len(self.total_pair) * self.cfg.TRAIN.SAMPLER))
            else:
                self.paired_list = self.total_pair
        else:
            self.paired_list = self.total_pair

        for _ in range(5):
            random.shuffle(self.paired_list)

        self.data_length = [x for x in range(len(self.paired_list))]

        self.mean = self.cfg.INPUT.MEAN
        self.std  = self.cfg.INPUT.STD
        self.norm = self.cfg.INPUT.NORM
        self.patch_size = self.cfg.PATCH.PATCH_SIZE
        if self.train:
            self.base_transforms = self.train_preprocess()
        else:
            self.base_transforms = self.val_preproces()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_preprocess(self):
        # TODO: add the important patch from crop, add blur
        base_transforms = Compose([
            RandomRGB() if self.cfg.DATAAUG.RGB else None,
            RandomCrop(self.patch_size) if self.cfg.DATAAUG.RANDOM_CROP else None,
            RandomHorizonFlip() if self.cfg.DATAAUG.HFLIP else None,
            RandomVerticalFlip() if self.cfg.DATAAUG.VFLIP else None,
            RandomRotate() if self.cfg.DATAAUG.ROTATE else None,
            RandomGamma() if self.cfg.DATAAUG.GAMMA else None,
            RandomSaturation() if self.cfg.DATAAUG.SATURATION else None,
            ToTensor2() if self.range == 255 else ToTensor(),
            Normalize(self.mean, self.std) if self.norm else None
            ])

        return base_transforms

    def val_preproces(self):
        if self.norm:
            base_transforms = Compose([
                CenterCrop(self.patch_size) if self.cfg.DATAAUG.CENTERCROP else None,
                ToTensor2() if self.range == 255 else ToTensor(),
                Normalize(self.mean, self.std)
            ])
        else:
            base_transforms = Compose([
                CenterCrop(self.patch_size) if self.cfg.DATAAUG.CENTERCROP else None,
                ToTensor2() if self.range == 255  else ToTensor(),
            ])
        return base_transforms

    def _make_dict(self, data_list):
        data_dict = {}
        for data in data_list:
            data_json = json.loads(data)
            data_dict[data_json["image_key"].split('.')[0]] = data_json["image_path"]
        return data_dict

    def _make_pair(self, lr_path, hr_path):
        lr_dict = self._make_dict([x.strip() for x in open(lr_path).readlines()])
        hr_dict = self._make_dict([x.strip() for x in open(hr_path).readlines()])
        assert len(lr_dict) == len(hr_dict), "the lr and hr data length must be same!!!"
        pair_list = []
        for image_key, lr_image_path in lr_dict.items():
            hr_image_path = hr_dict[image_key]
            pair_list.append([lr_image_path, hr_image_path])
        return pair_list

    def _load_image(self, img_path):
        if img_path[:4] == 'http':
            img = Image.open(BytesIO(urllib.request.urlopen(img_path).read())).convert('RGB')
            # img = np.asarray(img)
        else:
            img = Image.open(img_path)
            img = img.convert("RGB")
        return img

    # TODO: To get the patches need to load the all patches to the mem, need to speed up
    def _build_patches(self, shuffle=True):
        self.patches_list = [[], []]
        for i in tqdm(range(len(self.paired_list))):
            start_time = time.time()
            lr_image, hr_image = self._load_image(self.paired_list[i][0]), self._load_image(self.paired_list[i][1])
            print(f"load image time is: {time.time() - start_time}")
            start_time = time.time()
            lr_patch_list, hr_patch_list = get_patches(lr_image, hr_image, self.patch_size, self.step, self.max_nums)
            print(f"get_patches time is: {time.time() - start_time}")
            self.patches_list[0].extend(lr_patch_list)
            self.patches_list[1].extend(hr_patch_list)
        if shuffle:
            random.shuffle(self.patches_list)
        return self.patches_list

    def __len__(self):
        return len(self.paired_list)

    def __getitem__(self, index):
        for _ in range(20):
            try:
                lr_img_path, gt_img_path = self.paired_list[index][0], self.paired_list[index][1]
                lr_img = self._load_image(lr_img_path)
                gt_img = self._load_image(gt_img_path)
                break
            except Exception as e:
                index = random.choice(self.data_length)

        if self.base_transforms is not None:
            lr_img, gt_img = self.base_transforms(lr_img, gt_img)
        
        return lr_img, gt_img


if __name__ == "__main__":
    cfg = Config("/data/jiangmingchao/data/SR_NTIRE2021/config/wavelet_two_stage/wavelet.yaml")
    dataset = NTIRE_Track2(train=False)

    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(i)
    print(len(data))

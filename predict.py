"""
Inference model with TTA methods to improve our result
"""
import os
import cv2
import time
import json
import urllib
import random
import imageio
import argparse
import numpy as np
import torch.nn.functional as F

import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp

from io import BytesIO
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.augments import *
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_Upsample_Deep import AtrousNet_billinear_Wide as AtrousNetEluUpWide
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import utils as vutils
from config.Config import Config


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class single_image_loader(Dataset):
    def __init__(self, args, mode="jpg2png"):
        super(single_image_loader, self).__init__()
        self.args = args
        self.mode = mode

        self.base_transforms = self.infer_preprocess()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.image_folder = self.args.test_data
        self.file_list = self._get_image_list()

    def _get_image_list(self):
        image_path_list = [os.path.join(self.image_folder, x)
                           for x in os.listdir(self.image_folder)]
        return image_path_list

    def infer_preprocess(self):
        base_transforms = ToTensor2()
        return base_transforms

    def _load_image(self, img_path, num_retry=20):
        img = cv2.imread(img_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.args.TTA == "src":
            img = Image.fromarray(img)
        elif self.args.TTA == "rot_90":
            img_rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = Image.fromarray(img_rot_90)
        elif self.args.TTA == "rot_180":
            img_rot_180 = cv2.rotate(img, cv2.ROTATE_180)
            img = Image.fromarray(img_rot_180)
        elif self.args.TTA == "rot_270":
            img_rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = Image.fromarray(img_rot_270)
        elif self.args.TTA == "flip_h":
            img_flip_h = cv2.flip(img, 1)
            img = Image.fromarray(img_flip_h)
        elif self.args.TTA == "flip_v":
            img_flip_v = cv2.flip(img, 0)
            img = Image.fromarray(img_flip_v)
        elif self.args.TTA == "bgr":
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(img_bgr)

        return img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        lr_img_path = self.file_list[index]
        lr_img = self._load_image(lr_img_path)
        lr_image_key = lr_img_path.split('/')[-1]
        # if self.base_transforms:
        lr_img, _ = self.base_transforms(lr_img, lr_img)
        return lr_image_key, lr_img


def load_ckpt(model, weights):
    state_dict = torch.load(weights, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    return model


def save_images(args, output_images, output_images_keys):
    output_images = output_images.cpu().permute(0, 2, 3, 1).clamp_(0, 255).numpy()
    for i in range(output_images.shape[0]):
        output_img = output_images[i, :, :, :]
        output_key = output_images_keys[i]

        output_img = output_img.round().astype(np.uint8)

        if args.TTA == "src":
            image = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        elif args.TTA == "rot_90":
            image = cv2.rotate(output_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        elif args.TTA == "rot_180":
            image = cv2.rotate(output_img, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        elif args.TTA == "rot_270":
            image = cv2.rotate(output_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        elif args.TTA == "flip_h":
            image = cv2.flip(output_img, 1)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        elif args.TTA == "flip_v":
            image = cv2.flip(output_img, 0)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        elif args.TTA == "bgr":
            image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(args.save_images, output_key.replace(
                '.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SR DDP Inference')
    parser.add_argument('--weights_path', type=str,
                        default="/data/jiangmingchao/data/output_ckpt_with_logs/AtrousNetEluUpWide_unia_416x416_adamw_cosine_l1closs_ssim_120_0.6_crop_data_LRX2/ckpt/SR-AtrousNet_best_loss_6.3994362354278564.pth")
    parser.add_argument('--save_images', type=str,
                        default="/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data_1/src")
    parser.add_argument('--TTA', type=str, default="src")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--test_data', type=str, default="/data/jiangmingchao/data/dataset/SR_localdata/test_300")

    args = parser.parse_args()

    if not os.path.exists(args.save_images):
        os.mkdir(args.save_images)

    model = AtrousNetEluUpWide(3, 3)
    model = load_ckpt(model, args.weights_path)
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    dataset = single_image_loader(args)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=None,
        drop_last=False
    )

    for iter_id, batch_data in enumerate(dataloader):
        lr_key, lr_images = batch_data[0], batch_data[1]
        lr_images = lr_images.cuda()
        with torch.no_grad():
            sr_images = model(lr_images)

        save_images(args, sr_images, lr_key)
        print(f"Process the {iter_id} batch!!!")

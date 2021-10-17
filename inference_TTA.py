# -*- coding: utf-8 -*-
"""
@author    : GiantPandaSR
@data      : 2021-02-09
@describe  : Training with DDP or DataParallel 
"""
from __future__ import print_function
from config.Config import Config
from skimage.metrics import peak_signal_noise_ratio as psnr

# system
import warnings

warnings.filterwarnings("ignore")

import os 
import cv2 
import time
import argparse
import imageio
from tqdm import tqdm
# torch 
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data as data

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# model 
# from model.NTIRE2020_Deblur_top.uniA import AtrousNet
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_Upsample_Deep import AtrousNet_billinear_Wide as AtrousNet
# from model.NTIRE2021_Deblur.uniA_ELU.wavelet_SRCNN_remix import SRCNN as AtrousNet
# from model.NTIRE2021_Deblur.uniA_ELU.wavelet_deblur_remix import AtrousNet_wavlet_remix as AtrousNet
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import utils as vutils
from config.Config import Config
from data.augments import *
import json 

parser = argparse.ArgumentParser(description='SR DDP Inference')
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


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class single_image_loader(Dataset):
    def __init__(self, cfg, rec_path, mode="png2png"):
        super(single_image_loader, self).__init__()
        self.cfg = cfg
        self.range = self.cfg.INPUT.RANGE
        self.mode = mode
        self.lr_path = rec_path

        self.mean = self.cfg.INPUT.MEAN
        self.std = self.cfg.INPUT.STD
        self.norm = self.cfg.INPUT.NORM
        self.base_transforms = self.infer_preprocess()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.image_file = self.cfg.DATA.TRAIN.LR_PATH
        self.file_list = self._get_image_list()


    def _get_image_list(self):
        image_path_list = [json.loads(x.strip()) for x in open(self.lr_path).readlines()]
        return image_path_list

    def infer_preprocess(self):
        if self.norm:
            base_transforms = Compose([
                ToTensor2() if self.range == 255 else ToTensor(),
                Normalize(self.mean, self.std)
            ])
        else:
            base_transforms = Compose([
                ToTensor2() if self.range == 255 else ToTensor(),
            ])
        return base_transforms

    def _load_image(self, img_path, num_retry=20):
        for _ in range(num_retry):
            try:
                if img_path[:4] == 'http':
                    img = Image.open(BytesIO(urllib.request.urlopen(img_path).read())).convert('RGB')
                    # img = np.asarray(img)
                else:
                    img = cv2.imread(img_path, -1)
                    img_BGR = img
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    img_rot_180 = cv2.rotate(img, cv2.ROTATE_180)
                    img_rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    img_flip_h = cv2.flip(img, 1)
                    img_flip_v = cv2.flip(img, 0)
                    img = Image.fromarray(img)
                    img_rot_90 = Image.fromarray(img_rot_90)
                    img_rot_180 = Image.fromarray(img_rot_180)
                    img_rot_270 = Image.fromarray(img_rot_270)
                    img_flip_h = Image.fromarray(img_flip_h)
                    img_flip_v = Image.fromarray(img_flip_v)
                    img_BGR = Image.fromarray(img_BGR)
                break
            except Exception as e:
                time.sleep(5)
                print(f'Open image {img_path} failed, try again... resean is {e}')
        else:
            raise Exception(f'Open image: {img_path} failed!')

        return img, img_rot_90, img_rot_180, img_rot_270, img_flip_h, img_flip_v, img_BGR

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        lr_img_path = self.file_list[index]["image_path"]
        lr_image_key = self.file_list[index]["image_key"]
        lr_img, lr_img_rot_90, lr_img_rot_180, lr_img_rot_270, img_flip_h, img_flip_v, img_BGR = self._load_image(lr_img_path)

        if self.base_transforms is not None:
            lr_img, lr_img = self.base_transforms(lr_img, lr_img)
            lr_img_rot_90, lr_img_rot_90 = self.base_transforms(lr_img_rot_90,lr_img_rot_90)
            lr_img_rot_180, lr_img_rot_180 = self.base_transforms(lr_img_rot_180, lr_img_rot_180)
            lr_img_rot_270, lr_img_rot_270 = self.base_transforms(lr_img_rot_270, lr_img_rot_270)
            img_flip_h, img_flip_h = self.base_transforms(img_flip_h, img_flip_h)
            img_flip_v, img_flip_v = self.base_transforms(img_flip_v, img_flip_v)
            img_BGR, img_BGR = self.base_transforms(img_BGR, img_BGR)

        return lr_image_key, lr_img, lr_img_rot_90, lr_img_rot_180, lr_img_rot_270, img_flip_h, img_flip_v, img_BGR


def model_initializer(opt):
    # Non-distributed GPU Parallel
    device = opt['device']
    model_arch = "{}-{}".format("SR","AtrousNet")
    model = AtrousNet(3, 3)
    model_weights = torch.load(opt['model_pth'])
    model.load_state_dict(model_weights['state_dict'],strict=True)
    model = model.eval()
    model = model.to(device)
    return model

def inference(cfg, opt):
    model = model_initializer(opt)
    train_dataset = single_image_loader(cfg, opt['working_path'])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        num_workers=8)
    PSNR = []
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # Now only support single image inference
        # file_name = os.path.split(data[0][0])[1]
        # hr_image = cv2.imread(os.path.join(opt['HR_path'], file_name.replace('.jpg', '.png')))
        # hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        # print(file_name)
        # 1 -> 4
        img_data = data[1:]
        output_imgs = []
        for idx, img in enumerate(img_data):
            '''lr_img, lr_img_rot_90, lr_img_rot_180, lr_img_rot_270'''
            img = img.to(opt['device'])
            with torch.no_grad():
                output = model(img)
                for i in range(len(output)):
                    output_img = output[i,:,:,:].float().cpu()
                    file_name = data[0][i]

                    if cfg.INPUT.NORM:
                        denormalize = DeNormalize(cfg.INPUT.MEAN, cfg.INPUT.STD)
                        output_img = denormalize(output_img)
                    if cfg.INPUT.RANGE == 255:
                        output_img.clamp_(0,255)
                        output_img = output_img.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
                    else:
                        output_img.clamp_(0,1)
                        output_img = (output_img.permute(1, 2, 0).cpu().numpy()*255.0).round().astype(np.uint8)
                    if idx == 0:
                        output_imgs.append(output_img)
                    elif idx == 1:
                        output_imgs.append(cv2.rotate(output_img, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    elif idx == 2:
                        output_imgs.append(cv2.rotate(output_img, cv2.ROTATE_180))
                    elif idx == 3:
                        output_imgs.append(cv2.rotate(output_img, cv2.ROTATE_90_CLOCKWISE))
                    elif idx == 4:
                        output_imgs.append(cv2.flip(output_img, 1))
                    elif idx == 5:
                        output_imgs.append(cv2.flip(output_img, 0))
                    elif idx == 6:
                        output_imgs.append(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))

                    mean_img = np.mean(np.array(output_imgs), axis=0).round().astype(np.uint8)
                    # PSNR.append(psnr(hr_image, mean_img))
                    imageio.imwrite(os.path.join(opt['output_path'], file_name.replace('.jpg', '.png')), mean_img)
    # print('nAverage PSNR:{:.3f}'.format(sum(PSNR)/len(PSNR)))


if __name__ == '__main__':
    opt = dict()
    opt['device'] = "cuda"
    opt['model_pth'] = '/data/jiangmingchao/data/output_ckpt_with_logs/64w_2epoch_4e-5_pretrain_no_sr/ckpt/AtrousNetEluUpWide_best_psnr_27.845035552978516.pth'
    opt['working_path'] = '/data/jiangmingchao/data/dataset/SR_localdata/tta_3000.log'
    opt['output_path'] = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta"
    opt['config_file'] = "/data/jiangmingchao/data/SR_NTIRE2021/config/inference/ALL_TRAIN_CROP_INFERENCE.yaml"
    cfg = Config(opt['config_file'])()
    
    if not os.path.exists(opt['output_path']):
        os.makedirs(opt['output_path'])
    inference(cfg, opt)
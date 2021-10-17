"""inference the sr image reuslt with ddp
"""
import os
import cv2
import json
import argparse
import random
import time
import imageio
import numpy as np
import urllib
from io import BytesIO
import torch.nn.functional as F

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data as data

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.augments import *
from model.NTIRE2020_Deblur_top.uniA import AtrousNet
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import utils as vutils
from config.Config import Config

from model.NTIRE2020_Deblur_top.uniA.model_stage1 import AtrousNet
from model.NTIRE2021_Deblur.CARN.CARN import Net as CARNNet
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1 import AtrousNet as AtrousNetElu
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_Upsample_Deep import AtrousNet_billinear_Wide as AtrousNetEluUpWide
# dilation
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_Upsample_Deep_dilated import AtrousNet_billinear_Wide_dilated as AtrousNetEluUpWideDlidation
from model.NTIRE2021_Deblur.uniA_ELU.model_dilation_with_srcnn import AtrousNet_billinear_Wide_dilated_srcnn, AtrousNet_billinear_Wide_dilated_srcnn_output
# wavelet with crop
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_deblur_remix import AtrousNet_wavlet_remix
from model.NTIRE2021_Deblur.uniA_ELU.wavelet_SRCNN_remix import SRCNN
# wavelet after the upsample layer
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_dual_branch_tail import AtrousNet_SRCNN_tail
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_dual_branch_tail_no_upsample_elu import AtrousNet_SRCNN_tail_no_upsample_elu

from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_EfficientAttention import AtrousNet_billinear_EfficientAttention
from model.NTIRE2021_Deblur.uniA_ELU.model_stage1_ContextBlock import AtrousNet_billinear_ContextBlock

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
    def __init__(self, cfg, mode="jpg2png", dataset="realblur"):
        super(single_image_loader, self).__init__()
        self.cfg = cfg
        self.range = self.cfg.INPUT.RANGE
        self.mode = mode
        self.dataset = dataset

        self.mean = self.cfg.INPUT.MEAN
        self.std = self.cfg.INPUT.STD
        self.norm = self.cfg.INPUT.NORM
        self.base_transforms = self.infer_preprocess()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_file = self.cfg.DATA.TRAIN.LR_PATH
        self.file_list = self._get_image_list()

    def _get_image_list(self):
        image_path_list = [json.loads(x.strip()) for x in open(self.image_file).readlines()]
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

    def _padding_image(self, image, target_size=(720, 800)):
        """padding the image to target_size"""
        factor = 8
        h,w = image.shape[1], image.shape[2]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h% factor!=0 else 0
        padw = W-w if w%  factor!=0 else 0
        print(padh, padw)
        image = F.pad(image, (0,padw,0,padh), 'reflect')
        return image, (h, w)

    def _padding_image2(self, image, target_size=(800, 800)):
        """padding the image to target_size"""
        h,w = image.shape[1], image.shape[2]
        padh = target_size[1] - h
        padw = target_size[0] - w

        image = image.unsqueeze(0)
        images = F.pad(image, (0,padh,0,padw), padding_mode='reflect')
        images = images.squeeze(0)

        print(images.shape)
        return images, h, w


    def _load_image(self, img_path, num_retry=20):
        # for _ in range(num_retry):
        #     try:
        if img_path[:4] == 'http':
            img = Image.open(BytesIO(urllib.request.urlopen(img_path).read())).convert('RGB')
            # img = np.asarray(img)
        else:
            img = cv2.imread(img_path, -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.cfg.TRAIN.TTA == "src":
                img = Image.fromarray(img)
            elif self.cfg.TRAIN.TTA == "rot_90":
                img_rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img = Image.fromarray(img_rot_90)
            elif self.cfg.TRAIN.TTA == "rot_180":
                img_rot_180 = cv2.rotate(img, cv2.ROTATE_180)
                img = Image.fromarray(img_rot_180)
            elif self.cfg.TRAIN.TTA == "rot_270":
                img_rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img = Image.fromarray(img_rot_270)
            elif self.cfg.TRAIN.TTA == "flip_h":
                img_flip_h = cv2.flip(img, 1)
                img = Image.fromarray(img_flip_h)
            elif self.cfg.TRAIN.TTA == "flip_v":
                img_flip_v = cv2.flip(img, 0)
                img = Image.fromarray(img_flip_v)
            elif self.cfg.TRAIN.TTA == "bgr":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = Image.fromarray(img_bgr)

        #         break
        #     except Exception as e:
        #         time.sleep(5)
        #         print(f'Open image {img_path} failed, try again... resean is {e}')
        # else:
        #     raise Exception(f'Open image: {img_path} failed!')
        return img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        lr_img_path = self.file_list[index]["image_path"]
        lr_image_key = self.file_list[index]["image_key"]

        lr_img = self._load_image(lr_img_path)

        if self.base_transforms is not None:
            lr_img, lr_img = self.base_transforms(lr_img, lr_img)

        # if self.dataset == "realblur":
        #     lr_img, h, w = self._padding_image2(lr_img)
        #     # print(lr_img.shape)
        #     return lr_image_key, lr_img, h, w
        # else:
        return lr_image_key, lr_img


def load_ckpt(model, weights):
    state_dict = torch.load(weights, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    return model


def save_images(cfg, output_images, output_images_keys):
    output_images = output_images.cpu().permute(0, 2, 3, 1).clamp_(0, 255).numpy()
    for i in range(output_images.shape[0]):
        output_img = output_images[i,:,:,:]
        output_key = output_images_keys[i]
        # if "val" not in output_key:
            # output_key = 'val_' + output_key.split('/')[0] + '_' + output_key.split('/')[1]

        if cfg.INPUT.NORM:
            denormalize = DeNormalize(cfg.INPUT.MEAN, cfg.INPUT.STD)
            output_img = denormalize(output_img)

        if cfg.INPUT.RANGE == 255:
            output_img = output_img.round().astype(np.uint8)

        if cfg.TRAIN.TTA == "src":
            # start_time = time.time()
            # imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), output_img)
            # print("image_io wastet time: ", time.time() - start_time)
            # start_time = time.time()
            image = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
            # imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), output_img)
            # print("cv2 waste time: ", time.time() - start_time)
        elif cfg.TRAIN.TTA == "rot_90":
            image = cv2.rotate(output_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image)
        elif cfg.TRAIN.TTA == "rot_180":
            image = cv2.rotate(output_img, cv2.ROTATE_180)
            imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image)
        elif cfg.TRAIN.TTA == "rot_270":
            image = cv2.rotate(output_img, cv2.ROTATE_90_CLOCKWISE)
            imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image)
        elif cfg.TRAIN.TTA == "flip_h":
            image = cv2.flip(output_img, 1)
            imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image)
        elif cfg.TRAIN.TTA == "flip_v":
            image = cv2.flip(output_img, 0)
            imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image)
        elif cfg.TRAIN.TTA == "bgr":
            image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            imageio.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image)


def save_images2(cfg, output_images, output_images_keys):
    output_images1 = output_images[0].cpu().permute(0, 2, 3, 1).clamp_(0, 255).numpy()
    output_images2 = output_images[1].cpu().permute(0, 2, 3, 1).clamp_(0, 255).numpy()
    for i in range(output_images1.shape[0]):
        output_img1 = output_images1[i,:,:,:]
        output_img2 = output_images2[i,:,:,:]
        output_key = output_images_keys[i]

        if cfg.INPUT.NORM:
            denormalize = DeNormalize(cfg.INPUT.MEAN, cfg.INPUT.STD)
            output_img = denormalize(output_img)

        # if cfg.INPUT.RANGE == 255:
        #     output_img1 = output_img1.round().astype(np.uint8)
        #     output_img2 = output_img2.round().astype(np.uint8)

        if cfg.TRAIN.TTA == "src":
            # image = cv2.cvtColor(output_img1, cv2.COLOR_RGB2BGR)
            # image1 = cv2.cvtColor(output_img2, cv2.COLOR_RGB2BGR)
            image = (output_img1 + output_img2) / 2
            image = image.round().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cfg.SAVE.SR_IMAGES, output_key.replace('.jpg', '.png')), image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])


def main_worker(gpu, ngpus_per_node, args):
    cfg = Config(args.config_file)()
    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print('rank: {} / {}'.format(args.rank, args.world_size))
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)

    # model
    if cfg.MODEL.BACKBONE.NAME == "CARN":
        model = CARNNet(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet":
        model = AtrousNet(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetElu":
        model = AtrousNetElu(3, 3)
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetEluUpWide":
        model = AtrousNetEluUpWide(3, 3,
                    num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
                    d_mult=cfg.MODEL.BACKBONE.WIDTH
                    )
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetEluUpWideDlidation":
        model = AtrousNetEluUpWideDlidation(3, 3,
                    num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
                    d_mult=cfg.MODEL.BACKBONE.WIDTH
                    )
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNetEluUpWideWaveletRemix":
        model = AtrousNet_wavlet_remix(3, 3)

    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet_billinear_Wide_dilated_srcnn":
        model = AtrousNet_billinear_Wide_dilated_srcnn(3, 3,
                num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
                d_mult=cfg.MODEL.BACKBONE.WIDTH
            )
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet_SRCNN_tail":
        model = AtrousNet_SRCNN_tail(
            3,
            3,
            num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
            d_mult=cfg.MODEL.BACKBONE.WIDTH,
            efficientattention=cfg.MODEL.BACKBONE.EFFICIENT_ATTENTION,
            gcattention=cfg.MODEL.BACKBONE.GC_ATTENTION
        )

    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet_SRCNN_tail_no_upsample_elu":
        model = AtrousNet_SRCNN_tail_no_upsample_elu(
            3,
            3,
            num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
            d_mult=cfg.MODEL.BACKBONE.WIDTH
        )
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet_dilation_effiattention":
        model = AtrousNet_billinear_EfficientAttention(
            3,
            3,
            num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
            d_mult=cfg.MODEL.BACKBONE.WIDTH
        )

    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet_dilation_gc":
        model = AtrousNet_billinear_ContextBlock(
            3,
            3,
            num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
            d_mult=cfg.MODEL.BACKBONE.WIDTH
        )
    elif cfg.MODEL.BACKBONE.NAME == "AtrousNet_billinear_Wide_dilated_srcnn_output":
        model = AtrousNet_billinear_Wide_dilated_srcnn_output(
            3,
            3,
            num_blocks=cfg.MODEL.BACKBONE.NUM_BLOCKS,
            d_mult=cfg.MODEL.BACKBONE.WIDTH,
            srcnn_add=cfg.MODEL.SRCNN.ADD,
            srcnn_smooth=cfg.MODEL.SRCNN.SMOOTH
        )

    model = load_ckpt(model, cfg.MODEL.WEIGHTS)
    if args.rank == 0:
        print("================{}=============".format(cfg.MODEL.BACKBONE.NAME))
        print(model)
    if torch.cuda.is_available():
        model.cuda(args.gpu)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = model

    model.eval()
    device = model.device

    dataset = single_image_loader(cfg)
    if args.rank == 0:
        print("Validation dataset lengh: ", len(dataset))

    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    # dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=sampler,
        drop_last=False
    )
    total_length = len(dataloader) // cfg.TRAIN.BATCH_SIZE

    for iter_id, batch_data in enumerate(dataloader):
        if len(batch_data) > 2:
            lr_key, lr_images, img_h, img_w = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        else:
            lr_key, lr_images = batch_data[0], batch_data[1]
        lr_images = lr_images.to(device)

        # realblur
        if True:
            factor = 8
            h,w = lr_images.shape[2], lr_images.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            lr_images = torch.nn.functional.pad(lr_images, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            start_time = time.time()
            if cfg.MODEL.BACKBONE.NAME == "AtrousNet_billinear_Wide_dilated_srcnn_output":
                sr_images, sr_images2 = model(lr_images)
            else:
                sr_images = model(lr_images)
                # used for realblur
                if h is not None:
                    sr_images = sr_images[:,:,:h,:w]

            batch_time = time.time() - start_time

        save_images(cfg, sr_images, lr_key)
        # save_images2(cfg, [sr_images, sr_images2],lr_key)
        if args.rank == 0:
            print(f"Process the [{iter_id}/{total_length}], batch time waste {batch_time} !!!")


if __name__ == "__main__":

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

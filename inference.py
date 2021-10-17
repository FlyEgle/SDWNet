# -*- coding: utf-8 -*-
"""
@author    : GiantPandaSR
@datetime      : 2021-02-09
@describe  : Training with DDP or DataParallel 
"""
from __future__ import print_function
from config.Config import Config

# system
import warnings

warnings.filterwarnings("ignore")

import os 
import cv2 
import time
import imageio
from tqdm import tqdm
# torch 
from torch.utils.data import DataLoader
from data.augments import *
# model 
from model.NTIRE2020_Deblur_top.uniA import AtrousNet
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import utils as vutils
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
    def __init__(self, cfg, rec_path, mode="jpg2png"):
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
        self.file_list = self._get_image_list()

    def find_all_with_ext(self, path, ext):
        if not isinstance(ext, list):
            filter = [ext]
        else:
            filter = ext

        result = []

        for maindir, subdir, file_name_list in os.walk(path):
            #         print("1:",maindir) #当前主目录
            #         print("2:",subdir) #当前主目录下的所有目录
            #         print("3:",file_name_list)  #当前主目录下的所有文件

            for filename in file_name_list:
                apath = os.path.join(maindir, filename).replace('\\', '/')
                ext = os.path.splitext(apath)[1]

                if ext in filter:
                    result.append(apath)

        return result

    def _get_image_list(self):
        if self.mode == 'jpg2png':
            return self.find_all_with_ext(self.lr_path, ext='.jpg')
        elif self.mode == 'png2png':
            return self.find_all_with_ext(self.lr_path, ext='.jpg')

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
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                break
            except Exception as e:
                time.sleep(5)
                print(f'Open image {img_path} failed, try again... resean is {e}')
        else:
            raise Exception(f'Open image: {img_path} failed!')

        return img

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        lr_img_path = self.file_list[index]
        lr_img = self._load_image(lr_img_path)

        if self.base_transforms is not None:
            lr_img, lr_img = self.base_transforms(lr_img, lr_img)
        return lr_img_path, lr_img


def model_initializer(opt):
    # Non-distributed GPU Parallel
    device = opt['device']
    model_arch = "{}-{}".format("SR","AtrousNet")
    model = AtrousNet(in_channels = 3, out_channels = 3)
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
        batch_size=1,
        num_workers=8)

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # Now only support single image inference
        file_name = os.path.split(data[0][0])[1]
        img_data = data[1].to(opt['device'])

        with torch.no_grad():
            output = model(img_data)
            output_img = output[0, :, :, :].cpu()
            # output_img = output[0,:,:,:].float().cpu().numpy()

        if cfg.INPUT.NORM:
            denormalize = DeNormalize(cfg.INPUT.MEAN, cfg.INPUT.STD)
            output_img = denormalize(output_img)

        if cfg.INPUT.RANGE == 255:
            output_img.clamp_(0,255)
            output_img = output_img.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
            imageio.imwrite(os.path.join(opt['output_path'], file_name.replace('.jpg', '.png')), output_img)
        else:
            output_img.clamp_(0,1)
            output_img = (output_img.permute(1, 2, 0).cpu().numpy()*255.0).round().astype(np.uint8)
            imageio.imwrite(os.path.join(opt['output_path'], file_name.replace('.jpg', '.png')), output_img)


if __name__ == '__main__':
    opt = dict()
    opt['device'] = "cuda"
    opt['model_pth'] = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/weights/SR-AtrousNet_48x48.pth'
    opt['working_path'] = '/media/cydiachen/dataset/NTIRE2021/val/sub_images/1_to_4/val_NTIRE_jpeg_input_1_0'
    opt['output_path'] = os.path.join("./output/1_0",os.path.splitext(os.path.split(opt['model_pth'])[1])[0])
    opt['config_file'] = "/home/cydiachen/Desktop/SR/SR_NTIRE2021/config/resolution/unia_255_no_norm_48x48.yaml"
    cfg = Config(opt['config_file'])()
    if not os.path.exists(opt['output_path']):
        os.makedirs(opt['output_path'])
    inference(cfg, opt)
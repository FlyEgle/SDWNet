import os

import imageio
import numpy as np
import skimage.color as sc
import torch
from tqdm import tqdm
from model.NTIRE2020_Deblur_top.uniA.model_stage1 import AtrousNet


def find_all_with_ext(path,ext):
    if not isinstance(ext,list):
        filter = [ext]
    else:
        filter = ext

    result = []

    for maindir, subdir, file_name_list in os.walk(path):

        for filename in file_name_list:
            apath = os.path.join(maindir, filename).replace('\\','/')
            ext = os.path.splitext(apath)[1]

            if ext in filter:
                result.append(apath)

    return result

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0,255).round().div(pixel_range)

def set_channel(img, n_channels=3):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    c = img.shape[2]
    if n_channels == 1 and c == 3:
        img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
    elif n_channels == 3 and c == 1:
        img = np.concatenate([img] * n_channels, 2)
    return img

def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)
    return tensor

#TODO: Inference dataloader for single image


def image_read(filename, rgb_range):
    lr_0 = imageio.imread(filename)
    # lr_1 = cv2.imread(filename)
    # lr_1 = cv2.cvtColor(lr_1, cv2.COLOR_BGR2RGB)
    lr_0 = np.asarray(lr_0)
    lr_new = set_channel(lr_0, n_channels = 3)
    lr_t = np2Tensor(lr_new, rgb_range)
    return lr_t

def model_init(network, weights):
    model_weights = torch.load(weights)
    network = network
    network.load_state_dict(model_weights['state_dict'], strict=True)
    network.eval()
    return network

def inference(model_path, input_path, output_path, rgb_range=255):
    # # model_weights = torch.load(model_path)
    # network = AtrousNet(in_channels=3, out_channels=3)
    # network.load_state_dict(model_weights,strict = True)
    # network.eval()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    network = model_init(network=AtrousNet(in_channels=3, out_channels=3),
                         weights = model_path)

    filelist = find_all_with_ext(input_path, ext='.jpg')
    for idx, content in enumerate(tqdm(filelist)):
        image = image_read(content,rgb_range=255)
        image = image.to(device)
        image = image.unsqueeze(0)
        network = network.to("cuda")

        with torch.no_grad():
            output = network(image)
        output = quantize(output, rgb_range)
        for idx in range(output.shape[0]):
            normalized = output.mul(255 / rgb_range)
            tensor_cpu = normalized.byte().permute(0, 2, 3, 1).cpu()
            output_img = tensor_cpu[0].numpy()
            imageio.imwrite(os.path.join(output_path, os.path.split(content)[1].replace('.jpg','.png')), output_img)

def calc_psnr(sr, hr, rgb_range):
    import math
    if hr.nelement() == 1:return 0
    diff = (sr - hr) / rgb_range
    shave = 1
    if diff.size(1) > 1:
        diff = diff.mul(diff).sum(dim=1)
    valid = diff[...,shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)


if __name__ == "__main__":
    model_path = "/home/cydiachen/Desktop/SR_NTIRE2021/weights/SR-AtrousNet_epoch_115.pth"
    device = 'cuda'
    inference(model_path,"/media/cydiachen/Cydia-256ssd/val_blur_jpeg/val/val_blur_jpeg_allinone","./test/")
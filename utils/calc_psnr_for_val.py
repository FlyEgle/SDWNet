"""
-*- coding: utf-8 -*-
@author: GiantPandaSR
@datetime: 2021-03-03
@describe: Calculate the psnr for validaiton
"""
import imageio
import numpy as np 

import torch 
import torch.distributed as dist

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor2array(tensor):
    tensor.clamp_(0,255)
    image = tensor.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
    return image 


def get_psnr(sr_tensor, hr_tensor):
    sum_psnr = 0.0
    for i in range(sr_tensor.shape[0]):
        output_tensor = sr_tensor[i, :, :, :]
        sr_image = tensor2array(output_tensor)
        hr_image = tensor2array(hr_tensor)
        psnr = peak_signal_noise_ratio(sr_image, hr_image)
        sum_psnr += psnr 
    return sum_psnr






"""Filter the crop sample with the psnr rank
"""
import os
import cv2 
import imageio
import numpy as np 
from PIL import Image 

from multiprocessing import Pool
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def clc_psnr(data):
    lr_image, hr_image = data[0], data[1]
    
    lr_img = imageio.imread(lr_image)
    hr_img = imageio.imread(hr_image)

    psnr = peak_signal_noise_ratio(lr_img, hr_img)

    return psnr 


def make_dict(data_folder):
    image_list = [os.path.join(data_folder, data) for data in os.listdir(data_folder)]
    image_dict = {}
    for image in image_list:
        image_name = image.split('/')[-1].split('.')[0]
        image_dict[image_name] = image
    return image_dict 


def make_pair_list(lr_folder, hr_folder):
    lr_dict = make_dict(lr_folder)
    hr_dict = make_dict(hr_folder)
    
    data_paris = []
    for key, value in lr_dict.items():
        if key in hr_dict.keys():
            data_paris.append([value, hr_dcit[key]])
    
    return data_paris


if __name__ == "__main__":
    lr_train_path = "/data/local/SR_CROP/train/train_blur_jpeg"
    hr_train_path = "/data/local/SR_CROP/train/train_sharp"

    pairs_list = make_pair_list(lr_train_path, hr_train_path)

    pool = Pool(64)
    result = pool.map(clc_psnr, pairs_list[10])
    pool.close()
    pool.join()
    print(result)
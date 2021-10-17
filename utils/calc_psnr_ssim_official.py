# utils for goprol
import imageio
import os
import cv2
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from numpy import *
from multiprocessing import Pool


def main(gt_path, reconstructed_path):
    # ref_img = imageio.imread(gt_path)
    # res_img = imageio.imread(reconstructed_path)
    ref_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2YCR_CB)
    res_img = cv2.cvtColor(cv2.imread(reconstructed_path), cv2.COLOR_BGR2YCR_CB)
    psnr = peak_signal_noise_ratio(ref_img[:,:,0], res_img[:,:,0])
    ssim = structural_similarity(ref_img, res_img, multichannel =True, gaussian_weights = True, use_sample_covariance = True)

    return psnr, ssim


def main_pool(image_name):
    # sr_images = imageio.imread(os.path.join(sr_folder, image_name))
    sr_images = cv2.imread(os.path.join(sr_folder, image_name))
    sr_images = cv2.cvtColor(sr_images, cv2.COLOR_BGR2YCR_CB)

    hr_path = hr_dict[image_name]
    # hr_images = imageio.imread(hr_path)
    hr_images = cv2.imread(hr_path)
    hr_images = cv2.cvtColor(hr_images, cv2.COLOR_BGR2YCR_CB)

    psnr = peak_signal_noise_ratio(sr_images[:,:,0], hr_images[:,:,0])
    ssim = structural_similarity(sr_images, hr_images, multichannel =True, gaussian_weights = True, use_sample_covariance = True)

    return psnr, ssim


def main_sr_hr_pool(image_pairs):
    sr_images_path = image_pairs[0]
    hr_images_path = image_pairs[1]

    sr_images = cv2.cvtColor(cv2.imread(sr_images_path), cv2.COLOR_BGR2YCR_CB)
    hr_images = cv2.cvtColor(cv2.imread(hr_images_path), cv2.COLOR_BGR2YCR_CB)

    psnr = peak_signal_noise_ratio(sr_images[:,:,0], hr_images[:,:,0])
    ssim = structural_similarity(sr_images, hr_images, multichannel =True, gaussian_weights = True, use_sample_covariance = True)
    return psnr, ssim


def make_pairs(sr_folder, hr_file):
    sr_list = [os.path.join(sr_folder, x) for x in os.listdir(sr_folder)]
    hr_list = {json.loads(x.strip())["image_key"]:json.loads(x.strip())["image_path"] for x in open(hr_file).readlines()}
    pairs_list = []
    for data in sr_list:
        image_name = data.split('/')[-1]
        if image_name in hr_list:
            pairs_list.append([data, hr_list[image_name]])
    return pairs_list


if __name__ == "__main__":
    # sr_folder = '/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/wavelet_wide_32_block_16_416_no_crop'
    # sr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_for_paper/Deblurgan-2-test-result"
    # hr_file = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/file/blur/test_goprol_sharp.log"
    # sr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/realblur_j"
    # sr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/others_1"
    # sr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/wavelet_wide_32_block_16_416_no_crop"
    sr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/no_crop_result"
    # hr_file = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/file/blur/test_goprol_sharp.log"
    hr_file = "/data/jiangmingchao/data/dataset/file/hide2/sharp.log"
    pairs_list = make_pairs(sr_folder, hr_file)
    print(len(pairs_list))

    # sr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/hide_1"
    # hr_file = "/data/jiangmingchao/data/dataset/file/hide1/sharp.log"
    # hr_list = [json.loads(x.strip()) for x in open(hr_file).readlines()]
    # print(len(hr_list))
    # hr_dict = {x["image_key"]:x["image_path"] for x in hr_list}
    # # print(len(hr_dict))
    # psnr_list = []
    # ssim_list = []
    # data_name_list = os.listdir(sr_folder)
    # for data in tqdm(data_name_list):
    #     if data in hr_dict:
    #         psnr, ssim = main(hr_dict[data], os.path.join(sr_folder, data))
    #         psnr_list.append(psnr)
    #         ssim_list.append(ssim)
    pool = Pool(128)
    result = pool.map(main_sr_hr_pool, pairs_list)
    pool.close()
    pool.join()
    total_psnr = [x[0] for x in result]
    total_ssim = [x[1] for x in result]

    print(mean(total_psnr))
    print(mean(total_ssim))

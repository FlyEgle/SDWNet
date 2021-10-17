import cv2
import os
import imageio
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from numpy import *
from multiprocessing import Pool


def main(gt_path, reconstructed_path):
    ref_img = imageio.imread(gt_path)
    res_img = imageio.imread(reconstructed_path)

    psnr = peak_signal_noise_ratio(ref_img, res_img)
    ssim = structural_similarity(ref_img, res_img, multichannel =True, gaussian_weights = True, use_sample_covariance = True)

    return psnr, ssim


def main_pool(image_name):

    sr_images = imageio.imread(os.path.join(args.sr_folders, image_name))
    hr_images = imageio.imread(os.path.join(args.hr_folders, image_name))

    psnr = peak_signal_noise_ratio(sr_images, hr_images)
    ssim = structural_similarity(sr_images, hr_images, multichannel =True, gaussian_weights = True, use_sample_covariance = True)

    return psnr, ssim


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SR DDP Inference')
    parser.add_argument('--sr_folders', type=str,
                    default="/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data_1/test_3000_results")
    parser.add_argument('--hr_folders', type=str,
                    default="/data/jiangmingchao/data/dataset/SR_localdata/test_300")

    args = parser.parse_args()

    data_name_list = os.listdir(args.sr_folders)
    print(len(data_name_list))

    pool = Pool(64)
    result = pool.map(main_pool, data_name_list)
    pool.close()
    pool.join()
    total_psnr = [x[0] for x in result]
    total_ssim = [x[1] for x in result]
    print(mean(total_psnr))
    print(mean(total_ssim))

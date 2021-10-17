## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures

def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 1e-8

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  try:
    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=3)
  # print(warp_matrix)
  except Exception as e:
    warp_matrix = warp_matrix

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = cv2.cvtColor(cv2.imread(tar), cv2.COLOR_BGR2RGB)
    prd_img = cv2.cvtColor(cv2.imread(prd), cv2.COLOR_BGR2RGB)
    # print("tar",tar_img.shape)
    # print("prd",prd_img.shape)
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0

    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    return (PSNR,SSIM)

datasets = ['RealBlur_R']

for dataset in datasets:

    # file_path = os.path.join('results' , dataset)
    # file_path = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/no_crop_result1"
    file_path = "/data/jiangmingchao/data/dataset/SR_localdata/goprol_baseline/realblur_R_pretrain_result"
    # gt_path = os.path.join('Datasets', dataset, 'test', 'target')
    gt_path = "/data/jiangmingchao/data/dataset/realblur-r/test/target"

    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))

    # print(len(path_list))

    assert len(path_list) != 0, "Predicted files not found"
    assert len(gt_list) != 0, "Target files not found"

    psnr, ssim = [], []
    img_files =[(i, j) for i,j in zip(gt_list,path_list)]
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for filename, PSNR_SSIM in tqdm(zip(img_files, executor.map(proc, img_files))):
                psnr.append(PSNR_SSIM[0])
                ssim.append(PSNR_SSIM[1])
    except Exception as e:
        print(f"opencv reason {e}!!! {filename}")

    avg_psnr = sum(psnr)/len(psnr)
    avg_ssim = sum(ssim)/len(ssim)

    print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
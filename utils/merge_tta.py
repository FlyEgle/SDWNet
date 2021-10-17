import os 
import json

import cv2 
import numpy as np
import imageio 

from multiprocessing import Pool


src_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/src"
rot_90_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/rot_90"
rot_180_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/rot_180"
rot_270_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/rot_270"
rot_flip_h_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/flip_h"
rot_flip_v_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/flip_v"
rot_bgr_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/bgr"

src_list = os.listdir(src_folder)

save_folder = "/data/jiangmingchao/data/dataset/SR_localdata/test_3000_tta_data/test_3000_results"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

def merge_tta(data_name):
    
    src_img = cv2.imread(os.path.join(src_folder, data_name))
    rot_90_img = cv2.imread(os.path.join(rot_90_folder, data_name))
    rot_180_img = cv2.imread(os.path.join(rot_180_folder, data_name))
    rot_270_img = cv2.imread(os.path.join(rot_270_folder, data_name))
    flip_h_img = cv2.imread(os.path.join(rot_flip_h_folder, data_name))
    flip_v_img = cv2.imread(os.path.join(rot_flip_v_folder, data_name))
    bgr_img = cv2.imread(os.path.join(rot_bgr_folder, data_name))

    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    rot_90_img = cv2.cvtColor(rot_90_img, cv2.COLOR_BGR2RGB)
    rot_180_img = cv2.cvtColor(rot_180_img, cv2.COLOR_BGR2RGB)
    rot_270_img = cv2.cvtColor(rot_270_img, cv2.COLOR_BGR2RGB)
    flip_h_img = cv2.cvtColor(flip_h_img, cv2.COLOR_BGR2RGB)
    flip_v_img = cv2.cvtColor(flip_v_img, cv2.COLOR_BGR2RGB)
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    imgs = [src_img, rot_90_img, rot_180_img, rot_270_img, flip_h_img, flip_v_img, bgr_img]
    image = np.mean(np.array(imgs), axis=0).round().astype(np.uint8)

    imageio.imwrite(os.path.join(save_folder, data_name), image)


if __name__ == "__main__":
    pool = Pool(64)
    pool.map(merge_tta, src_list)
    pool.close()
    pool.join()
    print("Finish!!!!")

    

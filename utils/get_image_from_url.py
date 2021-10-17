import urllib.request as urt
import os
import json
from io import BytesIO
import imageio
from PIL import Image
import cv2
import numpy as np 
from tqdm import tqdm
from multiprocessing import Pool

save_folder = "/data/jiangmingchao/data/dataset/best_atrousnet_wide_up_hr_300_images"

def read_image(data):
    image_url = data["image_path"]
    image_key = data["image_key"]
    new_key = 'val_' + image_key.split('/')[0] + '_' + image_key.split('/')[1]
    resp = urt.urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_path = os.path.join(save_folder, new_key)
    # image.save(image_path, "PNG", quality=100)
    imageio.imwrite(image_path, image)

if __name__ == "__main__":
    data_file = "/data/jiangmingchao/data/SR_NTIRE2021/data/val/test_gt_from_val.log"
    data_list = [json.loads(x) for x in open(data_file).readlines()]
    # for data in tqdm(data_list):
    #     read_image(json.loads(data))
    pool = Pool(32)
    pool.map(read_image, data_list)
    pool.close()
    pool.join()
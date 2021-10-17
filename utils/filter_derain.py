import os
import cv2
import numpy as np
from multiprocessing import Pool


def read_image(image_path):
    image = cv2.imread(image_path)
    shape = image.shape
    return shape


if __name__ == "__main__":
    image_folder = "/data/jiangmingchao/data/dataset/DeRain/train/input"
    image_list = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]
    pool = Pool(64)
    result = pool.map(read_image, image_list)
    pool.close()
    pool.join()

    h, w = [],[]
    for r in result:
        h.append(r[0])
        w.append(r[1])

    print(max(h), max(w))
    print(min(h), min(w))
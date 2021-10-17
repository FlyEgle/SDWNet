import os
import cv2
import time
import numpy as np

from PIL import Image
from multiprocessing import Pool


def get_cropimage(patch_size, image, scale):
    crop_image_list = []
    height, width = image.shape[0], image.shape[1]
    row_step = patch_size[0] // scale
    col_step = patch_size[1] // scale

    row_interval = int((height - patch_size[0]) / row_step )
    col_interval = int((width - patch_size[1]) / col_step )

    raw_image = image.copy()

    idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(row_interval + 1):
        for j in range(col_interval + 1):
            crop_image = image[
                                i * row_step : i * row_step + patch_size[0],
                                j * col_step : j * col_step + patch_size[1],
                                :
                              ]
            crop_image_list.append(crop_image)
            # cv2.rectangle(raw_image,
            #                 (j * col_step,i * row_step),
            #                 (j * col_step + patch_size[1], i * row_step + patch_size[0]),
            #                 (0,255,255),
            #                 1
            #              )
            # cv2.putText(raw_image, f'{idx}', (j * col_step+10, i * row_step+10), font, 1.0, (255, 255, 255), 2)
            # cv2.imwrite(f"/data/remote/dataset/exp/img_{idx}.png", crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # idx += 1

    for i in range(row_interval + 1):
        crop_image = image[
            i * row_step : i * row_step + patch_size[0],
            width - patch_size[0] : width,
            :
        ]
        # cv2.imwrite(f"/data/remote/dataset/exp/img_{idx}.png", crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # idx += 1
        crop_image_list.append(crop_image)

    # cv2.imwrite("/data/remote/dataset/exp/img_show.png", raw_image)
    return crop_image_list


def get_image_list_lr(data_folder):
    image_list = []
    for image_folder in os.listdir(data_folder):
        image_folder_path = os.path.join(data_folder, image_folder)
        sub_folder_path = os.path.join(image_folder_path, "blur")
        for image in os.listdir(sub_folder_path):
            image_path = os.path.join(sub_folder_path, image)
            image_list.append(image_path)
    return image_list

def get_image_list_hr(data_folder):
    image_list = []
    for image_folder in os.listdir(data_folder):
        image_folder_path = os.path.join(data_folder, image_folder)
        sub_folder_path = os.path.join(image_folder_path, "sharp")
        for image in os.listdir(sub_folder_path):
            image_path = os.path.join(sub_folder_path, image)
            image_list.append(image_path)
    return image_list


def make_pair(lr_list, hr_list):
    lr_dict = {lr.split('.')[0].split('/')[-3] + '_' + lr.split('.')[0].split('/')[-1]: lr for lr in lr_list}
    hr_dict = {hr.split('.')[0].split('/')[-3] + '_' + hr.split('.')[0].split('/')[-1]: hr for hr in hr_list}

    pairs_list = []
    for key, value in lr_dict.items():
        if key in hr_dict.keys():
            pairs_list.append([value, hr_dict[key]])

    return pairs_list


def save_crop(image_path, crop_image_list, save_folder):
    for idx, crop_image in enumerate(crop_image_list):
        image_name = image_path.split('/')[-3] + '_' + image_path.split('/')[-1].split('.')[0] + '_' + str(idx+1) + '.png'
        save_path = os.path.join(save_folder, image_name)
        cv2.imwrite(save_path, crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def process(image_list):
    for lr, hr in image_list:
        lr_image = cv2.imread(lr)
        hr_image = cv2.imread(hr)
        lr_crop_image = get_cropimage((480, 480), lr_image, 4)
        hr_crop_image = get_cropimage((480, 480), hr_image, 4)
        save_crop(lr, lr_crop_image, lr_crop_output_folder)
        save_crop(hr, hr_crop_image, hr_crop_output_folder)


def process_mp(data):
    lr, hr = data[0], data[1]
    lr_image = cv2.imread(lr)
    hr_image = cv2.imread(hr)
    lr_crop_image = get_cropimage((480, 480), lr_image, 4)
    hr_crop_image = get_cropimage((480, 480), hr_image, 4)
    save_crop(lr, lr_crop_image, lr_crop_output_folder)
    save_crop(hr, hr_crop_image, hr_crop_output_folder)


if __name__ == "__main__":

    # lr_folder = "/data/remote/dataset/super2021/val/val_blur_jpeg"
    # hr_folder = "/data/remote/dataset/super2021/val/val_sharp"
    lr_folder = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/data/train/"
    hr_folder = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/data/train/"

    lr_train_list = get_image_list_lr(lr_folder)
    hr_train_list = get_image_list_hr(hr_folder)

    print(len(lr_train_list))
    print(len(hr_train_list))

    pairs_list = make_pair(lr_train_list, hr_train_list)
    # print(pairs_list)
    lr_crop_output_folder = "/data/jiangmingchao/data/dataset/SR_dataset/GoProL_crop_all/train/blur/"
    hr_crop_output_folder = "/data/jiangmingchao/data/dataset/SR_dataset/GoProL_crop_all/train/sharp/"

    print("Begin!!!!")
    start_time = time.time()
    pool = Pool(64)
    pool.map(process_mp, pairs_list)
    pool.close()
    pool.join()
    print("Finish!!!!")
    print(time.time() - start_time)

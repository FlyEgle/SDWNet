import os
import cv2
import json
import random

data_folder = "/data/jiangmingchao/data/dataset/best_atrousnet_wide_up_hr_300_images"
data_images_list = [os.path.join(data_folder, x) for x in os.listdir(data_folder)]
print(len(data_images_list))
random.shuffle(data_images_list)
image = cv2.imread(data_images_list[0])

with open("/data/jiangmingchao/data/dataset/hr_images_300_val.log", "w") as file:
    for data in data_images_list:
        image_key = data.split('/')[-1]
        result = {
            "image_key": image_key,
            "image_path": data
        }
        file.write(json.dumps(result, ensure_ascii=False) + '\n')

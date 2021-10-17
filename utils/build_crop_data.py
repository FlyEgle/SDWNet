import os
import json


data_folder = "/data/jiangmingchao/data/dataset/SR_dataset/GoProL_crop/train"
data_blur = os.path.join(data_folder, "blur")
data_sharp = os.path.join(data_folder, "sharp")

blur_image = [os.path.join(data_blur, x) for x in os.listdir(data_blur)]
sharp_image = [os.path.join(data_sharp, x) for x in os.listdir(data_sharp)]

def make_dict(image_list, output_file):
    # data_list = []
    with open(output_file, "w") as file:
        for image in image_list:
            image_key = image.split('/')[-1].split('_')[1] + '_' + image.split('/')[-1].split('_')[2]
            result = {
                "image_key": image_key,
                "image_path": image
            }
            file.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    blur_goprol_crop = "/data/jiangmingchao/data/dataset/SR_dataset/GoProL_crop/goprol_crop_blur.log"
    sharp_goprol_crop = "/data/jiangmingchao/data/dataset/SR_dataset/GoProL_crop/goprol_crop_sharp.log"
    make_dict(blur_image, blur_goprol_crop)
    make_dict(sharp_image, sharp_goprol_crop)




import os 
import json 

data_folder = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/data"
train_folder = os.path.join(data_folder, "train")
test_folder = os.path.join(data_folder, "test")

def get_file(data_folder):
    blur_image_path = []
    sharp_image_path= []
    for sub_folder in os.listdir(data_folder):
        sub_path = os.path.join(data_folder, sub_folder)
        blur_path = os.path.join(sub_path, "blur_gamma")
        sharp_path = os.path.join(sub_path, "sharp")
        
        image_key = os.listdir(sharp_path)
        for image in image_key:
            blur_image_path.append(os.path.join(blur_path, image))
            sharp_image_path.append(os.path.join(sharp_path, image))

    return blur_image_path , sharp_image_path

def make_log(image_path, output_file):
    with open(output_file, "w") as file:
        for data in image_path:
            image_key = data.split('/')[-3] + '_' + data.split('/')[-1]
            # print(image_key)
            result = {
                "image_key" : image_key,
                "image_path": data
            }
            file.write(json.dumps(result) + '\n')


train_blur_path, train_sharp_path = get_file(train_folder)
test_blur_path, test_sharp_path = get_file(test_folder)

train_blur_file = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/file/train_goprol_blur.log"
train_sharp_file = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/file/train_goprol_sharp.log"

test_blur_file = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/file/test_goprol_blur.log"
test_sharp_file = "/data/jiangmingchao/data/dataset/SR_dataset/GoProlL/file/test_goprol_sharp.log"

make_log(train_blur_path, train_blur_file)
make_log(train_sharp_path, train_sharp_file)


make_log(test_blur_path, test_blur_file)
make_log(test_sharp_path, test_sharp_file)
        
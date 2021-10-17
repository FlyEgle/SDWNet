import os
import json  


def make_val_file(data_file, output_file):
    data_list = open(data_file).readlines()
    with open(output_file, "w") as file:
        for data in data_list:
            data_json = json.loads(data.strip())
            image_key = data_json["image_key"]
            new_image_key = os.path.join("val", image_key)
            image_path = data_json["image_path"]
            result = {
                "image_key"   : new_image_key,
                "image_path"  : image_path
            }
            file.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    val_lr_path = "/data/jiangmingchao/data/SR_NTIRE2021/data/val/test_jpeg_from_val.log"
    val_hr_path = "/data/jiangmingchao/data/SR_NTIRE2021/data/val/test_gt_from_val.log"
        
    output_val_lr_path = "/data/jiangmingchao/data/SR_NTIRE2021/data/merge_train_val_jpeg/online_val_jpeg.log"
    output_val_hr_path = "/data/jiangmingchao/data/SR_NTIRE2021/data/merge_train_val_jpeg/online_val_gt.log"

    make_val_file(val_lr_path, output_val_lr_path)
    make_val_file(val_hr_path, output_val_hr_path)
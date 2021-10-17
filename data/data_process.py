"""
-*- coding:utf-8 -*-
@author   : GiantPandaSR
@date     : 2021-02-09
@describe : Build the training file, each line is a dict with the key and image path.
"""
import os
import json
import numpy as np

def make_data(data_file, data_head, output_file):
    data_list = [x.strip() for x in open(data_file)]
    data_dict = []
    with open(output_file, "w") as file:
        for data in data_list:
            image_key = data.split('/')[-2] + '/' + data.split('/')[-1]
            data_result = {
                            "image_key": image_key,
                            "image_path": os.path.join(data_head, data)
                           }
            data_json = json.dumps(data_result)
            file.write(data_json + '\n')


if __name__ == "__main__":
    data_file = "/data/jiangmingchao/data/SR_NTIRE2021/data/test/blur_bicubic_test.log"
    data_head = "http://ai-train-datasets.oss-cn-zhangjiakou-internal.aliyuncs.com/jiangmingchao/super2021dataset/"
    output_file = "/data/jiangmingchao/data/SR_NTIRE2021/data/test/test_bicubic.log"
    make_data(data_file, data_head, output_file)



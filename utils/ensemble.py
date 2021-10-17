import cv2
import os
import numpy as np
from tqdm import tqdm

folder_1 = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/SR-AtrousNet_256x256/1_to_4/val_NTIRE_jpeg_input_assemble'
folder_2 = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/SR-AtrousNet_480x480/1_to_4/val_NTIRE_jpeg_input_assemble'
folder_3 = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/SR-AtrousNet_512x512/1_to_4/val_NTIRE_jpeg_input_assemble'

folder_4 = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/SR-AtrousNet_256x256/1_to_16/val_NTIRE_jpeg_input_assemble'
folder_5 = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/SR-AtrousNet_480x480/1_to_16/val_NTIRE_jpeg_input_assemble'
folder_6 = '/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/SR-AtrousNet_512x512/1_to_16/val_NTIRE_jpeg_input_assemble'


sub_folder_1 = "/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/full_size/SR-AtrousNet_256x256/"
sub_folder_2 = "/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/full_size/SR-AtrousNet_480x480/"
sub_folder_3 = "/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/full_size/SR-AtrousNet_512x512/"


output_folder = "/home/cydiachen/Desktop/SR/SR_NTIRE2021/output/sub_images/ensemble_1_to_4&full&1_to_16"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ensemble_strategy = ['mean', 'max']

if not os.path.exists(os.path.join(output_folder ,ensemble_strategy[0])):
    os.makedirs(os.path.join(output_folder ,ensemble_strategy[0]))

if not os.path.exists(os.path.join(output_folder ,ensemble_strategy[1])):
    os.makedirs(os.path.join(output_folder ,ensemble_strategy[1]))


def img_reader(src):
    img = cv2.imread(src)
    return img

ref_filelist = os.listdir(folder_1)

for idx, content in enumerate(tqdm(ref_filelist)):
    ref_filename = os.path.join(folder_3, content)
    model_1_filename = os.path.join(folder_1, content)
    model_2_filename = os.path.join(folder_2, content)

    model_3_filename = os.path.join(sub_folder_1, content)
    model_4_filename = os.path.join(sub_folder_2, content)
    model_5_filename = os.path.join(sub_folder_3, content)

    model_6_filename = os.path.join(folder_4, content)
    model_7_filename = os.path.join(folder_5, content)
    model_8_filename = os.path.join(folder_6, content)

    img_1 = img_reader(ref_filename)
    img_2 = img_reader(model_1_filename)
    img_3 = img_reader(model_2_filename)

    img_4 = img_reader(model_3_filename)
    img_5 = img_reader(model_4_filename)
    img_6 = img_reader(model_5_filename)

    img_7 = img_reader(model_6_filename)
    img_8 = img_reader(model_7_filename)
    img_9 = img_reader(model_8_filename)

    img_arrays = np.array([img_1, img_2, img_3, img_4, img_5, img_6])
    mean_img = np.mean(img_arrays, axis = 0).round()
    max_img = np.max(img_arrays, axis = 0).round()

    cv2.imwrite(os.path.join(os.path.join(output_folder ,ensemble_strategy[0]), content), mean_img)
    cv2.imwrite(os.path.join(os.path.join(output_folder ,ensemble_strategy[1]), content), max_img)





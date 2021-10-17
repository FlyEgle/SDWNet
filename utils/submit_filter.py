import os
import shutil

working_path = "E:/Dataset/public_dataset/NTIRE2021/Track1/test_blur_bicubic/test/test_blur_bicubic/X4_allinone"
dst_path = "E:/Dataset/public_dataset/NTIRE2021/Track1/test_blur_bicubic/test/test_blur_bicubic/dst"
start_name = "000_00000009.png"


def get_video_split_id(folder_path,dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    video_folder_ids = os.listdir(folder_path)
    for id in video_folder_ids:
        current_folder_with_id = os.path.join(folder_path,id)
        filenames = os.listdir(current_folder_with_id)
        for file in filenames:
            current_file_with_id = os.path.join(current_folder_with_id, file)
            new_file_name= id+"_"+file
            new_file_path = os.path.join(dst_path, new_file_name)
            shutil.copy2(current_file_with_id, new_file_path)

def choose_video_frame(working_path):
    filelist = os.listdir(working_path)
    for idx in range(9, len(filelist), 10):
        TODO_file = os.path.join(working_path, filelist[idx])
        shutil.copy(TODO_file, os.path.join(dst_path, filelist[idx]))


# choose_video_frame(working_path)
if __name__ == "__main__":
    get_video_split_id(folder_path="/media/cydiachen/Cydia-256ssd/val_sharp/val/val_sharp",
                       dst_path="/media/cydiachen/Cydia-256ssd/val_sharp/val/val_sharp_allinone")
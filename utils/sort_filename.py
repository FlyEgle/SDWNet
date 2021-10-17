import os
import shutil

def get_video_split_id(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    video_folder_ids = os.listdir(src_path)
    for id in video_folder_ids:
        current_folder_with_id = os.path.join(src_path,id)
        filenames = os.listdir(current_folder_with_id)
        for file in filenames:
            current_file_with_id = os.path.join(current_folder_with_id, file)
            new_file_name = id + "_" + file
            new_file_path = os.path.join(dst_path, new_file_name)
            shutil.copy2(current_file_with_id, new_file_path)

def choose_video_frame(working_path, result_path):
    filelist = os.listdir(working_path)
    filelist = sorted(filelist)
    for idx in range(9,len(filelist), 10):
        TODO_file = os.path.join(working_path, filelist[idx])
        shutil.copy(TODO_file, os.path.join(result_path, filelist[idx]))

if __name__ == "__main__":
    get_video_split_id("/media/cydiachen/Cydia-256ssd/train_blur_jpeg/train/train_blur_jpeg",
                       "/media/cydiachen/Cydia-256ssd/train_blur_jpeg/train/train_blur_jpeg_allinone")

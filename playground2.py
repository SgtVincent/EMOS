import os
import gzip
import shutil

# def unzip_gz_files_in_directory(base_directory):
#     for root, dirs, files in os.walk(base_directory):
#         for file in files:
#             if file.endswith('.gz'):
#                 gz_file_path = os.path.join(root, file)
#                 output_file_path = os.path.join(root, file[:-3])  # 去掉 .gz 后缀

#                 # 解压缩 .gz 文件
#                 with gzip.open(gz_file_path, 'rb') as gz_file:
#                     with open(output_file_path, 'wb') as out_file:
#                         shutil.copyfileobj(gz_file, out_file)

# # 示例用法
# base_directory = './data/datasets/hssd_scene'  # 替换为包含文件夹的基目录
# unzip_gz_files_in_directory(base_directory)
# print("finish!")
# template_name = "1388270c7f27a56d274c87614bfba00644d7b1aa_part_5"
# template_name = template_name.split('_',1)[0]
# print("template_name:",template_name)
# import json
# sample_info = "[{\"episode_id\": 3, \"sample_frame\": [[1, 0], [16, 0], [41, 0], [42, 0], [43, 0], [43, 0], [160, 0], [176, 0], [177, 0], [178, 0]]}, {\"episode_id\": 2, \"sample_frame\": [[1, 0], [22, 0], [51, 0], [52, 0], [53, 0], [53, 0], [238, 0], [268, 0], [293, 0], [312, 0], [313, 0], [314, 0]]}, {\"episode_id\": 1, \"sample_frame\": [[1, 0], [85, 0], [114, 0], [139, 0], [165, 0], [187, 0], [188, 0], [189, 0], [189, 0], [491, 0], [500, 0], [501, 0], [502, 0]]}]"
# sample_info_str = json.loads(sample_info)

# for episode_ in sample_info_str:
#     print(int(episode_["episode_id"]))
# import cv2
# import numpy as np

# mask_img = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
# target_img = cv2.imread('TESTSET_dataset_hssd_13scene_3733/103997919_171031233/data_2.json.gz/episode_0/frame_121_agent_0_head_rgbFetchRobot_head_rgb.png', cv2.IMREAD_UNCHANGED)

# assert target_img.shape[:2] == mask_img.shape[:2], "两张图片的大小必须一致"

# # 分离mask图片的alpha通道
# alpha_channel = mask_img[:, :, 3] / 255.0

# # 创建一个空白图片用于存储结果
# result_img = np.zeros_like(target_img)

# # 使用alpha通道进行混合
# for c in range(0, 3):
#     result_img[:, :, c] = (alpha_channel * mask_img[:, :, c] +
#                            (1 - alpha_channel) * target_img[:, :, c])

# # 保存结果图片
# dataset_path = "./sat_DLC_13scene_dataset_1108_greenpoint"
# image_dataset_path = os.path.join(dataset_path, "image")

# file_dir_path_start = [os.path.join(image_dataset_path,name) for name in os.listdir(image_dataset_path)]
# file_dir_path_start = sorted(file_dir_path_start)
# print(len(file_dir_path_start))
# print("file_dir_path_start",file_dir_path_start)
# file_dir_path = file_dir_path_start[start_dir:end_dir]
# def get_numbers_from_filenames(directory):
#     numbers = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.scene_instance.json'):
#             # 提取文件名中 .scene_instance 前的数字部分
#             number = filename.split('.scene_instance')[0]
#             numbers.append(number)
#     return numbers

# directory_path = 'data/scene_datasets/hssd-hab/scenes'
# numbers_list = get_numbers_from_filenames(directory_path)
# print(numbers_list)
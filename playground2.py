import os
import gzip
import shutil
import numpy as np
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
# need_name = ["DATASET_filter_train","DLC_train18_dataset","DLC_train40_dataset"]
# folder_paths = [os.path.join('./', folder) 
#                 for folder in os.listdir('./') if (os.path.isdir(os.path.join('./', folder)) and any(name in folder for name in need_name))]
# print(sorted(folder_paths))
# sample_scene_dir_path = os.path.join('data/datasets','hssd_scene_1220')
# sample_scene_dir = []
# for entry in os.scandir(sample_scene_dir_path):
#     if entry.is_dir():
#         sample_scene_dir.append(os.path.basename(entry.path))
# sample_scene_dir = sorted(sample_scene_dir)
# print(len(sample_scene_dir))

# import torch

# num_gpus = torch.cuda.device_count()
# for i in range(num_gpus):
#     print(torch.cuda.get_device_name(i))
# hfov = 1.5707963267948966
#         # Intrinsic matrix K
# K = np.array([
#     [1 / np.tan(hfov / 2.), 0., 0., 0.],
#     [0., 1 / np.tan(hfov / 2.), 0., 0.],
#     [0., 0., 1, 0],
#     [0., 0., 0, 1]
# ])
# print(K)
import json
import re

def extract_fields(input_string):
    # 定义正则表达式模式来匹配指定字段的内容
    pattern = r'["\'](reasoning|action|action_information|summarization)["\']\s*:\s*["\'](.*?)["\'],'
    
    # 使用正则表达式查找所有匹配的字段
    matches = re.findall(pattern, input_string)
    
    # 创建一个新的字典来存储提取的字段
    extracted_data = {field: value for field, value in matches}
    
    # 将字典转换为JSON字符串
    
    return extracted_data

# # 示例输入字符串
input_string = '''{
    "reasoning": "The robot's task is to move the cracker box from the Home library to the Central countertop. Based on the history, the robot was instructed to search frame 6 in the Home library for the cracker box. However, the current image provided (Image-9) does not show the cracker box, and there are no indications or green points suggesting an actionable object or container. Thus, the robot should proceed to search frame 6, as per its previous instruction, to locate the target object.",
    "action": "search_scene_frame",
    "action_information": "6",
    "summarization": "The robot is continuing its task by searching frame 6 in the Home library to locate the cracker box."
}
'''

# 调用函数并打印结果
new_json = extract_fields(input_string)
# new_json["task_prompt"] = "fuck"
print(new_json)

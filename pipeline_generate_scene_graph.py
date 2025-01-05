import os
import multiprocessing,logging
from multiprocessing import get_logger
import subprocess
import time
from ruamel.yaml import YAML
yaml = YAML()
import argparse
import random,shutil
from tqdm import tqdm
def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def run_scene_graph_generate(args):
    # process_name = multiprocessing.current_process().name
    data_path,gpu_id = args
    item = str(os.path.basename(os.path.normpath(data_path)))
    dataset_path = os.path.join(data_path, 'scene_graph.gz')
    # print("dataset_path",dataset_path)
    output_path = data_path
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    seed = random.randint(1, 100000000)
    min_point_dis = 2.0
    zxz_yaml_path = './habitat-lab/habitat/config/benchmark/single_agent/config_fetch.yaml'
    with open(zxz_yaml_path,'r') as file:
        data = yaml.load(file)
    data['habitat']['dataset']['data_path'] = dataset_path
    data['habitat']['simulator']['habitat_sim_v0']['gpu_device_id'] = gpu_id
    data['habitat']['simulator']['seed'] = seed
    scene_yaml_path = os.path.join('./habitat-lab/habitat/config/benchmark/single_agent/override','scene_graph_pipeline',f'{yaml_dir_name}')
    os.makedirs(scene_yaml_path,exist_ok=True)
    config_yaml_path = os.path.join(scene_yaml_path,f'{item}.yaml')
    with open(config_yaml_path,'w') as file:
        yaml.dump(data,file)
    seed_path = os.path.join(output_path,"scene_graph_seed.txt")   
    with open(seed_path,'w') as f:
        f.write(str(seed))
    command = [
        "python",
        "-u",
        "habitat-lab/habitat/datasets/rearrange/generate_episode_graph_images.py",
        "--config", f"benchmark/single_agent/override/scene_graph_pipeline/{yaml_dir_name}/{item}.yaml",
        "--gpu_id", f"{gpu_id}",
        "--output_dir", f"{output_path}",
        "--seed", f"{seed}",
        "--min_point_dis",f"{min_point_dis}",
        "--meta_json_path",f"{data_path}/scene_graph_info.json",
        "--max_images",f"{max_images}",
        "--generate_type","scene_graph"
    ]
    log_file = f"{log_path}/{item}.log"
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    time.sleep(0.5)

if __name__ == '__main__':
    #!!!!需要重新配置一下zxz_fetch.yaml的lab_sensors和image_filter_list
    dataset_path = "./sat_DATASET_0101_train_armworkspace_rgb"
    parser = argparse.ArgumentParser(description="Setup dataset and log paths.")
    parser.add_argument('--start_dir', type=int, required=True, help='Start')
    parser.add_argument('--end_dir', type=int, required=True, help='End')
    parser.add_argument('--gpu_num', type=int, default=5)
    args = parser.parse_args()
    start_dir = args.start_dir
    end_dir = args.end_dir
    image_dataset_path = os.path.join(dataset_path, "image")
    yaml_dir_name = f"startfrom_{start_dir}endat_{end_dir}"
    log_path = f'./log/scene_graph_pipeline/{yaml_dir_name}'
    # clear_directory(log_path)
    os.makedirs(log_path, exist_ok=True)
    file_dir_path_start = [os.path.join(image_dataset_path,name) for name in os.listdir(image_dataset_path)]
    file_dir_path_start = sorted(file_dir_path_start)
    # print("file_dir_path_start",file_dir_path_start)
    file_dir_path = file_dir_path_start[start_dir:end_dir]
    process_num = 50
    gpu_num = args.gpu_num
    max_images = 8
    # scene_sample = ["102344115","103997919_171031233","104348463_171513588","103997970_171031287",
    # "108736689_177263340","102344193","107733912_175999623"]
    # scene_sample = ["102344115","103997919_171031233","104348463_171513588","103997970_171031287",
    # "108736689_177263340","102344193","107733912_175999623"]
    # scene_sample = ["102344193","103997970_171031287","104348463_171513588","108294465_176709960","108736689_177263340"]
    # scene_sample = ["108294465_176709960","107733912_175999623","102344193","103997970_171031287","108736689_177263340"]
    with multiprocessing.Pool(processes=process_num) as pool:
        args = [(item,int(i%gpu_num)) for i,item in enumerate(file_dir_path)]
        for _ in tqdm(pool.imap_unordered(run_scene_graph_generate,args), total=len(file_dir_path), desc="Process Files"):
            pass


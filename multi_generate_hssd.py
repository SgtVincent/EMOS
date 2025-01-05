import os
import multiprocessing
import subprocess
import time
import argparse
from tqdm import tqdm
from threading import Timer
import gc
import psutil,sys

def check_memory_usage(threshold_mb):
    # 获取当前进程的内存信息
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)  # 转换为MB
    for child in process.children(recursive=True):
        memory_usage_mb += child.memory_info().rss / (1024 * 1024)
    print(f"当前内存使用: {memory_usage_mb:.2f} MB")

    # 检查是否超过阈值
    if memory_usage_mb > threshold_mb:
        print(f"内存使用超过 {threshold_mb} MB，程序即将退出。")
        sys.exit()
def terminate_pool(pool):
    if pool is not None:
        pool.terminate()
        pool.close()
        pool.join()
        # print("Terminated pool due to timeout.")
def run_episode_generator(args):
    # 获取当前进程的名称
    process_name = multiprocessing.current_process().name
    
    data_name,gpu_id,item,yaml_dir,output_dir = args
    # 生成基于进程名称的输出文件名
    output_file = f"./{output_dir}/{item}/{data_name}.json.gz"
    command = [
        "python",
        "./habitat-lab/habitat/datasets/rearrange/run_episode_generator.py",
        "--run",
        "--config", f"{yaml_dir}/{item}.yaml",
        "--gpu_id", f"{gpu_id}",
        "--num-episodes", f"{batch_num}",
        "--out", f"{output_file}",
        "--type", "manipulation",
        "--resume", "habitat-mas/habitat_mas/data/robot_resume/FetchRobot_default.json"
    ]
    log_file = f"./log/sample/{data_name}.log"
    # with open(log_file, "w") as f:
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # with open(log_file, 'w') as log:
    #     subprocess.run(command, stdout=log, stderr=log)
    time.sleep(0.9)
def get_numbers_from_filenames(directory):
    numbers = []
    for filename in os.listdir(directory):
        if filename.endswith('.scene_instance.json'):
            # 提取文件名中 .scene_instance 前的数字部分
            number = filename.split('.scene_instance')[0]
            numbers.append(number)
    return numbers
def extract_keys_from_txt(file_path):
    keys = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or not line.startswith('(') or not line.endswith(')'):
                continue
            try:
                content = line[1:-1].split(',', 1)
                key = content[0].strip().strip("'")
                keys.append(key)
            except Exception as e:
                print(f"Error processing line: {line}, Error: {e}")
    return keys
if __name__ == '__main__':
    sum_episode = 400
    process_num = 50
    batch_num = 4
    parser = argparse.ArgumentParser(description="Setup dataset and log paths.")
    parser.add_argument('--start_dir', type=int, required=True, help='Start')
    parser.add_argument('--end_dir', type=int, required=True, help='End')
    parser.add_argument('--gpu_num', type=int, default=5)
    args = parser.parse_args()
    start_dir = args.start_dir
    end_dir = args.end_dir
    gpu_num = args.gpu_num
    num = int(sum_episode / batch_num)
    yaml_dir = "./allycb_dir"
    output_dir = 'hssd_scene_allycb_train_2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    zip_files = [f"data_{i}" for i in range(0,int(sum_episode/batch_num))]
    # scene_sample
    # 108294465_176709960
    # scene_sample = ["102344115","103997919_171031233","104348463_171513588","103997970_171031287",
    # "108736689_177263340","102344193","107733912_175999623"]
    #this is 13 train scene sampling
    #     scene_sample = ["104348463_171513588","103997970_171031287","108736689_177263340","102344193","107733912_175999623",
    # "102816786","103997643_171030747","105515211_173104185","102344115","103997919_171031233","102344457","102344529","104348202_171513150"]
    #this is 5 test unseen scene sampling
    # scene_sample = ["104862384_172226319","106878960_174887073","108736851_177263586","108736824_177263559","107734254_176000121"]
    #sample all!
    # scene_config_directory_path = 'data/scene_datasets/hssd-hab/scenes'
    # scene_sample = get_numbers_from_filenames(scene_config_directory_path)
    # scene_sample = sorted(scene_sample)
    # print("before_filer_scene_sample_len:",len(scene_sample))
    # # scene_sample = scene_sample[140:160]
    # # 读取文件中的字符串
    # with open('scene_have_sampled/data_1203.txt', 'r') as file:
    #     file_strings = set(line.strip() for line in file)
    # filtered_scene_sample = [s for s in scene_sample if s not in file_strings]
    # #移除重合的
    # scene_sample = filtered_scene_sample
    # print("scene_sample_len:",len(scene_sample))
    # scene_sample = scene_sample[15:40]
    #if you want to filter from all the scene,do the thing up.
    timeout = 1800
    
# 示例用法
    file_path = 'train_id.txt'  # 替换为您的文件路径
    keys = extract_keys_from_txt(file_path)
    log_path = "./log/sample"
    scene_sample = keys[start_dir:end_dir]
    os.makedirs(log_path,exist_ok=True)
    memory_threshold = 450976
    for a,item in enumerate(scene_sample):
        if not os.path.exists(os.path.join(output_dir, item)):
            os.makedirs(os.path.join(output_dir, item))
        start_time = time.time()
        check_memory_usage(memory_threshold)
        print(f"START--{item}//i---{str(a)}")
        with multiprocessing.Pool(processes=process_num) as pool:
            args = [(f"data_{i}",int(i%gpu_num),item,yaml_dir,output_dir) for i in range(0,int(sum_episode/batch_num))]
            results = []
            async_results = [pool.apply_async(run_episode_generator, (arg,), error_callback=lambda e: print(f"Error: {e}")) for arg in args]
            timer = Timer(timeout, terminate_pool, [pool])
            timer.start()
            try:
                pool.close()
                pool.join()
            finally:
                timer.cancel()
                end_time = time.time()
                item_process_time = end_time - start_time
                terminate_pool(pool)
            time.sleep(0.9)
            print(f"FINISH--{item};TIME:{item_process_time}")
        gc.collect()
    # timeout = 600
    # with multiprocessing.Pool(processes=process_num) as pool:
    #     for item in scene_sample:
    #         if not os.path.exists(os.path.join(output_dir, item)):
    #             os.makedirs(os.path.join(output_dir, item))
    #         args = [(f"data_{i}", int(i % gpu_num), item, yaml_dir, output_dir) for i in range(0, int(sum_episode / batch_num))]
    #         results = []
    #         # for _ in tqdm(pool.imap_unordered(run_episode_generator, args), total=num, desc=f"Processing {item} configs"):
    #         #     pass
    #         for arg in args:
    #             result = pool.apply_async(run_episode_generator, (arg,))
    #             results.append(result)
    #         for result in tqdm(results, total=len(args), desc=f"Processing {item} configs"):
    #             try:
    #                 result.get(timeout=timeout)
    #             except Exception as e:
    #                 print(f"Timeout or error occurred for {item}: {e}")
    #                 break  # 超时或发生错误时跳出循环，进入下一个 item
    # with tqdm(total=num,desc="Sample_Episode") as pbar:
    #     for j in range(0,num):
    #         start_time = time.time()
    #         processes = []
    #         for i in range(process_num):  
    #             process_name = f"process_{i + (process_num*j)}"
    #             p = multiprocessing.Process(target=run_episode_generator, name=process_name)
    #             p.start()
    #             processes.append(p)
            
    #         for p in processes:
    #             p.join()
            
    #         end_time = time.time()
    #         batch_time = end_time-start_time
    #         print(f"Batch {j} completed.")
    #         pbar.update(1)
    #         time.sleep(0.5) 


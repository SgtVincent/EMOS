import json
import numpy as np
import cv2
import os
import magnum
#可以与第一遍采数据合并！
def process_directory(base_dir,skip_len):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            json_path = os.path.join(dir_path, 'sum_data.json')
            
            if os.path.exists(json_path) and dir_name != "episode_0":
                try:
                # print("dir_name---------------------:",dir_name)
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                    result = []
                    data_trans = []
                    action = ["turn","nav_to_point", "pick","turn","nav_to_point", "place"]
                    agent_0_action = 0
                    nav_slide = []
                    i = 0
                    while(i < len(data['entities'])):
                        if data['entities'][i]['data']['agent_0_has_finished_oracle_nav'] == [1.0]:
                            nav_slide.append(i+1)
                            i+=1
                        i+=1
                    for i in range(0, len(data['entities'])):
                        step_data = data['entities'][i]
                        # agent_0_trans_matrix = step_data['data']['agent_0_robot_trans_martix']
                        agent_0_nowloc = step_data['data']['agent_0_localization_sensor']
                        # agent_1_nowloc = step_data['data']['agent_1_localization_sensor']
                        # agent_1_trans_matrix = step_data['data']['agent_1_robot_trans_martix']
                        # agent_0_eeglobal = step_data['data']['agent_0_ee_global_pos_sensor']
                        # agent_1_eeglobal = step_data['data']['agent_1_ee_global_pos_sensor']
                        # agent_1_pre_worldloc = next_step_data['data']['agent_1_localization_sensor']
                        agent_0_obj = step_data['data']['agent_0_obj_bounding_box']
                        # agent_1_objpos = step_data['data']['agent_1_obj_pos']
                        agent_0_camera = step_data['data']['agent_0_camera_extrinsic']
                        agent_0_target = step_data['data']['agent_0_target_bounding_box']
                        agent_0_rec_old = step_data['data']['agent_0_rec_bounding_box']
                        agent_0_rec = [[min(b[0] for b in agent_0_rec_old),
                                       min(b[1] for b in agent_0_rec_old),
                                       max(b[2] for b in agent_0_rec_old),
                                       max(b[3] for b in agent_0_rec_old)]]
                        # agent_0_pre_eepos = trans_worldloc_to_robotloc(agent_0_trans_matrix, agent_0_eeglobal)
                        # agent_1_pre_eepos = trans_worldloc_to_robotloc(agent_0_trans_matrix, agent_1_eeglobal)
                        # agent_0_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_pre_worldloc[:3])
                        # agent_1_pre_robotloc = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_pre_worldloc[:3])
                        # agent_0_obj_ro = trans_worldloc_to_robotloc(np.array(agent_0_trans_matrix), agent_0_objpos[:3])
                        # agent_1_obj_ro = trans_worldloc_to_robotloc(np.array(agent_1_trans_matrix), agent_1_objpos[:3])
                        # agent_0_prepix = _project_points_to_image(points_3d=agent_0_obj_ro, point_3d_match=agent_0_pre_robotloc[:3], camera_info=camera_info, cam_trans=camera_extrinsic_old)
                        # agent_1_prepix = _project_points_to_image(points_3d=agent_1_obj_ro, point_3d_match=agent_1_pre_robotloc[:3], camera_info=camera_info, cam_trans=camera_extrinsic_old)
                        if (agent_0_action == 0 or agent_0_action == 3) and agent_0_nowloc[:3]!= data['entities'][i+1]['data']['agent_0_localization_sensor'][:3]:
                            agent_0_action+=1
                        if (agent_0_action == 1 or agent_0_action == 4) and step_data['data']['agent_0_has_finished_oracle_nav'] == [1.0]:
                            agent_0_action += 1
                        if agent_0_action == 2 and data['entities'][i]['data']['agent_0_localization_sensor'] != data['entities'][i-1]['data']['agent_0_localization_sensor']:
                            agent_0_action += 1
                        # if (agent_1_action == 0 or agent_1_action == 2) and step_data['data']['agent_1_has_finished_oracle_nav'] == [1.0]:
                        #     agent_1_action += 1
                        # if agent_1_action == 1 and data['entities'][i]['data']['agent_1_localization_sensor'] != data['entities'][i-1]['data']['agent_1_localization_sensor']:
                        #     agent_1_action += 1

                        result = {
                            "step": i + 1,
                            "agent_0_now_worldloc":agent_0_nowloc,
                            "agent_0_obj": agent_0_obj,
                            "agent_0_rec": agent_0_rec,
                            "agent_0_target": agent_0_target,
                            "agent_0_martix": agent_0_camera,
                            # "agent_1_pre_worldloc":agent_1_pre_worldloc,
                            # "agent_0_pre_robotloc": agent_0_pre_robotloc.tolist(),
                            # "agent_0_trans_matrix": agent_0_trans_matrix,
                            # "agent_1_trans_matrix": agent_1_trans_matrix,
                            # "agent_1_nowloc": agent_1_nowloc,
                            # "agent_1_pre_robotloc": agent_1_pre_robotloc.tolist(),
                            # "agent_0_obj_ro": agent_0_obj_ro.tolist(),
                            # "agent_1_obj_ro": agent_1_obj_ro.tolist(),
                            "agent_0_action": action[agent_0_action],
                            # "agent_1_action": action[agent_1_action]
                        }
                        data_trans.append(result)
                    data_dir_path = os.path.join(dir_path, 'data')
                    if not os.path.exists(data_dir_path):
                        os.makedirs(data_dir_path)
                    output_json_path = os.path.join(data_dir_path, 'data_trans.json')
                    # output_json_path = os.path.join(dir_path, 'data_trans.json')
                    with open(output_json_path, 'w') as file:
                        json.dump(data_trans, file, indent=2)
                except:
                    continue    

# base_directory = './video_dir/'
# gz_name = f"manipulation_eval_process_0.json.gz"
# num_gz = 50
# for i in range(0,num_gz):
#     process_directory(os.path.join(base_directory,f"process_{i}.json.gz"))
process_directory('./video_dir/process_1.json.gz',30)
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re,shutil
import base64
import requests
import time,random
from PIL import Image
from io import BytesIO
from openai import OpenAI
from torchvision import transforms
import torch
from habitat_mas.pivot.run_pivot import run_pivot
json_format_info = {
    "reasoning":"Based on the current image information and history, think and infer the actions that need to be executed and action's information.",
    "action":"The action name of your reasoning result.",
    "action_information":"If the action is \"nav_to_point\",\"pick\" or \"place\", the information format should be \"<box>[[x1, y1, x2, y2]]</box>\" to indicate location information,where x1, y1, x2, y2 are the coordinates of the bounding box;if the action is \"search_scene_frame\",the information format should be \"id\",where id is the value of the frame index you choose.",
    "summarization":"summarize based on the current action information and history."
}
pivot_json_format_info = {
    "reasoning":"Based on the current image information and history, think and infer the actions that need to be executed and action's information.",
    "action":"The action name of your reasoning result.",
    "action_information":"If the action is \"nav_to_point\",\"pick\" or \"place\", the information format should be \"<box>[[x1, y1, x2, y2]]</box>\" to indicate location information,where x1, y1, x2, y2 are the coordinates of the bounding box;if the action is \"search_scene_frame\",the information format should be \"id\",where id is the value of the frame index you choose.",
    "summarization":"summarize based on the current action information and history."
}
def save_image(image, file_path):
    from PIL import Image
    img = Image.fromarray(image)
    img.save(file_path)
class PIVOTAgent:
    def __init__(self,**kwargs):
        self.robot_history = None
        self.agent_name = kwargs.get("agent_name", "agent_0")
        self.client = OpenAI(api_key='123', base_url='http://0.0.0.0:23333/v1')
        self.prepare_action_num = 3
        self.rgb_image_store_num = 0
        self.rag_image_name_list_set = [
'target_rec.png', 'goal_rec.png',
'random_scene_graph_2.png', 'random_scene_graph_3.png',
'random_scene_graph_4.png', 'random_scene_graph_5.png',
'random_scene_graph_6.png', 'random_scene_graph_7.png']
        self.prepare_action = [{
                    "name": "reset_arm",
                    "arguments": ['agent_0']
                },
                {
                    "name": "wait",
                    "arguments": ['1','agent_0']
                }]
        self.use_pivot = True

    def _init_model(self):
        client = OpenAI(api_key='', base_url='http://0.0.0.0:23333/v1')
        model_name = client.models.list().data[0].id
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                'role':
                'user',
                'content': [{
                    'type': 'text',
                    'text': 'describe this image',
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url':
                        'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
                    },
                }],
            }],
            temperature=0.8,
            top_p=0.8)
    def _2d_to_3d_single_point(self, depth_obs, depth_rot,depth_trans,pixel_x, pixel_y):
        # depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

        # hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
        # W, H = depth_camera.viewport[0], depth_camera.viewport[1]
        W = 512
        H = 512
        hfov = 1.5707963267948966
        # Intrinsic matrix K
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])

        # Normalize pixel coordinates
        xs = (2.0 * pixel_x / (W - 1)) - 1.0  # normalized x [-1, 1]
        ys = 1.0 - (2.0 * pixel_y / (H - 1))  # normalized y [1, -1]

        # Depth value at the pixel
        depth = depth_obs[0,pixel_y,pixel_x,0]

        # Create the homogeneous coordinates for the pixel in camera space
        xys = np.array([xs * depth, ys * depth, -depth, 1.0]).reshape(4, 1)

        # Apply the inverse of the intrinsic matrix to get camera space coordinates
        xy_c = np.matmul(np.linalg.inv(K), xys)

        # Get the rotation and translation of the camera
        depth_rotation = np.array(depth_rot)
        depth_translation = np.array(depth_trans)

        # Get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        # Apply transformation to get world coordinates
        T_camera_world = np.linalg.inv(T_world_camera)
        points_world = np.matmul(T_camera_world, xy_c)

        # Get non-homogeneous points in world space
        points_world = points_world[:3, :] / points_world[3, :]
        return points_world.flatten()
    def get_episode_prompt(self,data_path):
        data_dir_path = os.path.dirname(data_path)
        data_dir_num = os.path.basename(data_dir_path)
        prompt_json_path = os.path.join(os.path.dirname(data_dir_path),'test.json')
        with open(prompt_json_path, 'r') as file:
            data = json.load(file)
        task_description = next((item["task_description"] for item in data if item["image_number"] == str(data_dir_num)), None)
        assert task_description,"OH NO!CAN NOT FIND CURRENT TASK DESCRIPTION"
        return task_description
    def process_prompt(self,task_prompt,history = None):
        if not history:
            robot_history = ""
        else:
            robot_history = f"Robot's history:\"{history}\""
        image_token_pad = '<IMAGE_TOKEN>'
        # set json format for pivot
        json_format = pivot_json_format_info if self.use_pivot else json_format_info
        question_prompt_align_with_ovmm = f"""You are an AI visual assistant that can manage a single robot. You receive the robot's task,one image representing the robot's current view and eight frames of the scene from the robot's tour. You need to output the robot's next action. Actions the robot can perform are "search_scene_frame","nav_to_point","pick" and "place".
These frames are from the robot's tour of the scene:
Image-1:{image_token_pad}
Image-2:{image_token_pad}
Image-3:{image_token_pad}
Image-4:{image_token_pad}
Image-5:{image_token_pad}
Image-6:{image_token_pad}
Image-7:{image_token_pad}
Image-8:{image_token_pad}
If you can not find the target you need to identify,you should find the frame that the robot should navigate to complete the task,and output "search_scene_frame" action and the id of frame.
Robot's current view is: Image-9:{image_token_pad}.If you can find the position that you should navigate to, pick or place,you should output your action information.In robot's current view, some green points may appear,indicating the positions that the robot's arm can reach. When there are green points on the object that needs to be picked, it means the robot's arm can pick up the object. When there are enough green points on the goal container where the object needs to be placed, it means the robot's arm can place the object into the goal container.
Besides,you need to explain why you choose this action in your output and summarize by combining your chosen action with historical information.
Robot's Task: {task_prompt}{robot_history}Your output format should be in pure JSON format as follow:{json_format}."""
        return question_prompt_align_with_ovmm
    def process_message(self,data_path,robot_image,prompt): #sr:send&receive
        content = [
            {
                "type":"text",
                "text": prompt
            },
        ]
        data_dir_path = os.path.dirname(data_path)
        rag_image_path_list = [os.path.join(data_dir_path,item)
                               for item in self.rag_image_name_list_set]
        random.shuffle(rag_image_path_list)
        for rag_image_path in rag_image_path_list:
            with open(rag_image_path,"rb") as rag_image:
                temp_image_info = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(rag_image.read()).decode('utf-8')}",
                    }
                }
                content.append(temp_image_info)
        robot_image_PIL = Image.fromarray(np.squeeze(robot_image))
        robot_image_buffered = BytesIO()
        robot_image_PIL.save(robot_image_buffered,format = 'PNG')
        robot_image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(robot_image_buffered.getvalue()).decode('utf-8')}",
                # "url": f"data:image/png;base64,{base64.b64encode(robot_image_buffered.getvalue()).decode('utf-8')}",
            }
        }
        content.append(robot_image_info)
        return content,rag_image_path_list
    def query_and_receive(self,content,client):
        model_name = client.models.list().data[0].id
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                'role':'user',
                'content': content
            }],
            temperature=0.0,
            top_p=0.5)
        return response
    def process_nav_point(self,depth_obs, depth_rot,depth_trans,pixel_xy):
        pixel_x, pixel_y = pixel_xy
        point_3d = self._2d_to_3d_single_point(depth_obs.cpu(), depth_rot.cpu(),depth_trans.cpu(),pixel_x, pixel_y)
        IGNORE_NODE = [-100]
        point_3d_debug = np.concatenate((point_3d, IGNORE_NODE))
        # print("1:point_3d_debug:",point_3d_debug)
        # point_3d_debug = point_3d_debug[0]
        # print("2:point_3d_debug",point_3d_debug)
        # point_3d_debug = point_3d_debug.tolist()
        # print("3:point_3d_debug",point_3d_debug)
        return point_3d_debug
    def process_action(self,action,action_information,data_path,observations,rgb_image_path_list):
        if action == "search_scene_frame":
            data_dir_path = os.path.dirname(data_path)
            metadata_json_path = os.path.join(data_dir_path,'metadata.json')
            # print("rgb_image_path_list:",rgb_image_path_list)
            basename_list = [os.path.basename(item)
                             for item in rgb_image_path_list]
            print("basename_list:",basename_list)
            print("action_information_sc:",action_information)
            rgb_image_path = rgb_image_path_list[int(action_information)-1]
            rgb_image_name = os.path.basename(rgb_image_path)
            with open(metadata_json_path, 'r') as file:
                data = json.load(file)
            for item in data:
                if rgb_image_name in item["obs_files"]:
                    nav_position = item["position"]
            assert nav_position,"Code has bug"
            return {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": nav_position,
                    "robot": self.agent_name,
                }
            }
        if action == "nav_to_point":
            action_bbox = json.loads(action_information)[0]
            print("action_bbox:",action_bbox)
            action_point = [int((action_bbox[0] + (action_bbox[2]/2.0))*512.0/1000.0),
                            int((action_bbox[1] + (action_bbox[3]/2.0))*512.0/1000.0)]
            target_position = self.process_nav_point(observations['depth_obs'],
                                                  observations['depth_rot'],observations['depth_trans'],action_point),
            print("target_position:",target_position)
            target_position = target_position[0]
            target_position = target_position.tolist()
            return {
                "name": "nav_to_position",
                "arguments": {
                    "target_position":target_position,
                    "robot": self.agent_name,
                }
            }
        if (action == "pick") or (action == "place"):
            action_bbox = json.loads(action_information)[0]
            action_point = [int((action_bbox[0] + (action_bbox[2]/2.0))*512.0/1000.0),
                            int((action_bbox[1] + (action_bbox[3]/2.0))*512.0/1000.0)]
            return {
                "name": f"{action}_key_point",
                "arguments": {
                    "position": action_point,
                    "robot": self.agent_name,
                }
            }
        return {
            "name": f"wait",
            "arguments": ['3000',self.agent_name]
        }
    def process_vlm_output(self,data_path,vlm_output,rgb_image_path_list,observations,action_information=None):
        # print("vlm_output:",vlm_output)
        vlm_output = json.loads(vlm_output.choices[0].message.content)
        print("vlm_output:",vlm_output)
        vlm_action = vlm_output['action']
        if self.use_pivot:
            vlm_action_information = action_information
        else:
            vlm_action_information = vlm_output['action_information']
        vlm_action_summarization = vlm_output['summarization']
        vlm_return = self.process_action(
            vlm_action,vlm_action_information,data_path,observations,rgb_image_path_list
        )
        return vlm_return,vlm_action_summarization
    def vlm_eval_response(self,observations,data_path, camera_info=None):
        if self.prepare_action_num:
            action_num = self.prepare_action_num
            self.prepare_action_num -= 1
            return self.prepare_action[2-action_num]
        self.prepare_action_num = 2
        #this is for action cycle(ensure reset&wait before and action)
        robot_image = observations["arm_workspace_rgb"].cpu().numpy()
        # save_image(np.squeeze(robot_image.copy()),f"./eval_in_sim_info/{os.path.basename(os.path.dirname(data_path))}/robot_input_{self.rgb_image_store_num}.png")
        # print("robot_image:",robot_image.shape)
        self.rgb_image_store_num+=1
        task_prompt = self.get_episode_prompt(data_path)
        prompt = self.process_prompt(task_prompt,history = self.robot_history)
        content, rgb_image_path_list = self.process_message(data_path,robot_image,prompt)
        # print("query_content:",content)
        vlm_output = self.query_and_receive(content,self.client)

        # TODO: Query pivot
        if self.use_pivot:
            text_vlm_output = vlm_output.choices[0].message.content
            sampled_point = run_pivot(
                            im=robot_image,
                            query=text_vlm_output,
                            n_samples_init=8,
                            n_samples_opt=6,
                            n_iters=2,
                            n_parallel_trials=1,
                            # TODO: Add openai key here
                            openai_api_key=openai_key,
                            camera_info=camera_info,
                        )
            vlm_return,self.robot_history = self.process_vlm_output(
                data_path,vlm_output,rgb_image_path_list,observations,sampled_point
            )
        else:
            vlm_return, self.robot_history = self.process_vlm_output(
                data_path, vlm_output, rgb_image_path_list, observations,
            )
        return vlm_return

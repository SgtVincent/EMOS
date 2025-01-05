import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
import habitat
import random, time
from habitat.config.default import get_config
from habitat.config import read_write
from habitat.core.env import Env
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.datasets.rearrange.rearrange_dataset import (
    RearrangeDatasetV0,
    RearrangeEpisode,
)
from habitat.datasets.rearrange.navmesh_utils import (
    get_largest_island_index,
)
from habitat.utils.visualizations.utils import observations_to_image


def save_image(image, file_path):
    from PIL import Image

    img = Image.fromarray(image)
    img.save(file_path)


def inrange(s, min_range, max_range):
    return min_range < s < max_range


def get_hssd_single_agent_config(cfg_path, overrides=None):
    if overrides:
        config = get_config(cfg_path, overrides)

    else:
        config = get_config(cfg_path)
    # print("config:",config)
    # with read_write(config):
    # config.habitat.task.lab_sensors = {}
    # agent_config = get_agent_config(config.habitat.simulator)
    # agent_config.sim_sensors = {
    #     "rgb_sensor": HabitatSimRGBSensorConfig(),
    #     "depth_sensor": HabitatSimDepthSensorConfig(),
    # }
    return config


# Generate a random quaternion (x, y, z, w) restricted to the x-z plane
def random_quaternion_xz_plane():
    # Restrict the orientation to the x-z plane (no y component)
    theta = np.random.uniform(0, 2 * np.pi)  # random angle in the x-z plane
    qx = 0
    qy = np.sin(theta / 2)  # No rotation around the y-axis
    qz = 0
    qw = np.cos(theta / 2)
    # qz = np.cos(theta / 2)
    # qw = 0  # Unit quaternion constraint

    return [qx, qy, qz, qw]


def calculate_orientation_xz_plane(position, goal_position):
    """
    Calculate the orientation of the agent facing towards the goal position.
    """
    dx = goal_position[0] - position[0]
    dz = goal_position[2] - position[2]

    # Calculate the angle between the x-axis and the vector from the agent's position to the goal position
    theta = np.arctan2(dz, dx)

    # Convert the angle to a quaternion
    qx = 0
    qy = np.sin(theta / 2)  # No rotation around the y-axis
    qz = 0
    qw = np.cos(theta / 2)
    # qz = np.cos(theta / 2)
    # qw = 0  # Unit quaternion constraint

    return [qx, qy, qz, qw]


def get_orientation(loc):
    final_nav_targ = loc[:3]
    theta = loc[3]
    # is_z_positive = loc[4]
    # if is_z_positive == 0:
    #     dz = 3.0
    # else:
    #     dz = -3.0
    # dx = dz / np.tan(theta)
    # x_obj = final_nav_targ[0] + dx
    # z_obj = final_nav_targ[2] + dz
    # obj_targ_pos = [x_obj,final_nav_targ[1],z_obj]
    # orientation = calculate_orientation_xz_plane(final_nav_targ,obj_targ_pos)
    return np.array(final_nav_targ), np.array([theta])


def set_articulated_agent_base_state(
    sim: RearrangeSim, base_pos, base_rot, agent_id=0
):
    """
    Set the base position and rotation of the articulated agent.

    Args:
        sim: RearrangeSim object
        base_pos: np.ndarray of shape (3,) representing the base position of the agent
        base_rot: Optional[(4,), (1)] representing the base rotation of the agent, can be quaternion or rotation_y_rad
    """
    agent = sim.agents_mgr[agent_id].articulated_agent
    agent.base_pos = base_pos
    if len(base_rot) == 1:
        agent.base_rot = base_rot
    elif len(base_rot) == 4:
        # convert quaternion to rotation_y_rad with scipy
        r = R.from_quat(base_rot)
        agent.base_rot = r.as_euler("xyz", degrees=False)[1]
    else:
        raise ValueError(
            "base_rot should be either quaternion or rotation_y_rad"
        )


def set_obj_holded_on_agent_ee(sim: RearrangeSim, obj_id, agent_id=0):
    ep_objects = []
    for key, val in sim.ep_info.info["object_labels"].items():
        ep_objects.append(key)
    objects_info = {}
    rom = sim.get_rigid_object_manager()
    for i, handle in enumerate(rom.get_object_handles()):
        # print("handle:",handle)
        if handle in ep_objects:
            obj = rom.get_object_by_handle(handle)
            objects_info[obj.object_id] = handle
    obj_id = list(objects_info.keys())[0]
    grasp_mgr = sim.grasp_mgr
    grasp_mgr.snap_to_obj(obj_id, force=True)


def get_target_objects_info(
    sim: RearrangeSim,
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Get the target objects' information: object ids, start positions, and goal positions from the rearrange episode.
    """
    target_idxs = []
    target_handles = []
    start_positions = []
    goal_positions = []
    rom = sim.get_rigid_object_manager()

    for target_handle, trans in sim._targets.items():
        target_idx = sim._scene_obj_ids.index(
            rom.get_object_by_handle(target_handle).object_id
        )
        obj_id = rom.get_object_by_handle(target_handle).object_id
        start_pos = sim.get_scene_pos()[sim._scene_obj_ids.index(obj_id)]
        goal_pos = np.array(trans.translation)

        target_idxs.append(target_idx)
        target_handles.append(target_handle)
        start_positions.append(start_pos)
        goal_positions.append(goal_pos)

    return target_idxs, target_handles, start_positions, goal_positions


def generate_scene_graph_from_store_dict(args):
    # print("seed:",args.seed)
    np.random.seed(args.seed)
    config = get_hssd_single_agent_config(args.config)
    # print("config:",config)
    # config['habitat']['simulator']['habitat_sim_v0']['gpu_device_id'] = gpu_id
    # config['habitat']['simulator']['dataset']['data_path'] = dataset_path
    # print("config:",config)
    env = Env(config=config)

    dataset = env._dataset
    metadata = []
    max_images = args.max_images
    dist_to_target = args.dist_to_target
    min_dis = args.min_point_dis
    output_dir = args.output_dir
    # print("outputdir:",output_dir)
    meta_json_path = args.meta_json_path
    obs_keys = args.obs_keys
    for episode in dataset.episodes:
        # reset automatically calls next episode
        # print("episode_info:",episode)
        env.reset()
        sim: RearrangeSim = env.sim
        # get the largest island index
        largest_island_idx = get_largest_island_index(
            sim.pathfinder, sim, allow_outdoor=True
        )
        graph_positions = []
        graph_orientations = []
        graph_annotations = []
        with open(meta_json_path, "r") as file:
            data = json.load(file)
        object_rec_loc = data[0]["localization_sensor"]
        target_rec_loc = data[1]["localization_sensor"]
        object_rec_location, object_rec_orientation = get_orientation(
            object_rec_loc
        )
        target_rec_location, target_rec_orientation = get_orientation(
            target_rec_loc
        )
        graph_positions.append(object_rec_location)
        graph_orientations.append(object_rec_orientation)
        graph_annotations.append("object_rec")
        graph_positions.append(target_rec_location)
        graph_orientations.append(target_rec_orientation)
        graph_annotations.append("goal_rec")
        # Get scene objects ids
        scene_obj_ids = sim.scene_obj_ids

        # Get target objects' information
        # object_ids, object_handles, start_positions, goal_positions = get_target_objects_info(sim)
        # Sample navigable points near the target objects start positions and goal positions
        # And calculate the orientation of the agent facing towards the position
        # print("graph_positions:",graph_positions)
        # print("graph_orientations:",graph_orientations)

        # Sample navigable points if images are less than the maximum number of images
        set_articulated_agent_base_state(
            sim, graph_positions[0], graph_orientations[0], agent_id=0
        )
        observations = env.step({"action": (), "action_args": {}})
        env._episode_over = False
        _, _, w_rec, h_rec = observations["rec_bounding_box"][0]
        print("object_rec:", w_rec * h_rec)
        set_articulated_agent_base_state(
            sim, graph_positions[1], graph_orientations[1], agent_id=0
        )
        observations = env.step({"action": (), "action_args": {}})
        env._episode_over = False
        _,_,w_tar,h_tar = observations["target_bounding_box"][0]
        print("goal_rev:",w_tar*h_rec)

        if len(graph_positions) < max_images:
            i = 0
            min_bbox = args.random_min_bbox
            while max_images > len(graph_positions):
                # Sample a random navigable point in the scene
                current_time = time.time()
                local_random = random.Random(current_time)
                random_number = local_random.randint(1, 100000)
                np.random.seed(args.seed + random_number)
                point = sim.pathfinder.get_random_navigable_point(
                    island_index=largest_island_idx
                )
                orientation = random_quaternion_xz_plane()
                min_dis_flag = 0
                min_point_dis = min_dis
                for item in graph_positions[:2]:
                    x1 = point[0]
                    y1 = point[2]
                    x2 = item[0]
                    y2 = item[2]
                    # print("dis:",int(np.sqrt(((x1-x2) ** 2)+((y1-y2) ** 2))))
                    if (
                        int(np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)))
                        < min_point_dis
                    ):
                        min_dis_flag = 1
                        break
                if min_dis_flag == 0:
                    set_articulated_agent_base_state(
                        sim, point, orientation, agent_id=0
                    )
                    observations = env.step({"action": (), "action_args": {}})
                    env._episode_over = False
                    _, _, w_tar, h_tar = observations["target_bounding_box"][0]
                    _, _, w_rec, h_rec = observations["rec_bounding_box"][0]
                    print("test:boundingbox_tar:", w_tar * h_tar)
                    print("test:boundingbox_tar:", w_rec * h_rec)
                    if w_tar * h_tar < min_bbox and w_rec * h_rec < min_bbox:
                        i += 1
                        graph_positions.append(point)
                        graph_orientations.append(orientation)
                        graph_annotations.append("random")
        # print("graph_positions:",graph_positions)
        # print("graph_orientations:",graph_orientations)
        # print("graph_annotations:",graph_annotations)
        # Create a folder for this episode
        episode_dir = output_dir
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        # print("graph_positions:",graph_positions)
        # print("graph_orientations:",graph_orientations)
        for idx, (position, orientation, annotation) in enumerate(
            zip(graph_positions, graph_orientations, graph_annotations)
        ):
            # observations = env.sim.get_observations_at(position=position, rotation=orientation)
            # print("setin:",graph_positions,graph_orientations,flush = True)
            set_articulated_agent_base_state(
                sim, position, orientation, agent_id=0
            )
            observations = env.step({"action": (), "action_args": {}})

            # Force the episode to be active
            env._episode_over = False
            obs_file_list = []
            for obs_key in obs_keys:
                if obs_key in observations:
                    image = observations[obs_key]
                    if annotation == "object_rec":
                        image_file_name = f"target_rec.png"
                    elif annotation == "goal_rec":
                        image_file_name = f"goal_rec.png"
                    else:
                        image_file_name = f"random_scene_graph_{idx}.png"
                    image_file_path = os.path.join(
                        episode_dir, image_file_name
                    )
                    save_image(image, image_file_path)
                    obs_file_list.append(image_file_name)
            # for bbox in args.bbox:
            #     if bbox in observations:
            #         print(f"{bbox}:{observations[bbox]}")
            now_loc = observations["localization_sensor"]
            metadata.append(
                {
                    "episode_id": episode.episode_id,
                    "obs_files": obs_file_list,
                    "position": now_loc,
                    "rotation": orientation,
                    # "detected_objects": observations["objectgoal"],
                }
            )

        env._episode_over = True

    # Save metadata to JSON
    metadata_file_path = os.path.join(output_dir, "metadata.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)


def generate_scene_graph_for_single_task(args):
    config = get_hssd_single_agent_config(args.config)
    env = Env(config=config)
    dataset = env._dataset
    metadata = []
    obs_keys = args.obs_keys
    sample_info = json.loads(args.sample_info)
    output_dir = args.output_dir
    for episode in dataset.episodes:
        episode_id = int(episode.episode_id)
        env.reset()
        sim: RearrangeSim = env.sim
        largest_island_idx = get_largest_island_index(
            sim.pathfinder, sim, allow_outdoor=True
        )
        episode_dir = os.path.join(output_dir, f"episode_{episode_id}")
        check_if_sample = False
        for episode_ in sample_info:
            if int(episode_["episode_id"]) == episode_id:
                episode_info = episode_["sample_frame"]
                check_if_sample = True
                break
        if not check_if_sample:
            continue
        episode_json_path = os.path.join(
            episode_dir, "data", "data_trans.json"
        )
        with open(episode_json_path, "r") as file:
            episode_json = json.load(file)  # 用data_trans.json
        for frame_info in episode_info:
            frame_num = frame_info[0]
            # print("frame_num:",frame_num)
            loc = episode_json[frame_num + 1]["agent_0_now_worldloc"]
            pos_info, orien_info = get_orientation(loc)
            set_articulated_agent_base_state(
                sim, pos_info, orien_info, agent_id=0
            )
            observations = env.step({"action": (), "action_args": {}})
            env._episode_over = False
            for obs_key in obs_keys:
                if obs_key in observations:
                    image = observations[obs_key]
                    image_file_name = f"frame_{frame_num}_agent_0_head_rgbFetchRobot_head_rgb.png"
                    image_file_path = os.path.join(
                        episode_dir, image_file_name
                    )
                    save_image(image, image_file_path)

        env._episode_over = True


def generate_episode_image_from_store_dict(args):
    config = get_hssd_single_agent_config(args.config)
    env = Env(config=config)
    dataset = env._dataset
    metadata = []
    # obs_keys = args.obs_keys
    obs_keys = ["head_rgb","arm_workspace_rgb"]
    sample_info = json.loads(args.sample_info)
    # print("sample_info:",sample_info)
    output_dir = args.output_dir
    for episode in dataset.episodes:
        # print("episode:",episode)
        # raise ValueError("test")
        episode_id = int(episode.episode_id)
        env.reset()
        sim: RearrangeSim = env.sim
        # get the largest island index
        largest_island_idx = get_largest_island_index(
            sim.pathfinder, sim, allow_outdoor=True
        )
        episode_dir = os.path.join(output_dir, f"episode_{episode_id}")
        check_if_sample = False
        for episode_ in sample_info:
            if int(episode_["episode_id"]) == episode_id:
                episode_info = episode_["sample_frame"]
                check_if_sample = True
                break
        if not check_if_sample:
            continue
        episode_json_path = os.path.join(
            episode_dir, "data", "data_trans.json"
        )
        with open(episode_json_path, "r") as file:
            episode_json = json.load(file)  # 用data_trans.json
        get_obj_json_path = os.path.join(
            episode_dir, f"episode_{episode_id}.json"
        )
        with open(get_obj_json_path, "r") as file:
            get_obj_json = json.load(file)  # 用data_trans.json
        hav_get_obj_id = -1
        for item in get_obj_json:
            if item["action"]["name"] == "search_for_goal_rec":
                hav_get_obj_id = item["step"]
                break
        for frame_info in episode_info:
            frame_num = frame_info[0]
            # print("frame_num:",frame_num)
            loc = episode_json[frame_num + 1]["agent_0_now_worldloc"]
            pos_info, orien_info = get_orientation(loc)
            set_articulated_agent_base_state(
                sim, pos_info, orien_info, agent_id=0
            )
            if frame_num == hav_get_obj_id:
                set_obj_holded_on_agent_ee(sim, 2, agent_id=0)
            observations = env.step({"action": (), "action_args": {}})
            env._episode_over = False
            for obs_key in obs_keys:
                if obs_key in observations:
                    image = observations[obs_key]
                    image_file_name = (
                        f"{frame_num}_agent_0_rgbFetchRobot_{obs_key}.png"
                    )
                    image_file_path = os.path.join(
                        episode_dir, image_file_name
                    )
                    save_image(image, image_file_path)
                # print("key:",key)
            # print(
            #     "observations['arm_workspace_points']:",
            #     observations["arm_workspace_points"],
            # )
            try:
                metadata.append(
                    {
                        "episode_id": episode.episode_id,
                        "step": frame_num,
                        "green_points": observations["arm_workspace_points"],
                        # "detected_objects": observations["objectgoal"],
                    }
                )
            except:
                continue
        env._episode_over = True

    # Save metadata to JSON
    metadata_file_path = os.path.join(output_dir, "metadata_greenpoint.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
def generate_scene_graph(   #use for debug and test scene graph
    dataset_path= None,
    config_path="benchmark/single_agent/zxz_fetch.yaml",
    gpu_id = None,
    bbox_min=  18000,
    bbox_max = 25000,
    dist_to_target=6.0,
    max_trials = 100,
    max_images= 8,
    random_min_bbox = 400,
    min_dis = 3,
    obs_keys=["head_rgb"],
    output_dir = 'data/test_sc_generate_strategy/rearrange/hssd'
    ):
    override = []
    config = get_hssd_single_agent_config(config_path)
    # config['habitat']['simulator']['habitat_sim_v0']['gpu_device_id'] = gpu_id
    # config['habitat']['simulator']['dataset']['data_path'] = dataset_path
    env = Env(config=config)
    dataset = env._dataset
    metadata = []
    for episode in dataset.episodes:
        # reset automatically calls next episode
        # print("episode_info:",episode)
        env.reset()
        sim: RearrangeSim = env.sim
        # get the largest island index
        largest_island_idx = get_largest_island_index(
            sim.pathfinder, sim, allow_outdoor=True
        )
        graph_positions = []
        graph_orientations = []
        graph_annotations = []
        # Get scene objects ids
        scene_obj_ids = sim.scene_obj_ids

        # Get target objects' information
        object_ids, object_handles, start_positions, goal_positions = (
            get_target_objects_info(sim)
        )
        # Sample navigable points near the target objects start positions and goal positions
        # And calculate the orientation of the agent facing towards the position
        bbox_range_min = bbox_min
        bbox_range_max = bbox_max
        for idx, (start_pos, goal_pos) in enumerate(
            zip(start_positions, goal_positions)
        ):
            # Sample navigable point near the start and goal positions
            n_trial = 0
            radius = dist_to_target
            while n_trial < max_trials:  # find start point
                start_navigable_point = (
                    sim.pathfinder.get_random_navigable_point_near(
                        circle_center=start_pos,
                        radius=radius,
                        island_index=largest_island_idx,
                    )
                )
                start_orientation = calculate_orientation_xz_plane(
                    start_navigable_point, start_pos
                )
                set_articulated_agent_base_state(
                    sim, start_navigable_point, start_orientation, agent_id=0
                )
                observations = env.step({"action": (), "action_args": {}})
                env._episode_over = False
                _, _, w, h = observations["rec_bounding_box"][0]
                if not np.isnan(start_navigable_point[0]) and inrange(
                    w * h, bbox_range_min, bbox_range_max
                ):
                    break
                # else increase the radius limit and try again
                else:
                    radius += 0.1
                    n_trial += 1
            # print("trail_time_rec:",n_trial)
            n_trial = 0
            radius = dist_to_target
            while n_trial < max_trials:  # find goal point
                goal_navigable_point = (
                    sim.pathfinder.get_random_navigable_point_near(
                        circle_center=goal_pos,
                        radius=radius,
                        island_index=largest_island_idx,
                    )
                )
                goal_orientation = calculate_orientation_xz_plane(
                    goal_navigable_point, goal_pos
                )
                set_articulated_agent_base_state(
                    sim, goal_navigable_point, goal_orientation, agent_id=0
                )
                observations = env.step({"action": (), "action_args": {}})
                env._episode_over = False
                _, _, w, h = observations["target_bounding_box"][0]
                if not np.isnan(start_navigable_point[0]) and inrange(
                    w * h, bbox_range_min, bbox_range_max
                ):
                    break
                # else increase the radius limit and try again
                else:
                    radius += 0.1
                    n_trial += 1
            # print("trail_time_target:",n_trial)
            graph_positions.extend(
                [start_navigable_point, goal_navigable_point]
            )
            graph_orientations.extend([start_orientation, goal_orientation])
            graph_annotations.extend(["start_rec", "goal_rec"])
        # print("graph_positions:",graph_positions)
        # print("graph_orientations:",graph_orientations)

        # Sample navigable points if images are less than the maximum number of images
        if len(graph_positions) < max_images:
            i = 0
            min_bbox = random_min_bbox
            while max_images > len(graph_positions):
                # Sample a random navigable point in the scene
                point = sim.pathfinder.get_random_navigable_point(
                    island_index=largest_island_idx
                )
                orientation = random_quaternion_xz_plane()
                min_dis_flag = 0
                min_point_dis = min_dis
                for item in graph_positions[:2]:
                    x1 = point[0]
                    y1 = point[2]
                    x2 = item[0]
                    y2 = item[2]
                    # print("dis:",int(np.sqrt(((x1-x2) ** 2)+((y1-y2) ** 2))))
                    if (
                        int(np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)))
                        < min_point_dis
                    ):
                        min_dis_flag = 1
                        break
                if min_dis_flag == 0:
                    set_articulated_agent_base_state(
                        sim, point, orientation, agent_id=0
                    )
                    observations = env.step({"action": (), "action_args": {}})
                    env._episode_over = False
                    _, _, w_tar, h_tar = observations["target_bounding_box"][0]
                    _, _, w_rec, h_rec = observations["rec_bounding_box"][0]
                    if w_tar * h_tar < min_bbox and w_rec * h_rec < min_bbox:
                        i += 1
                        graph_positions.append(point)
                        graph_orientations.append(orientation)
                        graph_annotations.append("random")
        # print("graph_positions:",graph_positions)
        # print("graph_orientations:",graph_orientations)
        # print("graph_annotations:",graph_annotations)
        # Create a folder for this episode
        episode_dir = os.path.join(output_dir, f"episode_{episode.episode_id}")
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        # print("graph_positions:",graph_positions)
        # print("graph_orientations:",graph_orientations)
        for idx, (position, orientation) in enumerate(
            zip(graph_positions, graph_orientations)
        ):
            # observations = env.sim.get_observations_at(position=position, rotation=orientation)
            set_articulated_agent_base_state(
                sim, position, orientation, agent_id=0
            )
            observations = env.step({"action": (), "action_args": {}})

            # Force the episode to be active
            env._episode_over = False

            obs_file_list = []
            # for key in observations.keys():
            #     print(key)
            for obs_key in obs_keys:
                if obs_key in observations:
                    image = observations[obs_key]
                    image_file_name = (
                        f"episode_{episode.episode_id}_{obs_key}_{idx}.png"
                    )
                    image_file_path = os.path.join(
                        episode_dir, image_file_name
                    )
                    save_image(image, image_file_path)
                    obs_file_list.append(image_file_name)
            # for bbox in args.bbox:
            #     if bbox in observations:
            #         print(f"{bbox}:{observations[bbox]}")

            now_loc = observations["localization_sensor"]
            metadata.append(
                {
                    "episode_id": episode.episode_id,
                    "obs_files": obs_file_list,
                    "position": now_loc,
                    "rotation": orientation,
                    # "detected_objects": observations["objectgoal"],
                }
            )

        env._episode_over = True

    # Save metadata to JSON
    metadata_file_path = os.path.join(output_dir, "metadata.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)


def parse_args_new():
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark/single_agent/zxz_fetch_sample.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="TESTSET_dataset_hssd_13scene_3733/102816786/data_30.json.gz",
    )
    parser.add_argument("--obs_keys", nargs="+", default=["head_rgb"])
    parser.add_argument("--dist_to_target", type=float, default=6.0)
    parser.add_argument("--max_trials", type=int, default=60)
    parser.add_argument("--max_images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox_range_min", type=int, default=2000)
    parser.add_argument("--bbox_range_max", type=int, default=4500)
    parser.add_argument("--min_point_dis", type=float, default=3.2)
    parser.add_argument("--random_min_bbox", type=int, default=200)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--anti_position_path", type=str, default="")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument(
        "--meta_json_path", type=str, default="37/scene_graph_info.json"
    )
    parser.add_argument(
        "--generate_type", type=str, default="single_action_training"
    )
    parser.add_argument(
        "--sample_info",
        type=str,
        default=json.dumps(
            [
                {
                    "episode_id": 3,
                    "sample_frame": [
                        [1, 0],
                        [88, 0],
                        [95, 0],
                        [96, 0],
                        [97, 0],
                        [259, 0],
                        [328, 0],
                        [355, 0],
                        [356, 0],
                        [357, 0],
                    ],
                },
                {
                    "episode_id": 2,
                    "sample_frame": [
                        [1, 0],
                        [100, 0],
                        [126, 0],
                        [127, 0],
                        [128, 0],
                        [240, 0],
                        [330, 0],
                        [358, 0],
                        [359, 0],
                        [360, 0],
                    ],
                },
                {
                    "episode_id": 1,
                    "sample_frame": [
                        [1, 0],
                        [177, 0],
                        [187, 0],
                        [188, 0],
                        [189, 0],
                        [354, 0],
                        [569, 0],
                        [585, 0],
                        [586, 0],
                        [587, 0],
                    ],
                },
            ]
        ),
    )
    # parser.add_argument("--output_dir", type=str, default="data/sparse_slam/rearrange/mp3d")
    args = parser.parse_args()

    # Create output directory if it does not exist
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # Set seed
    return args


if __name__ == "__main__":
    args = parse_args_new()
    if args.debug:
        generate_scene_graph()
    elif args.generate_type == "scene_graph":
        generate_scene_graph_from_store_dict(args)
    elif args.generate_type == "single_action_training":
        generate_scene_graph_for_single_task(args)
    else:
        generate_episode_image_from_store_dict(args)

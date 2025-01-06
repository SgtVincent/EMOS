from collections import defaultdict, deque
import magnum as mn
import cv2
import numpy as np
from gym import spaces
import habitat_sim
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import re
from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    CollisionDetails,
    UsesArticulatedAgentInterface,
    batch_transform_point,
    get_angle_to_pos,
    get_camera_object_angle,
    get_camera_transform,
    rearrange_logger,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat_sim.physics import MotionType


@registry.register_sensor
class CameraInfoSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "camera_info"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.camera_name = config.get("depth_sensor_name", "head_rgb")

    def _get_uuid(self, *args, **kwargs):
        return CameraInfoSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Dict({
            "projection_matrix": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32
            ),
            "camera_matrix": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32
            ),
            "fov": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

    def get_observation(self, observations, *args, **kwargs):
        camera_info = {}
        if self.agent_id is not None:
            camera_name = f"agent_{self.agent_id}_{self.camera_name}"
        else:
            camera_name = self.camera_name
        # print("sensor_info:",)
        hfov = (
            float(self._sim._sensors[camera_name]._sensor_object.hfov)
            * np.pi
            / 180.0
        )
        camera = self._sim._sensors[camera_name]._sensor_object.render_camera
        camera_info[camera_name] = {
            "projection_matrix": np.array(camera.projection_matrix),
            "camera_matrix": np.array(camera.camera_matrix),
            "fov": hfov,
        }
        return camera_info


@registry.register_sensor
class ObjBBoxSenor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "obj_bounding_box"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        self.height = config.height
        self.width = config.width
        self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.3)
        self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        self.n = 1

        super().__init__(config=config)

        # self._debug_tf = config.get("debug_tf", False)
        self._debug_tf = True
        if self._debug_tf:
            self.pcl_o3d_list = []
            self._debug_save_counter = 0

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return ObjBBoxSenor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float64).max,
            shape=(self.n, 4),
            dtype=np.float64,
        )

    def get_observation(self, observations, *args, **kwargs):
        """Get the RGB image with reachable and unreachable points marked"""

        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            rgb_obs = observations[
                f"agent_{self.agent_id}_{self.rgb_sensor_name}"
            ]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            semantic_camera_name = f"head_semantic"

        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)

        """add semantic information"""
        ep_objects = []
        # for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
        # for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])
        for key, val in self._sim.ep_info.info["object_labels"].items():
            ep_objects.append(key)
        # only add obj semantic
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[semantic_camera_name].squeeze()
        mask = np.isin(
            semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_box = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                rect = cv2.minAreaRect(contour)
                bound = cv2.boxPoints(rect)
                x, y, w, h = cv2.boundingRect(contour)
                # x,y,w,h = bound
                bounding_box.append((x, y, w, h))

        if bounding_box:
            self.n = len(bounding_box)
            return bounding_box
        else:
            return np.array([[-1, -1, -1, -1]])


@registry.register_sensor
class RecBBoxSenor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "rec_bounding_box"

    def __init__(self, sim, task, config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        self.height = config.height
        self.width = config.width
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.3)
        self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        self.n = 1

        super().__init__(config=config)

        # self._debug_tf = config.get("debug_tf", False)
        self._debug_tf = True
        if self._debug_tf:
            self.pcl_o3d_list = []
            self._debug_save_counter = 0

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return RecBBoxSenor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float64).max,
            shape=(self.n, 4),
            dtype=np.float64,
        )

    def get_observation(self, observations, *args, **kwargs):
        """Get the RGB image with reachable and unreachable points marked"""
        ep_objects = []
        is_multi_agent = False
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            rgb_obs = observations[
                f"agent_{self.agent_id}_{self.rgb_sensor_name}"
            ]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
            is_multi_agent = True
        else:
            depth_obs = observations[self.depth_sensor_name]
            rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            semantic_camera_name = f"head_semantic"
            for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
                ep_objects.append(self._sim.ep_info.target_receptacles[0][i])

            # now the scene_graph_generate is simply have one target rec and one goal rec/
        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)
        if is_multi_agent:
            nav_to_obj_number = -1
            try:
                nav_to_target_idx = kwargs["action"]["action_args"][
                    f"agent_{self.agent_id}_oracle_nav_action"
                ]
                nav_to_target_idx = int(nav_to_target_idx[0]) - 1
                nav_to_obj = self._entities[nav_to_target_idx]
                nav_to_obj = str(nav_to_obj)
                if "any_targets" in nav_to_obj:
                    match = re.search(r"\|(\d+)-", str(nav_to_obj))
                    if match:
                        nav_to_obj_number = int(match.group(1))
                else:
                    return np.array([[-1, -1, -1, -1]])
            except Exception as e:
                print("e:", e, flush=True)
                return np.array([[-1, -1, -1, -1]])

            # print("info::",self._sim.ep_info.target_receptacles)
            # print("nav_to_obj_number:",nav_to_obj_number,flush = True)
            if nav_to_obj_number != -1:
                for i in range(
                    len(
                        self._sim.ep_info.target_receptacles[nav_to_obj_number]
                    )
                    - 1
                ):
                    ep_objects.append(
                        self._sim.ep_info.target_receptacles[
                            nav_to_obj_number
                        ][i]
                    )
            else:
                return np.array([[-1, -1, -1, -1]])
        # for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
        # for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])
        # for key, val in self._sim.ep_info.info['object_labels'].items():
        #     ep_objects.append(key)
        # only add obj semantic
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[semantic_camera_name].squeeze()
        mask = np.isin(
            semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_box = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                rect = cv2.minAreaRect(contour)
                bound = cv2.boxPoints(rect)
                x, y, w, h = cv2.boundingRect(contour)
                # x,y,w,h = bound
                bounding_box.append((x, y, w, h))

        if bounding_box:
            self.n = len(bounding_box)
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")
            for x, y, w, h in bounding_box:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
            union_x = int(min_x)
            union_y = int(min_y)
            union_w = int(max_x - min_x)
            union_h = int(max_y - min_y)
            self.n = len(bounding_box)
            return np.array([[union_x, union_y, union_w, union_h]])
            return bounding_box
        else:
            return np.array([[-1, -1, -1, -1]])


@registry.register_sensor
class DepthSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "depth_obs"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        # self.height = config.height
        # self.width = config.width
        # self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        # self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.3)
        # self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        # self.n = 1
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return DepthSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float64).max,
            shape=(256, 256, 1),
            dtype=np.float64,
        )

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            # semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            # rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            # semantic_camera_name = f"head_semantic"

        # rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)
        return depth_obs


@registry.register_sensor
class DepthRotSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "depth_rot"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        # self.height = config.height
        # self.width = config.width
        # self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        # self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.3)
        # self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        # self.n = 1
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return DepthRotSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float64).max,
            shape=(3, 3),
            dtype=np.float64,
        )

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            # semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            # rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            # semantic_camera_name = f"head_semantic"

        # rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_camera = self._sim._sensors[
            depth_camera_name
        ]._sensor_object.render_camera
        depth_rotation = np.array(depth_camera.camera_matrix.rotation())
        # print("test_rot:",depth_rotation,flush = True)
        return depth_rotation
@registry.register_sensor
class DepthTransSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "depth_trans"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        # self.height = config.height
        # self.width = config.width
        # self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        # self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.3)
        # self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        # self.n = 1
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return DepthTransSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float64).max,
            shape=(1, 3),
            dtype=np.float64,
        )

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            # semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            # rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            # semantic_camera_name = f"head_semantic"

        # rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_camera = self._sim._sensors[
            depth_camera_name
        ]._sensor_object.render_camera
        depth_translation = np.array(depth_camera.camera_matrix.translation)
        # print("test_trans:",depth_translation,flush = True)
        return depth_translation
@registry.register_sensor
class DepthProjectionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "depth_project"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")

        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return DepthProjectionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32)

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
        else:
            depth_obs = observations[self.depth_sensor_name]
            depth_camera_name = self.depth_sensor_name
        depth_camera = self._sim._sensors[
            depth_camera_name
        ]._sensor_object.render_camera
        depth_projection_ = np.array(depth_camera.projection_matrix)
        depth_projection = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                depth_projection[i][j] = depth_projection_[i][j]
        return depth_projection
@registry.register_sensor
class CameraMatrixSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "camera_matrix"
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")

        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return CameraMatrixSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32)

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
        else:
            depth_obs = observations[self.depth_sensor_name]
            depth_camera_name = self.depth_sensor_name
        depth_camera = self._sim._sensors[
            depth_camera_name
        ]._sensor_object.render_camera
        camera_matrix_ = depth_camera.camera_matrix
        camera_matrix = [[0.0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                camera_matrix[i][j] = camera_matrix_[i][j]
        return camera_matrix
@registry.register_sensor
class TargetBBoxSenor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "target_bounding_box"

    def __init__(self, sim, task, config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        self.height = config.height
        self.width = config.width
        self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.3)
        self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        self.n = 1
        self._task = task
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self.store_bbox_num = None
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return TargetBBoxSenor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float64).max,
            shape=(self.n, 4),
            dtype=np.float64,
        )

    def get_observation(self, observations, *args, **kwargs):
        ep_objects = []
        is_multi_agent = False
        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            rgb_obs = observations[
                f"agent_{self.agent_id}_{self.rgb_sensor_name}"
            ]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
            is_multi_agent = True
        else:
            depth_obs = observations[self.depth_sensor_name]
            rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            semantic_camera_name = f"head_semantic"
            for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
                ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])

        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)

        """add semantic information"""
        if is_multi_agent:
            if self.store_bbox_num:
                nav_to_obj_number = self.store_bbox_num
            else:
                try:
                    nav_to_target_idx = kwargs["action"]["action_args"][
                        f"agent_{self.agent_id}_oracle_nav_action"
                    ]
                    nav_to_target_idx = int(nav_to_target_idx[0]) - 1
                    nav_to_obj = self._entities[nav_to_target_idx]
                    nav_to_obj = str(nav_to_obj)

                    if "TARGET_any_targets" in nav_to_obj:
                        match = re.search(r"\|(\d+)-", str(nav_to_obj))
                        if match:
                            nav_to_obj_number = int(match.group(1))
                    else:
                        return np.array([[-1, -1, -1, -1]])
                except Exception as e:
                    # print("e:",e,flush = True)
                    return np.array([[-1, -1, -1, -1]])
            # for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
            #     ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
            for i in range(
                len(self._sim.ep_info.goal_receptacles[nav_to_obj_number]) - 1
            ):
                ep_objects.append(
                    self._sim.ep_info.goal_receptacles[nav_to_obj_number][i]
                )
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[semantic_camera_name].squeeze()
        mask = np.isin(
            semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_box = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                rect = cv2.minAreaRect(contour)
                bound = cv2.boxPoints(rect)
                x, y, w, h = cv2.boundingRect(contour)
                # x,y,w,h = bound
                bounding_box.append((x, y, w, h))
        if bounding_box:
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")
            for x, y, w, h in bounding_box:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
            union_x = int(min_x)
            union_y = int(min_y)
            union_w = int(max_x - min_x)
            union_h = int(max_y - min_y)
            self.n = len(bounding_box)
            return np.array([[union_x, union_y, union_w, union_h]])
        else:
            return np.array([[-1, -1, -1, -1]])

@registry.register_sensor
class ArmWorkspaceRGBSensor(UsesArticulatedAgentInterface, Sensor):
    """ Sensor to visualize the reachable workspace of an articulated arm """
    cls_uuid: str = "arm_workspace_rgb"

    def __init__(self, sim, task,config, *args, **kwargs):
        self._sim = sim
        # self.agent_idx = config.agent_idx
        self._task = task
        self.pixel_threshold = config.pixel_threshold
        self.height = config.height
        self.width = config.width
        self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
        self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.1)
        self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        self.single_agent_eval_option = config.get("single_agent_eval_option",False)
        self.pre_wait = False
        super().__init__(config=config)
                
        self._debug_tf = config.get("debug_tf", False)
        if self._debug_tf:
            self.pcl_o3d_list = []
            self._debug_save_counter = 0
            
    def _get_uuid(self, *args, **kwargs):
        # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
        return ArmWorkspaceRGBSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

    def _2d_to_3d(self, depth_name, depth_obs):
        # get the scene render camera and sensor object
        depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

        hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
        W, H = depth_camera.viewport[0], depth_camera.viewport[1]
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])

        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
        depth = depth_obs.reshape(1, W, W)
        xs = xs.reshape(1, W, W)
        ys = ys.reshape(1, W, W)
        # print(f"hfov:{hfov},W:{W},H:{H},K:{K},")
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c = np.matmul(np.linalg.inv(K), xys)

        depth_rotation = np.array(depth_camera.camera_matrix.rotation())
        depth_translation = np.array(depth_camera.camera_matrix.translation)

        # get camera-to-world transformation
        T_world_camera = np.eye(4)
        T_world_camera[0:3, 0:3] = depth_rotation
        T_world_camera[0:3, 3] = depth_translation

        T_camera_world = np.linalg.inv(T_world_camera)
        points_world = np.matmul(T_camera_world, xy_c)

        # get non_homogeneous points in world space
        points_world = points_world[:3, :] / points_world[3, :]
        # reshape to the scale of the image
        # points_world = points_world.reshape((3, H, W)).transpose(1, 2, 0)
        points_world = points_world.transpose(1, 0)

        return points_world

    def _3d_to_2d(self, sensor_name, point_3d):
        # get the scene render camera and sensor object
        render_camera = self._sim._sensors[sensor_name]._sensor_object.render_camera
        W, H = render_camera.viewport[0], render_camera.viewport[1]

        # use the camera and projection matrices to transform the point onto the near plane
        projected_point_3d = render_camera.projection_matrix.transform_point(
            render_camera.camera_matrix.transform_point(point_3d)
        )
        # convert the 3D near plane point to integer pixel space
        point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
        point_2d = point_2d / render_camera.projection_size()[0]
        point_2d += mn.Vector2(0.5)
        point_2d *= render_camera.viewport
        # print(f"info:view:{render_camera.viewport}/size:{render_camera.projection_size()[0]}/")
        out_bound = 10
        point_2d = np.nan_to_num(point_2d, nan=W+out_bound, posinf=W+out_bound, neginf=-out_bound)
        return point_2d.astype(int)

    def voxel_grid_filter(self, points, voxel_size):
        voxel_indices = np.floor(points / voxel_size).astype(int)

        voxel_dict = {}
        for i, voxel_index in enumerate(voxel_indices):
            voxel_key = tuple(voxel_index)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []
            voxel_dict[voxel_key].append(points[i])

        downsampled_points = []
        for voxel_key in voxel_dict:
            voxel_points = np.array(voxel_dict[voxel_key])
            mean_point = voxel_points.mean(axis=0)
            downsampled_points.append(mean_point)

        return np.array(downsampled_points)

    def _is_reachable(self, cur_articulated_agent, ik_helper, point, thresh=0.05):
        cur_base_pos, cur_base_orn = ik_helper.get_base_state()
        base_transformation = cur_articulated_agent.base_transformation
    
        orn_quaternion = mn.Quaternion.from_matrix(base_transformation.rotation())
        base_pos = base_transformation.translation
        base_orn = list(orn_quaternion.vector)
        base_orn.append(orn_quaternion.scalar)
        ik_helper.set_base_state(base_pos, base_orn)
        # point_base = cur_articulated_agent.base_transformation.inverted().transform_vector(point)
        point_base = point

        cur_joint_pos = cur_articulated_agent.arm_joint_pos

        des_joint_pos = ik_helper.calc_ik(point_base)
        # temporarily set arm joint position
        if cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
            cur_articulated_agent.arm_motor_pos = des_joint_pos
        if cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
            cur_articulated_agent.arm_joint_pos = des_joint_pos
            cur_articulated_agent.fix_joint_values = des_joint_pos

        des_ee_pos = ik_helper.calc_fk(des_joint_pos)

        # revert arm joint position
        if cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
            cur_articulated_agent.arm_motor_pos = cur_joint_pos
        if cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
            cur_articulated_agent.arm_joint_pos = cur_joint_pos
            cur_articulated_agent.fix_joint_values = cur_joint_pos

        ik_helper.set_base_state(cur_base_pos, cur_base_orn)

        return np.linalg.norm(np.array(point_base) - np.array(des_ee_pos)) < thresh

    def get_observation(self, observations, *args, **kwargs):
        """ Get the RGB image with reachable and unreachable points marked """
        if self.agent_id is not None:
            depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
            rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
            semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            semantic_camera_name = f"head_semantic"
        if self.single_agent_eval_option:
            wait_flag = self._task.actions["wait"].skill_done
        # assert wait_flag is bool, "wait_flag is not bool"
            if not (not self.pre_wait and wait_flag):
                self.pre_wait = wait_flag
                return np.array([0],dtype=np.int8)
            self.pre_wait = wait_flag
        
        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)

        """add semantic information"""
        ep_objects = []
        nav_to_obj_number = -1
        try:
            nav_to_target_idx = kwargs['action']['action_args'][
                f"agent_{self.agent_id}_oracle_nav_action"
            ]    
            nav_to_target_idx = int(nav_to_target_idx[0]) - 1
            nav_to_obj = self._entities[nav_to_target_idx]
            match = re.search(r'\|(\d+)-', str(nav_to_obj))
            if match:
                nav_to_obj_number = int(match.group(1))
        except:
            # print(f"Error loading navigation action: {e}")
            pass
        
        # print("info::",self._sim.ep_info.target_receptacles)
        # print("nav_to_obj_number:",nav_to_obj_number,flush = True)
        if nav_to_obj_number!= -1:
            for i in range(len(self._sim.ep_info.target_receptacles[nav_to_obj_number]) - 1):
                ep_objects.append(self._sim.ep_info.target_receptacles[nav_to_obj_number][i])
            for i in range(len(self._sim.ep_info.goal_receptacles[nav_to_obj_number]) - 1):
                ep_objects.append(self._sim.ep_info.goal_receptacles[nav_to_obj_number][i])
        else:
            for item in range(len(self._sim.ep_info.target_receptacles)):
                for i in range(len(self._sim.ep_info.target_receptacles[item]) - 1):
                    ep_objects.append(self._sim.ep_info.target_receptacles[item][i])
        
            for i in range(len(self._sim.ep_info.goal_receptacles[item]) - 1):
                ep_objects.append(self._sim.ep_info.goal_receptacles[item][i])
            # for key, val in self._sim.ep_info.info['object_labels'].items():
            #     ep_objects.append(key)
        # print("ep_objects:",ep_objects,flush=True)
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            # print("handle:",handle)
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start

        semantic_obs = observations[semantic_camera_name].squeeze()

        mask = np.isin(semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset).astype(np.uint8)

        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_box = []
        colored_mask = np.zeros_like(rgb_obs)
        colored_mask[mask == 1] = [0, 0, 255]
        # rgb_obs = cv2.addWeighted(rgb_obs, 0.5, colored_mask, 0.5, 0)
        for contour in contours:
            if cv2.contourArea(contour) > 0:  # 过滤掉面积为0的轮廓
                x, y, w, h = cv2.boundingRect(contour)
                bounding_box.append((x, y, w, h))
                # 可选：在原始图像上绘制边界框
                # cv2.rectangle(rgb_obs, (x, y), (x + w, y + h), (255, 0, 0), 1)  #是否标boundingbox
        # for obj_id in objects_info.keys():
        #     positions = np.where(semantic_obs == obj_id + obj_id_offset)
        #     if positions[0].size > 0:
        #         center_x = int(np.mean(positions[1]))
        #         center_y = int(np.mean(positions[0]))
        #         cv2.putText(rgb_obs, objects_info[obj_id], (center_x, center_y),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Reproject depth pixels to 3D points
        points_world = self._2d_to_3d(depth_camera_name, depth_obs)
        # downsample the 3d-points
        down_sampled_points = self.voxel_grid_filter(points_world, self.down_sample_voxel_size)

        # Check reachability and color points
        colors = []

        articulated_agent_mgr = self._sim.agents_mgr._all_agent_data[self.agent_id if self.agent_id is not None else 0]
        cur_articulated_agent = articulated_agent_mgr.articulated_agent
        ik_helper = articulated_agent_mgr.ik_helper

        for point in down_sampled_points:
            reachable = self._is_reachable(cur_articulated_agent, ik_helper, point)
            # Green if reachable, red if not
            colors.append([0, 255, 0] if reachable else [255, 0, 0])

        pixel_coords = []
        if self._debug_tf and 'obj_pos' in kwargs:
            if kwargs['obj_pos'] is not None:
                pixel_coord = self._3d_to_2d(depth_camera_name, np.array(list(kwargs['obj_pos'])))
                if np.any(np.isnan(pixel_coord)) or np.any(np.isinf(pixel_coord)):
                    print("obj_pos is invalid")
                else:
                    down_sampled_points = np.array([list(kwargs['obj_pos'])])
                    colors = [[0, 255, 0]]

        # Project the points to the image and color the pixels with circles
        for point in down_sampled_points:
            pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

        for pixel_coord, color in zip(pixel_coords, colors):
            x, y = pixel_coord.astype(int)
            if color == [0, 255, 0]:
                cv2.circle(rgb_obs, (x, y), 2, color, -1)
                    # the arm now is out of robot's head_rgb so that don't need mask
        # mask_img = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
        # bgr = mask_img[:, :, :3]
        # alpha = mask_img[:, :, 3]
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # mask_img_rga = cv2.merge((rgb, alpha))
        # assert rgb_obs.shape[:2] == mask_img_rga.shape[:2]
        # alpha_channel = mask_img_rga[:, :, 3] / 255.0
        # rgb_obs
        # for c in range(0, 3):
        #     rgb_obs[:, :, c] = (alpha_channel * mask_img_rga[:, :, c] +
        #                         (1 - alpha_channel) * rgb_obs[:, :, c])
        return rgb_obs


@registry.register_sensor
class ArmWorkspacePointsSensor(ArmWorkspaceRGBSensor):
    """Sensor to visualize the reachable workspace of an articulated arm"""

    cls_uuid: str = "arm_workspace_points"

    def _get_uuid(self, *args, **kwargs):
        return ArmWorkspacePointsSensor.cls_uuid

    def get_observation(self, observations, *args, **kwargs):
        """Get the RGB image with reachable and unreachable points marked"""

        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
        else:
            depth_obs = observations[self.depth_sensor_name]
            depth_camera_name = self.depth_sensor_name

        depth_obs = np.ascontiguousarray(depth_obs)
        points_world = self._2d_to_3d(depth_camera_name, depth_obs)
        # downsample the 3d-points
        down_sampled_points = self.voxel_grid_filter(
            points_world, self.down_sample_voxel_size
        )

        # Project the points to the image and color the pixels with circles
        pixel_coords = []
        for point in down_sampled_points:
            pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

        flat_points = []
        space_points = []
        ik_helper = self._sim.get_agent_data(self.agent_id).ik_helper
        cur_articulated_agent = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent
        for idx, (point_3d, point_2d) in enumerate(
            zip(down_sampled_points, pixel_coords)
        ):
            reachable = self._is_reachable(
                cur_articulated_agent, ik_helper, point_3d
            )
            if reachable:
                point = np.concatenate((point_2d, [idx]), axis=0)
                flat_points.append(point)
                space_points.append(point_3d)
        return np.array(flat_points)


@registry.register_sensor
class ArmWorkspaceRGBThirdSensor(ArmWorkspaceRGBSensor):
    """Sensor to visualize the reachable workspace of an articulated arm"""

    cls_uuid: str = "arm_workspace_rgb_third"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(sim=sim, config=config, *args, **kwargs)

        self.rgb_sensor_name = config.get("rgb_sensor_name", "third_rgb")
        self.depth_sensor_name = config.get("depth_sensor_name", "third_depth")


@registry.register_sensor
class ObjectMasksSensor(UsesArticulatedAgentInterface, Sensor):
    """Sensor to mask the objects that are interested"""

    cls_uuid: str = "object_masks"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return ObjectMasksSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, config, **kwargs):
        # The observation space is flexible, should not be used as gym input
        return spaces.Box(
            low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64
        )

    def get_observation(self, observations, *args, **kwargs):
        """Get the detected objects from the semantic sensor data"""
        objects_info = {}
        # print("sim",self._sim)
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            objects_info[obj.object_id] = handle
        """articulated object not in used for now"""
        # aom = self._sim.get_articulated_object_manager()
        # for i, handle in enumerate(aom.get_object_handles()):
        #     obj = aom.get_object_by_handle(handle)
        #     objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[
            f"agent_{self.agent_id}_head_semantic"
        ].squeeze()
        rgb_obs = observations[f"agent_{self.agent_id}_head_rgb"]

        mask = np.isin(
            semantic_obs, np.array(self._sim.scene_obj_ids) + obj_id_offset
        ).astype(np.uint8)
        colored_mask = np.zeros_like(rgb_obs)
        colored_mask[mask == 1] = [0, 0, 255]
        masked_rgb = cv2.addWeighted(rgb_obs, 0.5, colored_mask, 0.5, 0)

        for obj_id in np.array(self._sim.scene_obj_ids):
            positions = np.where(semantic_obs == obj_id + obj_id_offset)
            if positions[0].size > 0:
                center_x = int(np.mean(positions[1]))
                center_y = int(np.mean(positions[0]))
                cv2.putText(
                    masked_rgb,
                    objects_info[obj_id],
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        return masked_rgb


@registry.register_sensor
class NavWorkspaceRGBSensor(
    ArmWorkspaceRGBSensor, UsesArticulatedAgentInterface, Sensor
):
    """Sensor to visualize the reachable workspace of an articulated arm"""

    cls_uuid: str = "nav_workspace_rgb"

    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(sim=sim, config=config, task=task)

    def _get_uuid(self, *args, **kwargs):
        return NavWorkspaceRGBSensor.cls_uuid

    def _is_reachable(self, point):
        agent_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)

        return found_path

    def _get_plane_points(self, points):
        X = points[:, [0, 2]]
        y = points[:, 1]

        ransac = make_pipeline(
            PolynomialFeatures(1), RANSACRegressor(LinearRegression())
        )
        ransac.fit(X, y)
        inlier_mask = ransac.named_steps["ransacregressor"].inlier_mask_
        plane_points = points[inlier_mask]
        return plane_points

    def get_observation(self, observations, *args, **kwargs):
        """Get the RGB image with reachable and unreachable points marked"""

        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            rgb_obs = observations[
                f"agent_{self.agent_id}_{self.rgb_sensor_name}"
            ]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
            semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            semantic_camera_name = "head_semantic"

        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)

        """add semantic information"""
        ep_objects = []
        for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
            ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
        for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
            ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])
        for key, val in self._sim.ep_info.info["object_labels"].items():
            ep_objects.append(key)

        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start

        semantic_obs = observations[semantic_camera_name].squeeze()

        mask = np.isin(
            semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset
        ).astype(np.uint8)
        colored_mask = np.zeros_like(rgb_obs)
        colored_mask[mask == 1] = [0, 0, 255]
        rgb_obs = cv2.addWeighted(rgb_obs, 0.5, colored_mask, 0.5, 0)

        for obj_id in objects_info.keys():
            positions = np.where(semantic_obs == obj_id + obj_id_offset)
            if positions[0].size > 0:
                center_x = int(np.mean(positions[1]))
                center_y = int(np.mean(positions[0]))
                cv2.putText(
                    rgb_obs,
                    objects_info[obj_id],
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        # Reproject depth pixels to 3D points
        points_world = self._2d_to_3d(depth_camera_name, depth_obs)

        """get the ground points using RANSAC"""
        # 过滤接近地面高度的点
        ground_points_mask = (
            np.abs(points_world[:, 1] - points_world[:, 1].min()) < 0.1
        )
        ground_points = points_world[ground_points_mask]
        ground_points = self._get_plane_points(ground_points)

        # downsample the 3d-points
        down_sampled_points = self.voxel_grid_filter(
            ground_points, self.down_sample_voxel_size
        )

        # Check reachability and color points
        colors = []
        for point in down_sampled_points:
            reachable = self._is_reachable(point)
            # Green if reachable, red if not
            colors.append([0, 255, 0] if reachable else [255, 0, 0])

        # Project the points to the image and color the pixels with circles
        pixel_coords = []
        for point in down_sampled_points:
            pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

        for pixel_coord, color in zip(pixel_coords, colors):
            x, y = pixel_coord.astype(int)
            if color == [0, 255, 0]:
                cv2.circle(rgb_obs, (x, y), 2, color, -1)

        return rgb_obs


@registry.register_sensor
class NavWorkspacePointsSensor(NavWorkspaceRGBSensor):
    """Sensor to visualize the reachable workspace of an articulated arm"""

    cls_uuid: str = "nav_workspace_points"

    def _get_uuid(self, *args, **kwargs):
        return NavWorkspacePointsSensor.cls_uuid

    def get_observation(self, observations, *args, **kwargs):
        """Get the RGB image with reachable and unreachable points marked"""

        if self.agent_id is not None:
            depth_obs = observations[
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            ]
            depth_camera_name = (
                f"agent_{self.agent_id}_{self.depth_sensor_name}"
            )
        else:
            depth_obs = observations[self.depth_sensor_name]
            depth_camera_name = self.depth_sensor_name

        depth_obs = np.ascontiguousarray(depth_obs)

        # Reproject depth pixels to 3D points
        points_world = self._2d_to_3d(depth_camera_name, depth_obs)

        """get the ground points using RANSAC"""
        # 过滤接近地面高度的点
        ground_points_mask = (
            np.abs(points_world[:, 1] - points_world[:, 1].min()) < 0.1
        )
        ground_points = points_world[ground_points_mask]
        ground_points = self._get_plane_points(ground_points)

        # downsample the 3d-points
        down_sampled_points = self.voxel_grid_filter(
            ground_points, self.down_sample_voxel_size
        )

        # Project the points to the image and color the pixels with circles
        pixel_coords = []
        for point in down_sampled_points:
            pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

        # Check reachability and color points
        flat_points = []
        space_points = []

        for idx, (point_3d, point_2d) in enumerate(
            zip(down_sampled_points, pixel_coords)
        ):
            reachable = self._is_reachable(point_3d)
            if reachable:
                point = np.concatenate((point_2d, [idx]), axis=0)
                flat_points.append(point)
                space_points.append(point_3d)

        return np.array([space_points, flat_points])

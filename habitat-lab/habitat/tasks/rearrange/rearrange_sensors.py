#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys

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

class MultiObjSensor(PointGoalSensor):
    """
    Abstract parent class for a sensor that specifies the locations of all targets.
    """

    def __init__(self, *args, task, **kwargs):
        self._task = task
        self._sim: RearrangeSim
        super().__init__(*args, task=task, **kwargs)

    def _get_observation_space(self, *args, **kwargs):
        n_targets = self._task.get_n_targets()
        return spaces.Box(
            shape=(n_targets * 3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )


@registry.register_sensor
class TargetCurrentSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    This is the ground truth object position sensor relative to the robot end-effector coordinate frame.
    """

    cls_uuid: str = "obj_goal_pos_sensor"

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        self._sim: RearrangeSim
        T_inv = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .inverted()
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        pos = scene_pos[idxs]
        print(f"{self.agent_id}_info:{idxs}")
        for i in range(pos.shape[0]):
            print(f"position{i}:{pos[i]}")
            pos[i] = T_inv.transform_point(pos[i])

        return pos.reshape(-1)


@registry.register_sensor
class TargetStartSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "obj_start_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()
        pos = self._sim.get_target_objs_start()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


class PositionGpsCompassSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        n_targets = self._task.get_n_targets()
        self._polar_pos = np.zeros(n_targets * 2, dtype=np.float32)
        return spaces.Box(
            shape=(n_targets * 2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_positions(self) -> np.ndarray:
        raise NotImplementedError("Must override _get_positions")

    def get_observation(self, task, *args, **kwargs):
        pos = self._get_positions()
        articulated_agent_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation

        rel_pos = batch_transform_point(
            pos, articulated_agent_T.inverted(), np.float32
        )

        for i, rel_obj_pos in enumerate(rel_pos):
            rho, phi = cartesian_to_polar(rel_obj_pos[0], rel_obj_pos[1])
            self._polar_pos[(i * 2) : (i * 2) + 2] = [rho, -phi]
        # TODO: This is a hack. For some reason _polar_pos in overriden by the other
        # agent.
        return self._polar_pos.copy()


@registry.register_sensor
class TargetStartGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_start_gps_compass"

    def _get_uuid(self, *args, **kwargs):
        return TargetStartGpsCompassSensor.cls_uuid

    def _get_positions(self) -> np.ndarray:
        return self._sim.get_target_objs_start()


@registry.register_sensor
class TargetGoalGpsCompassSensor(PositionGpsCompassSensor):
    cls_uuid: str = "obj_goal_gps_compass"
        # Agent not initialized 
        # self._init_cord = self.cur_articulated_agent.ee_transform().translation

    def _get_uuid(self, *args, **kwargs):
        return TargetGoalGpsCompassSensor.cls_uuid
    def _get_positions(self) -> np.ndarray:

        # idx, pos = self._sim.get_targets()
        idxs, target = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        #print(f"target{target},target_pos{target_pos}")
        # print("test:",idx)
        # print("compass_goal",pos)
        _entities = self._task.pddl_problem.get_ordered_entities_list()
        # print("en:",_entities,flush=True)
        # obj_pos_0 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[0]
        # )
        # obj_pos_1 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[1]
        # )
        # obj_pos_2 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[2]
        # )
        # obj_pos_3 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[3]
        # )
        # obj_pos_4 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[4]
        # )
        # obj_pos_5 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[5]
        # )
        # print("ordered_action",self._task.pddl_problem.get_ordered_actions())
        # print("obj_pos0::",obj_pos_0,flush=True)
        # print("obj_pos1::",obj_pos_1,flush=True)
        # print("obj_pos2::",obj_pos_2,flush=True)
        # print("obj_pos3::",obj_pos_3,flush=True)
        # print("obj_pos4::",obj_pos_4,flush=True)
        # print("obj_pos5::",obj_pos_5,flush=True)
        entity_pos = {element: self._task.pddl_problem.sim_info.get_entity_pos(element)
                      for element in _entities}
        return scene_pos
@registry.register_sensor
class ObjListSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "obj_list_info"

    def __init__(self, sim, task, *args, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjListSensor.cls_uuid

    def _get_observation_space(self, *args, **kwargs):
        low = -np.inf
        high = np.inf
        return spaces.Dict({
            "TARGET_any_targets|0-T:goal_entity_type": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
            "TARGET_any_targets|1-T:goal_entity_type": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
            "any_targets|0-T:movable_entity_type": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
            "any_targets|1-T:movable_entity_type": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
            "robot_0-T:robot_entity_type": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
            "robot_1-T:robot_entity_type": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
        })

    def get_observation(self, observations, episode, *args, **kwargs):
        #print(f"target{target},target_pos{target_pos}")
        # print("test:",idx)
        # print("compass_goal",pos)
        _entities = self._task.pddl_problem.get_ordered_entities_list()
        # print("en:",_entities)
        # obj_pos_0 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[0]
        # )
        # obj_pos_1 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[1]
        # )
        # obj_pos_2 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[2]
        # )
        # obj_pos_3 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[3]
        # )
        # obj_pos_4 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[4]
        # )
        # obj_pos_5 = self._task.pddl_problem.sim_info.get_entity_pos(
        #     _entities[5]
        # )
        # # print("ordered_action",self._task.pddl_problem.get_ordered_actions())
        # print("obj_pos0::",obj_pos_0)
        # print("obj_pos1::",obj_pos_1)
        # print("obj_pos2::",obj_pos_2)
        # print("obj_pos3::",obj_pos_3)
        # print("obj_pos4::",obj_pos_4)
        # print("obj_pos5::",obj_pos_5)
        entity_pos = {element: np.array(self._task.pddl_problem.sim_info.get_entity_pos(element),dtype = np.float32)
                      for element in _entities}
        return entity_pos

@registry.register_sensor
class AbsTargetStartSensor(MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "abs_obj_start_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        pos = self._sim.get_target_objs_start()
        return pos.reshape(-1)


@registry.register_sensor
class GoalSensor(UsesArticulatedAgentInterface, MultiObjSensor):
    """
    Relative to the end effector
    """

    cls_uuid: str = "obj_goal_sensor"

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.ee_transform()
        T_inv = global_T.inverted()

        _, pos = self._sim.get_targets()
        return batch_transform_point(pos, T_inv, np.float32).reshape(-1)


@registry.register_sensor
class AbsGoalSensor(MultiObjSensor):
    cls_uuid: str = "abs_obj_goal_sensor"

    def get_observation(self, *args, observations, episode, **kwargs):
        _, pos = self._sim.get_targets()
        return pos.reshape(-1)


@registry.register_sensor
class JointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._arm_joint_mask = config.arm_joint_mask

    def _get_uuid(self, *args, **kwargs):
        return "joint"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        if config.arm_joint_mask is not None:
            assert config.dimensionality == np.sum(config.arm_joint_mask)
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_mask_joint(self, joints_pos):
        """Select the joint location"""
        mask_joints_pos = []
        for i in range(len(self._arm_joint_mask)):
            if self._arm_joint_mask[i]:
                mask_joints_pos.append(joints_pos[i])
        return mask_joints_pos

    def get_observation(self, observations, episode, *args, **kwargs):
        if 'physics_target_sps' in kwargs:
            self._sim.step_physics(1.0 / kwargs['physics_target_sps'])
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_joint_pos
        if self._arm_joint_mask is not None:
            joints_pos = self._get_mask_joint(joints_pos)
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class HumanoidJointSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "humanoid_joint_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        curr_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        if isinstance(curr_agent, KinematicHumanoid):
            joints_pos = curr_agent.get_joint_transform()[0]
            return np.array(joints_pos, dtype=np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)


@registry.register_sensor
class JointVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return "joint_vel"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(config.dimensionality,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        joints_pos = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.arm_velocity
        return np.array(joints_pos, dtype=np.float32)


@registry.register_sensor
class EEPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_pos"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEPositionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        # print("eesss:",ee_pos)
        # if self.agent_id == 0:print(str(self.agent_id)+"ee_pos_global:",ee_pos)
        local_ee_pos = trans.inverted().transform_point(ee_pos)
        
        # print("hello:",local_ee_pos)
        # print("goog",goob[:3])
        return np.array(local_ee_pos, dtype=np.float32)

@registry.register_sensor
class EEGlobalPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "ee_global_pos"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EEGlobalPositionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        return np.array(ee_pos, dtype=np.float32)
    
@registry.register_sensor
class TransofrobotSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "robot_trans_martix"

    def _get_uuid(self, *args, **kwargs):
        return TransofrobotSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,4),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        # print(f"agent_{self.agent_id}_trans:{trans}")
        return np.array(trans, dtype=np.float32)


@registry.register_sensor
class RelativeRestingPositionSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "relative_resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RelativeRestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        base_trans = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent.base_transformation
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )
        local_ee_pos = base_trans.inverted().transform_point(ee_pos)

        relative_desired_resting = task.desired_resting - local_ee_pos

        return np.array(relative_desired_resting, dtype=np.float32)


@registry.register_sensor
class RestingPositionSensor(Sensor):
    """
    Desired resting position in the articulated_agent coordinate frame.
    """

    cls_uuid: str = "resting_position"

    def _get_uuid(self, *args, **kwargs):
        return RestingPositionSensor.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, task, *args, **kwargs):
        return np.array(task.desired_resting, dtype=np.float32)


@registry.register_sensor
class LocalizationSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "localization_sensor"

    def __init__(self, sim, config,task, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task

    def _get_uuid(self, *args, **kwargs):
        return LocalizationSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(4,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        articulated_agent = self._sim.get_agent_data(
            self.agent_id
        ).articulated_agent
        T = articulated_agent.base_transformation
        forward = np.array([1.0, 0, 0])
        heading_angle = get_angle_to_pos(T.transform_vector(forward))
        cur_articulated_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        # agent_pos = cur_articulated_agent.base_pos
        # print("task:",self._task.actions,flush = True)
        # for key, value in self._task.actions.items():
        #     try:
        #         print(f"Key: {key}, is_done: {value.skill_done}")
        #     except:
        #         continue
        return np.array(
            [*articulated_agent.base_pos, heading_angle], dtype=np.float32
        )
        # return agent_pos

@registry.register_sensor
class NavTempTranSensor(UsesArticulatedAgentInterface, Sensor):
    """
    ObjPoint
    """

    cls_uuid = "obj_pos"

    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task
        self._targets = {}
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
    def _get_uuid(self, *args, **kwargs):
        return NavTempTranSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    
    def get_observation(self, observations, episode, *args, **kwargs):
        # nav_to_target_idx = kwargs[
        #     self._action_arg_prefix + "oracle_nav_action"
        # ]
        # if nav_to_target_idx <= 0 or nav_to_target_idx > len(
        #     self._poss_entities
        # ):
        #     return
        # nav_to_target_idx = int(nav_to_target_idx[0]) - 1
        # print("nav_to_target_idx:",nav_to_target_idx)
        if self.agent_id == 0:
            nav_to_target_idx = 1
        else:
            nav_to_target_idx = 3
        nav_to_obj = self._entities[nav_to_target_idx]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            nav_to_obj
        )
        return obj_pos
@registry.register_sensor
class TargetPointSensor(UsesArticulatedAgentInterface,Sensor):
    cls_uuid = "target_position"

    def __init__(self, sim, config, task, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._task = task
        self._targets = {}
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
    def _get_uuid(self, *args, **kwargs):
        return TargetPointSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )
    def get_observation(self, observations, episode, *args, **kwargs):
        # nav_to_target_idx = kwargs[
        #     self._action_arg_prefix + "oracle_nav_action"
        # ]
        # if nav_to_target_idx <= 0 or nav_to_target_idx > len(
        #     self._poss_entities
        # ):
        #     return
        # nav_to_target_idx = int(nav_to_target_idx[0]) - 1
        #match_with_the_pddl!!!!!
        if self.agent_id == 0:
            nav_to_target_idx = 0
        else:
            nav_to_target_idx = 1
        nav_to_obj = self._entities[nav_to_target_idx]
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            nav_to_obj
        )
        return obj_pos
@registry.register_sensor
class IsHoldingSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Binary if the robot is holding an object or grasped onto an articulated object.
    """

    cls_uuid: str = "is_holding"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return IsHoldingSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        # if not self.agent_id:
        #     return np.array(
        #         int(self._sim.get_agent_data(self.agent_id).grasp_mgr.force_is_snapped),
        #         dtype=np.float32,
        #     ).reshape((1,))
        is_holding_bool = np.array(
            int(self._sim.get_agent_data(self.agent_id).grasp_mgr.is_grasped),
            dtype=np.float32,
        ).reshape((1,))
        return is_holding_bool


@registry.register_sensor
class ObjectToGoalDistanceSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Euclidean distance from the target object to the goal.
    """

    cls_uuid: str = "object_to_goal_distance_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjectToGoalDistanceSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, *args, episode, **kwargs):
        task = kwargs['task']
        assert 'object_to_goal_distance' in task.measurements.measures
        task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].update_metric(episode=episode)
        obj_to_goal_dist = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()

        dist_to_goal = obj_to_goal_dist[str(task.targ_idx)]
        return dist_to_goal


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """
    Euclidean distance from the target object to the goal.
    """

    cls_uuid: str = "object_to_goal_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        distances = np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class GfxReplayMeasure(Measure):
    cls_uuid: str = "gfx_replay_keyframes_string"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._enable_gfx_replay_save = (
            self._sim.sim_config.sim_cfg.enable_gfx_replay_save
        )
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return GfxReplayMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._gfx_replay_keyframes_string = None
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        if not task._is_episode_active and self._enable_gfx_replay_save:
            self._metric = (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        else:
            self._metric = ""

    def get_metric(self, force_get=False):
        if force_get and self._enable_gfx_replay_save:
            return (
                self._sim.gfx_replay_manager.write_saved_keyframes_to_string()
            )
        return super().get_metric()


@registry.register_measure
class ObjAtGoal(Measure):
    """
    Returns if the target object is at the goal (binary) for each of the target
    objects in the scene.
    """

    cls_uuid: str = "obj_at_goal"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._config = config
        self._succ_thresh = self._config.succ_thresh
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjAtGoal.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                ObjectToGoalDistance.cls_uuid,
            ],
        )
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        obj_to_goal_dists = task.measurements.measures[
            ObjectToGoalDistance.cls_uuid
        ].get_metric()

        self._metric = {
            str(idx): dist < self._succ_thresh
            for idx, dist in obj_to_goal_dists.items()
        }


@registry.register_measure
class EndEffectorToGoalDistance(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "ee_to_goal_distance"

    def __init__(self, sim, *args, **kwargs):
        self._sim = sim
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToGoalDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, observations, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        goals = self._sim.get_targets()[1]

        distances = np.linalg.norm(goals - ee_pos, ord=2, axis=-1)

        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class EndEffectorToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the end-effector and all current target object COMs.
    """

    cls_uuid: str = "ee_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        assert (
            self._config.center_cone_vector is not None
            if self._config.if_consider_gaze_angle
            else True
        ), "Want to consider grasping gaze angle but a target center_cone_vector is not provided in the config."
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        ee_pos = (
            self._sim.get_agent_data(self.agent_id)
            .articulated_agent.ee_transform()
            .translation
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]

        distances = np.linalg.norm(target_pos - ee_pos, ord=2, axis=-1)

        # Ensure the gripper maintains a desirable distance
        distances = abs(
            distances - self._config.desire_distance_between_gripper_object
        )

        if self._config.if_consider_gaze_angle:
            # Get the camera transformation
            cam_T = get_camera_transform(
                self._sim.get_agent_data(self.agent_id).articulated_agent
            )
            # Get angle between (normalized) location and the vector that the camera should
            # look at
            obj_angle = get_camera_object_angle(
                cam_T, target_pos[0], self._config.center_cone_vector
            )
            distances += obj_angle

        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class BaseToObjectDistance(UsesArticulatedAgentInterface, Measure):
    """
    Gets the distance between the base and all current target object COMs.
    """

    cls_uuid: str = "base_to_object_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BaseToObjectDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, **kwargs):
        base_pos = np.array(
            (
                self._sim.get_agent_data(
                    self.agent_id
                ).articulated_agent.base_pos
            )
        )

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = np.array(scene_pos[idxs])
        distances = np.linalg.norm(
            target_pos[:, [0, 2]] - base_pos[[0, 2]], ord=2, axis=-1
        )
        self._metric = {str(idx): dist for idx, dist in enumerate(distances)}


@registry.register_measure
class EndEffectorToRestDistance(Measure):
    """
    Distance between current end effector position and position where end effector rests within the robot body.
    """

    cls_uuid: str = "ee_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return EndEffectorToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        self._metric = rest_dist

@registry.register_measure
class ReturnToRestDistance(UsesArticulatedAgentInterface, Measure):
    """
    Distance between end-effector and resting position if the articulated agent is holding the object.
    """

    cls_uuid: str = "return_to_rest_distance"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ReturnToRestDistance.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        to_resting = observations[RelativeRestingPositionSensor.cls_uuid]
        rest_dist = np.linalg.norm(to_resting)

        snapped_id = self._sim.get_agent_data(self.agent_id).grasp_mgr.snap_idx
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.abs_targ_idx]
        picked_correct = snapped_id == abs_targ_obj_idx

        if picked_correct:
            self._metric = rest_dist
        else:
            T_inv = (
                self._sim.get_agent_data(self.agent_id)
                .articulated_agent.ee_transform()
                .inverted()
            )
            idxs, _ = self._sim.get_targets()
            scene_pos = self._sim.get_scene_pos()
            pos = scene_pos[idxs][0]
            pos = T_inv.transform_point(pos)

            self._metric = np.linalg.norm(task.desired_resting - pos)


@registry.register_measure
class RobotCollisions(UsesArticulatedAgentInterface, Measure):
    """
    Returns a dictionary with the counts for different types of collisions.
    """

    cls_uuid: str = "robot_collisions"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotCollisions.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_coll_info = CollisionDetails()
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        cur_coll_info = self._task.get_cur_collision_info(self.agent_id)
        self._accum_coll_info += cur_coll_info
        self._metric = {
            "total_collisions": self._accum_coll_info.total_collisions,
            "robot_obj_colls": self._accum_coll_info.robot_obj_colls,
            "robot_scene_colls": self._accum_coll_info.robot_scene_colls,
            "obj_scene_colls": self._accum_coll_info.obj_scene_colls,
        }


@registry.register_measure
class RobotForce(UsesArticulatedAgentInterface, Measure):
    """
    The amount of force in newton's accumulatively applied by the robot.
    """

    cls_uuid: str = "articulated_agent_force"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._count_obj_collisions = self._task._config.count_obj_collisions
        self._min_force = self._config.min_force
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RobotForce.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._accum_force = 0.0
        self._prev_force = None
        self._cur_force = None
        self._add_force = None
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    @property
    def add_force(self):
        return self._add_force

    def update_metric(self, *args, episode, task, observations, **kwargs):
        articulated_agent_force, _, overall_force = self._task.get_coll_forces(
            self.agent_id
        )

        if self._count_obj_collisions:
            self._cur_force = overall_force
        else:
            self._cur_force = articulated_agent_force

        if self._prev_force is not None:
            self._add_force = self._cur_force - self._prev_force
            if self._add_force > self._min_force:
                self._accum_force += self._add_force
                self._prev_force = self._cur_force
            elif self._add_force < 0.0:
                self._prev_force = self._cur_force
            else:
                self._add_force = 0.0
        else:
            self._prev_force = self._cur_force
            self._add_force = 0.0

        self._metric = {
            "accum": self._accum_force,
            "instant": self._cur_force,
        }


@registry.register_measure
class NumStepsMeasure(Measure):
    """
    The number of steps elapsed in the current episode.
    """

    cls_uuid: str = "num_steps"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NumStepsMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric += 1


@registry.register_measure
class ZeroMeasure(Measure):
    """
    The number of steps elapsed in the current episode.
    """

    cls_uuid: str = "zero"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ZeroMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self._metric = 0

    def update_metric(self, *args, **kwargs):
        self._metric = 0


@registry.register_measure
class ForceTerminate(Measure):
    """
    If the accumulated force throughout this episode exceeds the limit.
    """

    cls_uuid: str = "force_terminate"

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._max_accum_force = self._config.max_accum_force
        self._max_instant_force = self._config.max_instant_force
        self._task = task
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ForceTerminate.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                RobotForce.cls_uuid,
            ],
        )

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        force_info = task.measurements.measures[
            RobotForce.cls_uuid
        ].get_metric()
        accum_force = force_info["accum"]
        instant_force = force_info["instant"]
        if self._max_accum_force > 0 and accum_force > self._max_accum_force:
            rearrange_logger.debug(
                f"Force threshold={self._max_accum_force} exceeded with {accum_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        elif (
            self._max_instant_force > 0
            and instant_force > self._max_instant_force
        ):
            rearrange_logger.debug(
                f"Force instant threshold={self._max_instant_force} exceeded with {instant_force}, ending episode"
            )
            self._task.should_end = True
            self._metric = True
        else:
            self._metric = False


@registry.register_measure
class DidViolateHoldConstraintMeasure(UsesArticulatedAgentInterface, Measure):
    cls_uuid: str = "did_violate_hold_constraint"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DidViolateHoldConstraintMeasure.cls_uuid

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim

        super().__init__(*args, sim=sim, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, **kwargs):
        self._metric = self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint()


class RearrangeReward(UsesArticulatedAgentInterface, Measure):
    """
    An abstract class defining some measures that are always a part of any
    reward function in the Habitat 2.0 tasks.
    """

    def __init__(self, *args, sim, config, task, **kwargs):
        self._sim = sim
        self._config = config
        self._task = task
        self._force_pen = self._config.force_pen
        self._max_force_pen = self._config.max_force_pen
        self._count_coll_pen = self._config.count_coll_pen
        self._max_count_colls = self._config.max_count_colls
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._prev_count_coll = 0

        target_measure = [RobotForce.cls_uuid, ForceTerminate.cls_uuid]
        if self._want_count_coll():
            target_measure.append(RobotCollisions.cls_uuid)

        task.measurements.check_measure_dependencies(self.uuid, target_measure)

        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        reward = 0.0

        # For force collision reward (userful for dynamic simulation)
        reward += self._get_coll_reward()

        # For count-based collision reward and termination (userful for kinematic simulation)
        if self._want_count_coll():
            reward += self._get_count_coll_reward()

        # For hold constraint violation
        if self._sim.get_agent_data(
            self.agent_id
        ).grasp_mgr.is_violating_hold_constraint():
            reward -= self._config.constraint_violate_pen

        # For force termination
        force_terminate = task.measurements.measures[
            ForceTerminate.cls_uuid
        ].get_metric()
        if force_terminate:
            reward -= self._config.force_end_pen

        self._metric = reward

    def _get_coll_reward(self):
        reward = 0

        force_metric = self._task.measurements.measures[RobotForce.cls_uuid]
        # Penalize the force that was added to the accumulated force at the
        # last time step.
        reward -= max(
            0,  # This penalty is always positive
            min(
                self._force_pen * force_metric.add_force,
                self._max_force_pen,
            ),
        )
        return reward

    def _want_count_coll(self):
        """Check if we want to consider penality from count-based collisions"""
        return self._count_coll_pen != -1 or self._max_count_colls != -1

    def _get_count_coll_reward(self):
        """Count-based collision reward"""
        reward = 0

        count_coll_metric = self._task.measurements.measures[
            RobotCollisions.cls_uuid
        ]
        cur_total_colls = count_coll_metric.get_metric()["total_collisions"]

        # Check the step collision
        if (
            self._count_coll_pen != -1.0
            and cur_total_colls - self._prev_count_coll > 0
        ):
            reward -= self._count_coll_pen

        # Check the max count collision
        if (
            self._max_count_colls != -1.0
            and cur_total_colls > self._max_count_colls
        ):
            reward -= self._config.count_coll_end_pen
            self._task.should_end = True

        # update the counter
        self._prev_count_coll = cur_total_colls

        return reward


@registry.register_measure
class DoesWantTerminate(Measure):
    """
    Returns 1 if the agent has called the stop action and 0 otherwise.
    """

    cls_uuid: str = "does_want_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DoesWantTerminate.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions["rearrange_stop"].does_want_terminate


@registry.register_measure
class BadCalledTerminate(Measure):
    """
    Returns 0 if the agent has called the stop action when the success
    condition is also met or not called the stop action when the success
    condition is not met. Returns 1 otherwise.
    """

    cls_uuid: str = "bad_called_terminate"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return BadCalledTerminate.cls_uuid

    def __init__(self, config, task, *args, **kwargs):
        super().__init__(**kwargs)
        self._success_measure_name = task._config.success_measure
        self._config = config

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DoesWantTerminate.cls_uuid, self._success_measure_name],
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        does_action_want_stop = task.measurements.measures[
            DoesWantTerminate.cls_uuid
        ].get_metric()
        is_succ = task.measurements.measures[
            self._success_measure_name
        ].get_metric()

        self._metric = (not is_succ) and does_action_want_stop


@registry.register_measure
class RuntimePerfStats(Measure):
    cls_uuid: str = "habitat_perf"

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return RuntimePerfStats.cls_uuid

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._sim.enable_perf_logging()
        self._disable_logging = config.disable_logging
        super().__init__()

    def reset_metric(self, *args, **kwargs):
        self._metric_queue = defaultdict(deque)
        self._metric = {}

    def update_metric(self, *args, task, **kwargs):
        for k, v in self._sim.get_runtime_perf_stats().items():
            self._metric_queue[k].append(v)
        if self._disable_logging:
            self._metric = {}
        else:
            self._metric = {
                k: np.mean(v) for k, v in self._metric_queue.items()
            }

@registry.register_sensor
class HasFinishedOracleNavSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Returns 1 if the agent has finished the oracle nav action. Returns 0 otherwise.
    """

    cls_uuid: str = "has_finished_oracle_nav"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return HasFinishedOracleNavSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        # print('aa',self._task.actions)
        # print("kwargs",kwargs,flush = True)
        if self.agent_id is not None:
            use_k = f"agent_{self.agent_id}_oracle_nav_action"
        else:
            use_k = "oracle_nav_action"

        # if use_k not in self._task.actions:
        #     return np.array(False, dtype=np.float32)[..., None]
        # else:
        #     nav_action = self._task.actions[use_k]
        #     return np.array(nav_action.skill_done, dtype=np.float32)[..., None]

        # check if use_k in actions_list
        if use_k not in self._task.actions:
            use_k = "oracle_nav_coord_action"
        assert use_k in self._task.actions, f"your oracle_nav_action name is not defined...usek:{use_k},actions:{self._task.actions}"
        nav_action = self._task.actions[use_k]
        return np.array(nav_action.skill_done, dtype=np.float32)[..., None]

@registry.register_sensor
class NavObjPointSensor(UsesArticulatedAgentInterface, Sensor):

    cls_uuid: str = "target_pos"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        self._entities = self._task.pddl_problem.get_ordered_entities_list()
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return NavObjPointSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
       
        return spaces.Box(shape=(3,),
            low=0,
            high=1,
            dtype=np.float32,)
    def get_observation(self, task, *args, **kwargs):
        if self.agent_id == 0:
            nav_to_target_idx = 0
        else:
            nav_to_target_idx = 1
        try:
            nav_to_target_idx = kwargs['action']['action_args'][
                f"agent_{self.agent_id}_oracle_nav_action"
            ]
            
            nav_to_target_idx = int(nav_to_target_idx[0]) - 1

            # final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            #     nav_to_target_idx
            # )
            # print(f"{self.agent_id}:nav_to_target_idx:{nav_to_target_idx}//self._get_target_for_idx:{self._get_target_for_idx}//",flush = True)
        except Exception as e:
            print(f"Error loading navigation action: {e}")
        nav_to_obj = self._entities[nav_to_target_idx]
        # print(f"agent_{self.agent_id}_target_obj_num_index:{nav_to_obj}",flush = True)
        obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
            nav_to_obj
        )
        return obj_pos

@registry.register_sensor
class GetOracleNavTempSensor(UsesArticulatedAgentInterface, Sensor):
    """
    输出导航到目标点的path
    """

    cls_uuid: str = "oracle_nav_temp"

    def __init__(self, sim, config, *args, task, **kwargs):
        
        self._task = task
        self._sim = sim
        self._targets = {}
    
    
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return GetOracleNavTempSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        # observation_space = spaces.Dict({
        #     "key" : spaces.Box(shape=(1,),low=0, high=1000,dtype=np.float32),
        #     "value" : spaces.Box(shape=(2,),low=0, high=1000,dtype = np.float32)
        # }
        # )
        # # return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)
        # return observation_spac
        # return spaces.Box(shape=(20,3), low=0, high=1, dtype=np.float32)
        return spaces.Box(shape=(20,3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,)
    
    
    def _path_to_point(self,point):
        cur_articulated_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        agent_pos = cur_articulated_agent.base_pos
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points
    
    def get_observation(self, task, *args, **kwargs):
        idxs,goals = self._sim.get_targets()
        # print("aaa:",idxs)
        # print("bbb;",goals)
        # scence_pos = self._sim.get_scene_pos()
        # setter_pos = scence_pos[idxs]
        # setter_pos = setter_pos[1]
        first_three_targets = goals
        # sum = task.nav_goal_pos
        # path = self._path_to_point(final_nav_targ)
        # print("ss:",path)
        path = self._path_to_point(first_three_targets[0])
        return path

    # def get_observations(self, observations, episode, *args, **kwargs):
    #     if self.agent_id is not None:
    #         use_k = f"agent_{self.agent_id}_oracle_nav_action"
    #     else:
    #         use_k = "oracle_nav_action"

    #     if use_k not in self._task.actions:
    #         return np.array(False, dtype=np.float32)[..., None]
    #     else:
    #         nav_action = self._task.actions[use_k]
    #         idxs, target_pos = self._sim.get_targets()
    #         print("target_pos",target_pos)
    #         scene_pos = self._sim.get_scene_pos()
    #         target_point = scene_pos[idxs]
    #         print("target_point",target_point)
    #         first_three_targets = target_point[0]
    #         # print("11:",first_three_targets)
    #         path_points = nav_action._path_to_point(first_three_targets)  # 获取路径点

    #         if len(path_points) == 0:
    #             return np.array([[0,0,0]]*20, dtype=np.float32)
    #         first_point = path_points
    #         return np.array(first_point[:20], dtype=np.float32)
    #         # first_point = path_points[0]
    #         # # return np.array(nav_action.skill_done, dtype=np.float32)[..., None]
    #         # return np.array(nav_action._path_to_point(), dtype=np.float32)[..., None]

    
@registry.register_sensor
class HasFinishedArmActionSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Returns 1 if the agent has finished the arm action. Returns 0 otherwise.
    """

    cls_uuid: str = "has_finished_arm_action"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return HasFinishedArmActionSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if self.agent_id is not None:
            use_k = f"agent_{self.agent_id}_arm_action"
        else:
            use_k = "arm_action"
            
        if use_k not in self._task.actions:
            return np.array(False, dtype=np.float32)[..., None]
        else:
            pick_action = self._task.actions[use_k]
            return np.array(pick_action.skill_done, dtype=np.float32)[..., None]

@registry.register_sensor
class HasFinishedHumanoidPickSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Returns 1 if the agent has finished the oracle nav action. Returns 0 otherwise.
    """

    cls_uuid: str = "has_finished_human_pick"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return HasFinishedHumanoidPickSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(1,), low=0, high=1, dtype=np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if self.agent_id is not None:
            use_k = f"agent_{self.agent_id}_humanoid_pick_action"
        else:
            use_k = "humanoid_pick_action"

        nav_action = self._task.actions[use_k]

        return np.array(nav_action.skill_done, dtype=np.float32)[..., None]


@registry.register_sensor
class ArmDepthBBoxSensor(UsesArticulatedAgentInterface, Sensor):
    """Bounding box sensor to check if the object is in frame"""

    cls_uuid: str = "arm_depth_bbox_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self._height = config.height
        self._width = config.width

    def _get_uuid(self, *args, **kwargs):
        return ArmDepthBBoxSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(
                config.height,
                config.width,
                1,
            ),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, task, *args, **kwargs):
        # Get a correct observation space
        if self.agent_id is None:
            target_key = "articulated_agent_arm_panoptic"
            assert target_key in observations
        else:
            target_key = (
                f"agent_{self.agent_id}_articulated_agent_arm_panoptic"
            )
            assert target_key in observations

        img_seg = observations[target_key]

        # Check the size of the observation
        assert (
            img_seg.shape[0] == self._height
            and img_seg.shape[1] == self._width
        )

        # Check if task has the attribute of the abs_targ_idx
        assert hasattr(task, "abs_targ_idx")

        # Get the target from sim, and ensure that the index is offset
        tgt_idx = (
            self._sim.scene_obj_ids[task.abs_targ_idx]
            + self._sim.habitat_config.object_ids_start
        )
        tgt_mask = (img_seg == tgt_idx).astype(int)

        # Get the bounding box
        bbox = np.zeros(tgt_mask.shape)
        if np.sum(tgt_mask) != 0:
            rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
            bbox[rmin:rmax, cmin:cmax] = 1.0

        return np.float32(bbox)

@registry.register_sensor
class DetectedObjectsSensor(UsesArticulatedAgentInterface, Sensor):
    """ Sensor to detect all objects ids in the semantic sensors """
    cls_uuid: str = "detected_objects"
    
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.pixel_threshold = config.pixel_threshold

    def _get_uuid(self, *args, **kwargs):
        return DetectedObjectsSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args, config, **kwargs):
        # The observation space is flexible, should not be used as gym input
        return spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64)

    # This method assumes the existence of a method to get the semantic sensor's data
    def get_observation(self, observations, *args, **kwargs):
        """ Get the detected objects from the semantic sensor data """
        
        observation_keys = list(observations.keys())
        # Retrieve the semantic sensor data
        if self.agent_id is None:
            target_keys = [key for key in observation_keys if "semantic" in key]
        else:
            target_keys = [
                key for key in observation_keys 
                if f"agent_{self.agent_id}" in key and "semantic" in key
            ]
        
        sensor_detected_objects = {}
        
        for key in target_keys:
            semantic_sensor_data = observations[key]
            
            # Count the occurrence of each object ID in the semantic sensor data
            unique, counts = np.unique(semantic_sensor_data, return_counts=True)
            objects_count = dict(zip(unique, counts))
            
            # Filter objects based on the size threshold
            sensor_detected_objects[key] = [obj_id for obj_id, count in objects_count.items() if count > self.pixel_threshold]
        
        # Concatenate all detected objects from all sensors
        detected_objects = np.unique(np.concatenate(list(sensor_detected_objects.values())))
        if self.agent_id == 0:
            print("sensor_detected_objects:",sensor_detected_objects)
            print("detected:",np.unique(np.concatenate(list(sensor_detected_objects.values()))))
        return detected_objects
    

@registry.register_sensor
class CameraExtrinsicSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "camera_extrinsic"
    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")

    def _get_uuid(self, *args, **kwargs):
        return CameraExtrinsicSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(low=0.0, high=np.finfo(np.float64).max, shape=(4, 4), dtype=np.float64)
    def _get_camera_extrinsic(self, camera_name)-> np.ndarray:
        """get camera extrinsic from habitat simulator config
        Assume the depth and color sensor are aligned and have the same extrinsic parameters
        Args:
            sim (haibtat_sim.Simulator): simulator config class
            camera_name: name of the camera sensor
        """
        camera_key = camera_name.replace("_depth", "")
        cur_articulated_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
        cam_info = cur_articulated_agent.params.cameras[camera_key]
        from habitat.articulated_agents.utils import get_articulated_agent_camera_transform_from_cam_info
        cam_trans = get_articulated_agent_camera_transform_from_cam_info(
            cur_articulated_agent, cam_info)
        return cam_trans

    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
        else:
            depth_camera_name = self.depth_sensor_name
        camera_extrinsic = self._get_camera_extrinsic(depth_camera_name)
        render_camera = self._sim._sensors[f"agent_{self.agent_id}_head_depth"]._sensor_object.render_camera
    
        return np.array(render_camera.camera_matrix)
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
        return spaces.Box(low=0.0, 
                          high=np.finfo(np.float64).max, shape=(self.n,4), dtype=np.float64)
    
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

        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)

        """add semantic information"""
        ep_objects = []
        # for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
        # for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])
        for key, val in self._sim.ep_info.info['object_labels'].items():
            ep_objects.append(key)
        #only add obj semantic
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[semantic_camera_name].squeeze()
        mask = np.isin(semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset).astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            return np.array([[-1,-1,-1,-1]])
@registry.register_sensor
class RecBBoxSenor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "rec_bounding_box"
    def __init__(self, sim,task, config, *args, **kwargs):
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
        return spaces.Box(low=0.0, 
                          high=np.finfo(np.float64).max, shape=(self.n,4), dtype=np.float64)
    
    def get_observation(self, observations, *args, **kwargs):
        """ Get the RGB image with reachable and unreachable points marked """
        ep_objects = []
        is_multi_agent = False
        if self.agent_id is not None:
            depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
            rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
            semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
            is_multi_agent = True
        else:
            depth_obs = observations[self.depth_sensor_name]
            rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            semantic_camera_name = f"head_semantic"
            for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
                ep_objects.append(self._sim.ep_info.target_receptacles[0][i])

            #now the scene_graph_generate is simply have one target rec and one goal rec/
        rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_obs = np.ascontiguousarray(depth_obs)
        if is_multi_agent:
            nav_to_obj_number = -1
            try:
                nav_to_target_idx = kwargs['action']['action_args'][
                    f"agent_{self.agent_id}_oracle_nav_action"
                ]    
                nav_to_target_idx = int(nav_to_target_idx[0]) - 1
                nav_to_obj = self._entities[nav_to_target_idx]
                nav_to_obj = str(nav_to_obj)
                if 'any_targets' in nav_to_obj:
                    match = re.search(r'\|(\d+)-', str(nav_to_obj))
                    if match:
                        nav_to_obj_number = int(match.group(1))
                else:
                    return np.array([[-1,-1,-1,-1]])
            except Exception as e:
                print("e:",e,flush = True)
                return np.array([[-1,-1,-1,-1]])
            
            # print("info::",self._sim.ep_info.target_receptacles)
            # print("nav_to_obj_number:",nav_to_obj_number,flush = True)
            if nav_to_obj_number!= -1:
                for i in range(len(self._sim.ep_info.target_receptacles[nav_to_obj_number]) - 1):
                    ep_objects.append(self._sim.ep_info.target_receptacles[nav_to_obj_number][i])
            else:
                return np.array([[-1,-1,-1,-1]])
        # for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
        # for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
        #     ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])
        # for key, val in self._sim.ep_info.info['object_labels'].items():
        #     ep_objects.append(key)
        #only add obj semantic
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[semantic_camera_name].squeeze()
        mask = np.isin(semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset).astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
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
            return np.array([[union_x,union_y,union_w,union_h]])
            return bounding_box
        else:
            return np.array([[-1,-1,-1,-1]])
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
        return spaces.Box(low=0.0, high=np.finfo(np.float64).max, shape=(256, 256, 1), dtype=np.float64)
    
    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
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
        return spaces.Box(low=0.0, high=np.finfo(np.float64).max, shape=(3, 3), dtype=np.float64)
    
    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
            # semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            # rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            # semantic_camera_name = f"head_semantic"

        # rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_camera = self._sim._sensors[depth_camera_name]._sensor_object.render_camera
        depth_rotation = np.array(depth_camera.camera_matrix.rotation())
        # print("test_rot:",depth_rotation,flush = True)
        return depth_rotation
@registry.register_sensor
class DepthTransSensor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "depth_trans"
    def __init__(self, sim, config, task,*args, **kwargs):
        self._sim = sim
        self._task = task
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
        return spaces.Box(low=0.0, high=np.finfo(np.float64).max, shape=(1, 3), dtype=np.float64)
    
    def get_observation(self, observations, *args, **kwargs):
        if self.agent_id is not None:
            depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
            # rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
            # semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
        else:
            depth_obs = observations[self.depth_sensor_name]
            # rgb_obs = observations[self.rgb_sensor_name]
            depth_camera_name = self.depth_sensor_name
            # semantic_camera_name = f"head_semantic"

        # rgb_obs = np.ascontiguousarray(rgb_obs)
        depth_camera = self._sim._sensors[depth_camera_name]._sensor_object.render_camera
        depth_translation = np.array(depth_camera.camera_matrix.translation)
        # print("test_trans:",depth_translation,flush = True)
        return depth_translation
@registry.register_sensor
class TargetBBoxSenor(UsesArticulatedAgentInterface, Sensor):
    cls_uuid: str = "target_bounding_box"
    def __init__(self, sim,task, config, *args, **kwargs):
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
        return spaces.Box(low=0.0, 
                          high=np.finfo(np.float64).max, shape=(self.n,4), dtype=np.float64)
    
    def get_observation(self, observations, *args, **kwargs):
        ep_objects = []
        is_multi_agent = False
        if self.agent_id is not None:
            depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
            rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
            depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
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
                    nav_to_target_idx = kwargs['action']['action_args'][
                        f"agent_{self.agent_id}_oracle_nav_action"
                    ]    
                    nav_to_target_idx = int(nav_to_target_idx[0]) - 1
                    nav_to_obj = self._entities[nav_to_target_idx]
                    nav_to_obj = str(nav_to_obj)
                    
                    if 'TARGET_any_targets' in nav_to_obj:
                        match = re.search(r'\|(\d+)-', str(nav_to_obj))
                        if match:
                            nav_to_obj_number = int(match.group(1))
                    else:
                        return np.array([[-1,-1,-1,-1]])
                except Exception as e:
                    # print("e:",e,flush = True)
                    return np.array([[-1,-1,-1,-1]])
            # for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
            #     ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
            for i in range(len(self._sim.ep_info.goal_receptacles[nav_to_obj_number]) - 1):
                ep_objects.append(self._sim.ep_info.goal_receptacles[nav_to_obj_number][i])
        objects_info = {}
        rom = self._sim.get_rigid_object_manager()
        for i, handle in enumerate(rom.get_object_handles()):
            if handle in ep_objects:
                obj = rom.get_object_by_handle(handle)
                objects_info[obj.object_id] = handle
        obj_id_offset = self._sim.habitat_config.object_ids_start
        semantic_obs = observations[semantic_camera_name].squeeze()
        mask = np.isin(semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset).astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_box = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                rect = cv2.minAreaRect(contour)
                bound = cv2.boxPoints(rect)
                x, y, w, h = cv2.boundingRect(contour)
                # x,y,w,h = bound
                bounding_box.append((x, y, w, h))
        if bounding_box:
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
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
            return np.array([[union_x,union_y,union_w,union_h]])
        else:
            return np.array([[-1,-1,-1,-1]])
# @registry.register_sensor
# class ArmWorkspaceRGBSensor(UsesArticulatedAgentInterface, Sensor):
#     """ Sensor to visualize the reachable workspace of an articulated arm """
#     cls_uuid: str = "arm_workspace_rgb"

#     def __init__(self, sim, task,config, *args, **kwargs):
#         self._sim = sim
#         # self.agent_idx = config.agent_idx
#         self._task = task
#         self.pixel_threshold = config.pixel_threshold
#         self.height = config.height
#         self.width = config.width
#         self.rgb_sensor_name = config.get("rgb_sensor_name", "head_rgb")
#         self.depth_sensor_name = config.get("depth_sensor_name", "head_depth")
#         self.down_sample_voxel_size = config.get("down_sample_voxel_size", 0.1)
#         self.ctrl_lim = config.get("down_sample_voxel_size", 0.1)
#         self._entities = self._task.pddl_problem.get_ordered_entities_list()
#         self.single_agent_eval_option = config.get("single_agent_eval_option")
#         self.pre_wait = 0
#         super().__init__(config=config)
                
#         self._debug_tf = config.get("debug_tf", False)
#         if self._debug_tf:
#             self.pcl_o3d_list = []
#             self._debug_save_counter = 0
            
#     def _get_uuid(self, *args, **kwargs):
#         # return f"agent_{self.agent_idx}_{ArmWorkspaceRGBSensor.cls_uuid}"
#         return ArmWorkspaceRGBSensor.cls_uuid

#     def _get_sensor_type(self, *args, **kwargs):
#         return SensorTypes.COLOR

#     def _get_observation_space(self, *args, config, **kwargs):
#         return spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

#     def _2d_to_3d(self, depth_name, depth_obs):
#         # get the scene render camera and sensor object
#         depth_camera = self._sim._sensors[depth_name]._sensor_object.render_camera

#         hfov = float(self._sim._sensors[depth_name]._sensor_object.hfov) * np.pi / 180.
#         W, H = depth_camera.viewport[0], depth_camera.viewport[1]
#         K = np.array([
#             [1 / np.tan(hfov / 2.), 0., 0., 0.],
#             [0., 1 / np.tan(hfov / 2.), 0., 0.],
#             [0., 0., 1, 0],
#             [0., 0., 0, 1]
#         ])

#         xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
#         depth = depth_obs.reshape(1, W, W)
#         xs = xs.reshape(1, W, W)
#         ys = ys.reshape(1, W, W)
#         # print(f"hfov:{hfov},W:{W},H:{H},K:{K},")
#         xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
#         xys = xys.reshape(4, -1)
#         xy_c = np.matmul(np.linalg.inv(K), xys)

#         depth_rotation = np.array(depth_camera.camera_matrix.rotation())
#         depth_translation = np.array(depth_camera.camera_matrix.translation)

#         # get camera-to-world transformation
#         T_world_camera = np.eye(4)
#         T_world_camera[0:3, 0:3] = depth_rotation
#         T_world_camera[0:3, 3] = depth_translation

#         T_camera_world = np.linalg.inv(T_world_camera)
#         points_world = np.matmul(T_camera_world, xy_c)

#         # get non_homogeneous points in world space
#         points_world = points_world[:3, :] / points_world[3, :]
#         # reshape to the scale of the image
#         # points_world = points_world.reshape((3, H, W)).transpose(1, 2, 0)
#         points_world = points_world.transpose(1, 0)

#         return points_world

#     def _3d_to_2d(self, sensor_name, point_3d):
#         # get the scene render camera and sensor object
#         render_camera = self._sim._sensors[sensor_name]._sensor_object.render_camera
#         W, H = render_camera.viewport[0], render_camera.viewport[1]

#         # use the camera and projection matrices to transform the point onto the near plane
#         projected_point_3d = render_camera.projection_matrix.transform_point(
#             render_camera.camera_matrix.transform_point(point_3d)
#         )
#         # convert the 3D near plane point to integer pixel space
#         point_2d = mn.Vector2(projected_point_3d[0], -projected_point_3d[1])
#         point_2d = point_2d / render_camera.projection_size()[0]
#         point_2d += mn.Vector2(0.5)
#         point_2d *= render_camera.viewport
#         # print(f"info:view:{render_camera.viewport}/size:{render_camera.projection_size()[0]}/")
#         out_bound = 10
#         point_2d = np.nan_to_num(point_2d, nan=W+out_bound, posinf=W+out_bound, neginf=-out_bound)
#         return point_2d.astype(int)

#     def voxel_grid_filter(self, points, voxel_size):
#         voxel_indices = np.floor(points / voxel_size).astype(int)

#         voxel_dict = {}
#         for i, voxel_index in enumerate(voxel_indices):
#             voxel_key = tuple(voxel_index)
#             if voxel_key not in voxel_dict:
#                 voxel_dict[voxel_key] = []
#             voxel_dict[voxel_key].append(points[i])

#         downsampled_points = []
#         for voxel_key in voxel_dict:
#             voxel_points = np.array(voxel_dict[voxel_key])
#             mean_point = voxel_points.mean(axis=0)
#             downsampled_points.append(mean_point)

#         return np.array(downsampled_points)

#     def _is_reachable(self, cur_articulated_agent, ik_helper, point, thresh=0.05):
#         cur_base_pos, cur_base_orn = ik_helper.get_base_state()
#         base_transformation = cur_articulated_agent.base_transformation
    
#         orn_quaternion = mn.Quaternion.from_matrix(base_transformation.rotation())
#         base_pos = base_transformation.translation
#         base_orn = list(orn_quaternion.vector)
#         base_orn.append(orn_quaternion.scalar)
#         ik_helper.set_base_state(base_pos, base_orn)
#         # point_base = cur_articulated_agent.base_transformation.inverted().transform_vector(point)
#         point_base = point

#         cur_joint_pos = cur_articulated_agent.arm_joint_pos

#         des_joint_pos = ik_helper.calc_ik(point_base)
#         # temporarily set arm joint position
#         if cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
#             cur_articulated_agent.arm_motor_pos = des_joint_pos
#         if cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
#             cur_articulated_agent.arm_joint_pos = des_joint_pos
#             cur_articulated_agent.fix_joint_values = des_joint_pos

#         des_ee_pos = ik_helper.calc_fk(des_joint_pos)

#         # revert arm joint position
#         if cur_articulated_agent.sim_obj.motion_type == MotionType.DYNAMIC:
#             cur_articulated_agent.arm_motor_pos = cur_joint_pos
#         if cur_articulated_agent.sim_obj.motion_type == MotionType.KINEMATIC:
#             cur_articulated_agent.arm_joint_pos = cur_joint_pos
#             cur_articulated_agent.fix_joint_values = cur_joint_pos

#         ik_helper.set_base_state(cur_base_pos, cur_base_orn)

#         return np.linalg.norm(np.array(point_base) - np.array(des_ee_pos)) < thresh

#     def get_observation(self, observations, *args, **kwargs):
#         """ Get the RGB image with reachable and unreachable points marked """
#         if self.agent_id is not None:
#             depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
#             rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
#             depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
#             semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
#         else:
#             depth_obs = observations[self.depth_sensor_name]
#             rgb_obs = observations[self.rgb_sensor_name]
#             depth_camera_name = self.depth_sensor_name
#             semantic_camera_name = f"head_semantic"
#         wait_flag = int(self._task.actions["wait"][0])
#         if not(self.pre_wait is 1 and wait_flag is 0):
#             self.pre_wait = 0
#             return np.array([0.])
#         rgb_obs = np.ascontiguousarray(rgb_obs)
#         depth_obs = np.ascontiguousarray(depth_obs)

#         """add semantic information"""
#         ep_objects = []
#         nav_to_obj_number = -1
#         try:
#             nav_to_target_idx = kwargs['action']['action_args'][
#                 f"agent_{self.agent_id}_oracle_nav_action"
#             ]    
#             nav_to_target_idx = int(nav_to_target_idx[0]) - 1
#             nav_to_obj = self._entities[nav_to_target_idx]
#             match = re.search(r'\|(\d+)-', str(nav_to_obj))
#             if match:
#                 nav_to_obj_number = int(match.group(1))
#         except Exception as e:
#             print(f"Error loading navigation action: {e}")
        
#         # print("info::",self._sim.ep_info.target_receptacles)
#         # print("nav_to_obj_number:",nav_to_obj_number,flush = True)
#         if nav_to_obj_number!= -1:
#             for i in range(len(self._sim.ep_info.target_receptacles[nav_to_obj_number]) - 1):
#                 ep_objects.append(self._sim.ep_info.target_receptacles[nav_to_obj_number][i])
#             for i in range(len(self._sim.ep_info.goal_receptacles[nav_to_obj_number]) - 1):
#                 ep_objects.append(self._sim.ep_info.goal_receptacles[nav_to_obj_number][i])
#         else:
#             for item in range(len(self._sim.ep_info.target_receptacles)):
#                 for i in range(len(self._sim.ep_info.target_receptacles[item]) - 1):
#                     ep_objects.append(self._sim.ep_info.target_receptacles[item][i])
        
#             for i in range(len(self._sim.ep_info.goal_receptacles[item]) - 1):
#                 ep_objects.append(self._sim.ep_info.goal_receptacles[item][i])
#             # for key, val in self._sim.ep_info.info['object_labels'].items():
#             #     ep_objects.append(key)
#         # print("ep_objects:",ep_objects,flush=True)
#         objects_info = {}
#         rom = self._sim.get_rigid_object_manager()
#         for i, handle in enumerate(rom.get_object_handles()):
#             # print("handle:",handle)
#             if handle in ep_objects:
#                 obj = rom.get_object_by_handle(handle)
#                 objects_info[obj.object_id] = handle
#         obj_id_offset = self._sim.habitat_config.object_ids_start

#         semantic_obs = observations[semantic_camera_name].squeeze()

#         mask = np.isin(semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset).astype(np.uint8)

#         contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         bounding_box = []
#         colored_mask = np.zeros_like(rgb_obs)
#         colored_mask[mask == 1] = [0, 0, 255]
#         # rgb_obs = cv2.addWeighted(rgb_obs, 0.5, colored_mask, 0.5, 0)
#         for contour in contours:
#             if cv2.contourArea(contour) > 0:  # 过滤掉面积为0的轮廓
#                 x, y, w, h = cv2.boundingRect(contour)
#                 bounding_box.append((x, y, w, h))
#                 # 可选：在原始图像上绘制边界框
#                 # cv2.rectangle(rgb_obs, (x, y), (x + w, y + h), (255, 0, 0), 1)  #是否标boundingbox
#         # for obj_id in objects_info.keys():
#         #     positions = np.where(semantic_obs == obj_id + obj_id_offset)
#         #     if positions[0].size > 0:
#         #         center_x = int(np.mean(positions[1]))
#         #         center_y = int(np.mean(positions[0]))
#         #         cv2.putText(rgb_obs, objects_info[obj_id], (center_x, center_y),
#         #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#         # Reproject depth pixels to 3D points
#         points_world = self._2d_to_3d(depth_camera_name, depth_obs)
#         # downsample the 3d-points
#         down_sampled_points = self.voxel_grid_filter(points_world, self.down_sample_voxel_size)

#         # Check reachability and color points
#         colors = []

#         articulated_agent_mgr = self._sim.agents_mgr._all_agent_data[self.agent_id if self.agent_id is not None else 0]
#         cur_articulated_agent = articulated_agent_mgr.articulated_agent
#         ik_helper = articulated_agent_mgr.ik_helper

#         for point in down_sampled_points:
#             reachable = self._is_reachable(cur_articulated_agent, ik_helper, point)
#             # Green if reachable, red if not
#             colors.append([0, 255, 0] if reachable else [255, 0, 0])

#         pixel_coords = []
#         if self._debug_tf and 'obj_pos' in kwargs:
#             if kwargs['obj_pos'] is not None:
#                 pixel_coord = self._3d_to_2d(depth_camera_name, np.array(list(kwargs['obj_pos'])))
#                 if np.any(np.isnan(pixel_coord)) or np.any(np.isinf(pixel_coord)):
#                     print("obj_pos is invalid")
#                 else:
#                     down_sampled_points = np.array([list(kwargs['obj_pos'])])
#                     colors = [[0, 255, 0]]

#         # Project the points to the image and color the pixels with circles
#         for point in down_sampled_points:
#             pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

#         for pixel_coord, color in zip(pixel_coords, colors):
#             x, y = pixel_coord.astype(int)
#             if color == [0, 255, 0]:
#                 if self._debug_tf:
#                     if x < 256 and y < 256:
#                         print(f"obj_pos can be seen: {x}, {y}")
#                     else:
#                         print(f"obj_pos can not be seen: {x}, {y}")
#                 cv2.circle(rgb_obs, (x, y), 2, color, -1)
#                     # the arm now is out of robot's head_rgb so that don't need mask
#         # mask_img = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
#         # bgr = mask_img[:, :, :3]
#         # alpha = mask_img[:, :, 3]
#         # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         # mask_img_rga = cv2.merge((rgb, alpha))
#         # assert rgb_obs.shape[:2] == mask_img_rga.shape[:2]
#         # alpha_channel = mask_img_rga[:, :, 3] / 255.0
#         # rgb_obs
#         # for c in range(0, 3):
#         #     rgb_obs[:, :, c] = (alpha_channel * mask_img_rga[:, :, c] +
#         #                         (1 - alpha_channel) * rgb_obs[:, :, c])
#         return rgb_obs

# @registry.register_sensor
# class ArmWorkspacePointsSensor(ArmWorkspaceRGBSensor):
#     """ Sensor to visualize the reachable workspace of an articulated arm """
#     cls_uuid: str = "arm_workspace_points"

#     def _get_uuid(self, *args, **kwargs):
#         return ArmWorkspacePointsSensor.cls_uuid

#     def get_observation(self, observations, *args, **kwargs):
#         """ Get the RGB image with reachable and unreachable points marked """

#         if self.agent_id is not None:
#             depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
#             depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
#         else:
#             depth_obs = observations[self.depth_sensor_name]
#             depth_camera_name = self.depth_sensor_name

#         depth_obs = np.ascontiguousarray(depth_obs)
#         points_world = self._2d_to_3d(depth_camera_name, depth_obs)
#         # downsample the 3d-points
#         down_sampled_points = self.voxel_grid_filter(points_world, self.down_sample_voxel_size)

#         # Project the points to the image and color the pixels with circles
#         pixel_coords = []
#         for point in down_sampled_points:
#             pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

#         flat_points = []
#         space_points = []
#         ik_helper = self._sim.get_agent_data(self.agent_id).ik_helper
#         cur_articulated_agent = self._sim.get_agent_data(self.agent_id).articulated_agent
#         for idx, (point_3d, point_2d) in enumerate(zip(down_sampled_points, pixel_coords)):
#             reachable = self._is_reachable(cur_articulated_agent, ik_helper, point_3d)
#             if reachable:
#                 point = np.concatenate((point_2d, [idx]), axis=0)
#                 flat_points.append(point)
#                 space_points.append(point_3d)
#         return np.array(flat_points)


# @registry.register_sensor
# class ArmWorkspaceRGBThirdSensor(ArmWorkspaceRGBSensor):
#     """ Sensor to visualize the reachable workspace of an articulated arm """
#     cls_uuid: str = "arm_workspace_rgb_third"

#     def __init__(self, sim, config, *args, **kwargs):
#         super().__init__(sim=sim, config=config, *args, **kwargs)

#         self.rgb_sensor_name = config.get("rgb_sensor_name", "third_rgb")
#         self.depth_sensor_name = config.get("depth_sensor_name", "third_depth")


# @registry.register_sensor
# class ObjectMasksSensor(UsesArticulatedAgentInterface, Sensor):
#     """ Sensor to mask the objects that are interested """
#     cls_uuid: str = "object_masks"

#     def __init__(self, sim, config, *args, **kwargs):
#         super().__init__(config=config)
#         self._sim = sim

#     def _get_uuid(self, *args, **kwargs):
#         return ObjectMasksSensor.cls_uuid

#     def _get_sensor_type(self, *args, **kwargs):
#         return SensorTypes.SEMANTIC

#     def _get_observation_space(self, *args, config, **kwargs):
#         # The observation space is flexible, should not be used as gym input
#         return spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64)

#     def get_observation(self, observations, *args, **kwargs):
#         """ Get the detected objects from the semantic sensor data """
#         objects_info = {}
#         # print("sim",self._sim)
#         rom = self._sim.get_rigid_object_manager()
#         for i, handle in enumerate(rom.get_object_handles()):
#             obj = rom.get_object_by_handle(handle)
#             objects_info[obj.object_id] = handle
#         """articulated object not in used for now"""
#         # aom = self._sim.get_articulated_object_manager()
#         # for i, handle in enumerate(aom.get_object_handles()):
#         #     obj = aom.get_object_by_handle(handle)
#         #     objects_info[obj.object_id] = handle
#         obj_id_offset = self._sim.habitat_config.object_ids_start
#         semantic_obs = observations[f'agent_{self.agent_id}_head_semantic'].squeeze()
#         rgb_obs = observations[f'agent_{self.agent_id}_head_rgb']

#         mask = np.isin(semantic_obs, np.array(self._sim.scene_obj_ids) + obj_id_offset).astype(np.uint8)
#         colored_mask = np.zeros_like(rgb_obs)
#         colored_mask[mask == 1] = [0, 0, 255]
#         masked_rgb = cv2.addWeighted(rgb_obs, 0.5, colored_mask, 0.5, 0)

#         for obj_id in np.array(self._sim.scene_obj_ids):
#             positions = np.where(semantic_obs == obj_id + obj_id_offset)
#             if positions[0].size > 0:
#                 center_x = int(np.mean(positions[1]))
#                 center_y = int(np.mean(positions[0]))
#                 cv2.putText(masked_rgb, objects_info[obj_id], (center_x, center_y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#         return masked_rgb


# @registry.register_sensor
# class NavWorkspaceRGBSensor(ArmWorkspaceRGBSensor, UsesArticulatedAgentInterface, Sensor):
#     """ Sensor to visualize the reachable workspace of an articulated arm """
#     cls_uuid: str = "nav_workspace_rgb"

#     def __init__(self, sim, config, task,*args, **kwargs):
#         super().__init__(sim=sim, config=config,task=task)

#     def _get_uuid(self, *args, **kwargs):
#         return NavWorkspaceRGBSensor.cls_uuid

#     def _is_reachable(self, point):
#         agent_pos = self._sim.get_agent_data(self.agent_id).articulated_agent.base_pos

#         path = habitat_sim.ShortestPath()
#         path.requested_start = agent_pos
#         path.requested_end = point
#         found_path = self._sim.pathfinder.find_path(path)

#         return found_path

#     def _get_plane_points(self, points):
#         X = points[:, [0, 2]]
#         y = points[:, 1]

#         ransac = make_pipeline(PolynomialFeatures(1),
#                                RANSACRegressor(LinearRegression()))
#         ransac.fit(X, y)
#         inlier_mask = ransac.named_steps['ransacregressor'].inlier_mask_
#         plane_points = points[inlier_mask]
#         return plane_points

#     def get_observation(self, observations, *args, **kwargs):
#         """ Get the RGB image with reachable and unreachable points marked """

#         if self.agent_id is not None:
#             depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
#             rgb_obs = observations[f"agent_{self.agent_id}_{self.rgb_sensor_name}"]
#             depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
#             semantic_camera_name = f"agent_{self.agent_id}_head_semantic"
#         else:
#             depth_obs = observations[self.depth_sensor_name]
#             rgb_obs = observations[self.rgb_sensor_name]
#             depth_camera_name = self.depth_sensor_name
#             semantic_camera_name = "head_semantic"

#         rgb_obs = np.ascontiguousarray(rgb_obs)
#         depth_obs = np.ascontiguousarray(depth_obs)

#         """add semantic information"""
#         ep_objects = []
#         for i in range(len(self._sim.ep_info.target_receptacles[0]) - 1):
#             ep_objects.append(self._sim.ep_info.target_receptacles[0][i])
#         for i in range(len(self._sim.ep_info.goal_receptacles[0]) - 1):
#             ep_objects.append(self._sim.ep_info.goal_receptacles[0][i])
#         for key, val in self._sim.ep_info.info['object_labels'].items():
#             ep_objects.append(key)

#         objects_info = {}
#         rom = self._sim.get_rigid_object_manager()
#         for i, handle in enumerate(rom.get_object_handles()):
#             if handle in ep_objects:
#                 obj = rom.get_object_by_handle(handle)
#                 objects_info[obj.object_id] = handle
#         obj_id_offset = self._sim.habitat_config.object_ids_start

#         semantic_obs = observations[semantic_camera_name].squeeze()

#         mask = np.isin(semantic_obs, np.array(list(objects_info.keys())) + obj_id_offset).astype(np.uint8)
#         colored_mask = np.zeros_like(rgb_obs)
#         colored_mask[mask == 1] = [0, 0, 255]
#         rgb_obs = cv2.addWeighted(rgb_obs, 0.5, colored_mask, 0.5, 0)

#         for obj_id in objects_info.keys():
#             positions = np.where(semantic_obs == obj_id + obj_id_offset)
#             if positions[0].size > 0:
#                 center_x = int(np.mean(positions[1]))
#                 center_y = int(np.mean(positions[0]))
#                 cv2.putText(rgb_obs, objects_info[obj_id], (center_x, center_y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#         # Reproject depth pixels to 3D points
#         points_world = self._2d_to_3d(depth_camera_name, depth_obs)

#         """get the ground points using RANSAC"""
#         # 过滤接近地面高度的点
#         ground_points_mask = np.abs(points_world[:, 1] - points_world[:, 1].min()) < 0.1
#         ground_points = points_world[ground_points_mask]
#         ground_points = self._get_plane_points(ground_points)

#         # downsample the 3d-points
#         down_sampled_points = self.voxel_grid_filter(ground_points, self.down_sample_voxel_size)

#         # Check reachability and color points
#         colors = []
#         for point in down_sampled_points:
#             reachable = self._is_reachable(point)
#             # Green if reachable, red if not
#             colors.append([0, 255, 0] if reachable else [255, 0, 0])

#         # Project the points to the image and color the pixels with circles
#         pixel_coords = []
#         for point in down_sampled_points:
#             pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

#         for pixel_coord, color in zip(pixel_coords, colors):
#             x, y = pixel_coord.astype(int)
#             if color == [0, 255, 0]:
#                 cv2.circle(rgb_obs, (x, y), 2, color, -1)

#         return rgb_obs

# @registry.register_sensor
# class NavWorkspacePointsSensor(NavWorkspaceRGBSensor):
#     """ Sensor to visualize the reachable workspace of an articulated arm """
#     cls_uuid: str = "nav_workspace_points"

#     def _get_uuid(self, *args, **kwargs):
#         return NavWorkspacePointsSensor.cls_uuid

#     def get_observation(self, observations, *args, **kwargs):
#         """ Get the RGB image with reachable and unreachable points marked """

#         if self.agent_id is not None:
#             depth_obs = observations[f"agent_{self.agent_id}_{self.depth_sensor_name}"]
#             depth_camera_name = f"agent_{self.agent_id}_{self.depth_sensor_name}"
#         else:
#             depth_obs = observations[self.depth_sensor_name]
#             depth_camera_name = self.depth_sensor_name

#         depth_obs = np.ascontiguousarray(depth_obs)

#         # Reproject depth pixels to 3D points
#         points_world = self._2d_to_3d(depth_camera_name, depth_obs)

#         """get the ground points using RANSAC"""
#         # 过滤接近地面高度的点
#         ground_points_mask = np.abs(points_world[:, 1] - points_world[:, 1].min()) < 0.1
#         ground_points = points_world[ground_points_mask]
#         ground_points = self._get_plane_points(ground_points)

#         # downsample the 3d-points
#         down_sampled_points = self.voxel_grid_filter(ground_points, self.down_sample_voxel_size)

#         # Project the points to the image and color the pixels with circles
#         pixel_coords = []
#         for point in down_sampled_points:
#             pixel_coords.append(self._3d_to_2d(depth_camera_name, point))

#         # Check reachability and color points
#         flat_points = []
#         space_points = []

#         for idx, (point_3d, point_2d) in enumerate(zip(down_sampled_points, pixel_coords)):
#             reachable = self._is_reachable(point_3d)
#             if reachable:
#                 point = np.concatenate((point_2d, [idx]), axis=0)
#                 flat_points.append(point)
#                 space_points.append(point_3d)

#         return np.array([space_points, flat_points])

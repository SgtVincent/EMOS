from functools import partial
import os
import sys
import time
from typing import List
import pathlib
import numpy as np
import pickle
import open3d as o3d
import copy 
import plotly.graph_objects as go
from scipy import stats
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.articulated_agent_manager import ArticulatedAgentData
from habitat.articulated_agents.mobile_manipulator import MobileManipulator

from habitat_mas.utils.constants import coco_categories, coco_label_mapping
from habitat_mas.scene_graph.scene_graph_base import SceneGraphBase
from habitat_mas.scene_graph.utils import (
    visualize_scene_graph,
    # generate_region_adjacency_description,
    # generate_region_objects_description,
    generate_objects_description,
    generate_agents_description
)
from habitat_mas.perception.grid_map import GridMap
from habitat_mas.perception.nav_mesh import NavMesh

class SceneGraphHSSD(SceneGraphBase):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.sim: RearrangeSim = None
        self.gt_point_cloud = None
        
        self.meters_per_grid = kwargs.get('meters_per_grid', 0.05)
        self.object_grid_scale = kwargs.get('object_grid_scale', 1)
        self.aligned_bbox = kwargs.get('aligned_bbox', True)
        self.enable_region_layer = False
        # Util habitat-sim v0.3.1, HM3D region annotation missing
        # self.compute_region_bbox = kwargs.get('compute_region_bbox', True)

    def load_gt_scene_graph(self, sim: RearrangeSim):
        # register habitat simulator
        self.sim = sim
        # get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = self.sim.pathfinder.get_bounds()
        self.current_height = sim.get_agent(0).state.position[1]

        # 1. load navmesh from habitat simulator
        navmesh_vertices: List[np.ndarray] = self.sim.pathfinder.build_navmesh_vertices()
        navmesh_indices: List[int]= self.sim.pathfinder.build_navmesh_vertex_indices()
        self.nav_mesh = NavMesh(
            vertices=np.stack(navmesh_vertices, axis=0),
            triangles=np.array(navmesh_indices).reshape(-1, 3),
        )
            
        # 3. load object layer from habitat simulator
        rom = self.sim.get_rigid_object_manager()
        for object_handle, obj_id in sim.handle_to_object_id.items():
            # alias name for object
            if obj_id in self.object_layer.obj_ids:
                #TODO: add alias to object node
                continue
            
            
            abs_obj_id = self.sim.scene_obj_ids[obj_id]
            cur_pos = rom.get_object_by_id(
                abs_obj_id
            ).transformation.translation
            cur_rot = rom.get_object_by_id(
                abs_obj_id
            ).transformation.rotation

            obj_label = None
            if object_handle in sim._handle_to_goal_name:
                obj_label = sim._handle_to_goal_name[object_handle]
            
            self.object_layer.add_object(
                cur_pos,
                cur_rot,
                # size,
                id=obj_id,
                label=obj_label,
                full_name=object_handle,
                # class_name=obj.category.name(),
                # bbox=node_bbox,
            )
        
        target_trans = self.sim._get_target_trans()
        if len(target_trans):
            targets = {}
            for id, (target_id, trans) in enumerate(target_trans):
                targets[id] = trans

            if self.sim.ep_info.goal_receptacles and len(self.sim.ep_info.goal_receptacles):
                for target_id, goal_recep in enumerate(self.sim.ep_info.goal_receptacles):
                    goal_recep_handle = goal_recep[0]
                    assert target_id in targets
                    target = targets[target_id]
                    target_pos = target.translation
                    target_rot = target.rotation

                    goal_recep = rom.get_object_by_handle(goal_recep_handle)
                    abs_obj_id = goal_recep.object_id + self.sim.habitat_config.object_ids_start
                    self.object_layer.add_object(
                        target_pos,
                        target_rot,
                        id=abs_obj_id,
                        full_name=goal_recep_handle,
                        label=f"TARGET_any_targets|{target_id}"
                    )

        # 5. load agents from habitat simulator
        for i, agent_name in enumerate(sim.agents_mgr.agents_order):
            agent: MobileManipulator = sim.agents_mgr[i].articulated_agent
            agent_name = agent_name
            
            # Assuming agent_layer.add_agent is modified to accept more parameters
            self.agent_layer.add_agent(
                agent_id=i, 
                agent_name=agent_name,
                position=agent.base_pos,
                orientation=agent.base_rot,
                # Add more parameters here
                description=sim.agents_mgr[i].cfg['articulated_agent_type']
            )
                    
                    
    def load_gt_geometry(self):
        # load the ply file of the scene
        scene_id = self.sim.config.sim_cfg.scene_id
        ply_file_path = scene_id[:-4] + "_semantic.ply"
        
        # load the ply file with open3d 
        if os.path.exists(ply_file_path):
            self.gt_point_cloud = o3d.io.read_point_cloud(ply_file_path)
        else:
            print(f"PLY file {ply_file_path} not found")


if __name__ == "__main__":
    import habitat_sim
    import os 
    from habitat_mas.test.data_utils import get_fetch_hssd_env

    env = get_fetch_hssd_env()
    env.reset()

    # initialize scene graph
    sg = SceneGraphHSSD()
    sg.load_gt_scene_graph(env.sim)

    # save the navmesh triangle mesh and region bounding box dict to files
    # o3d.io.write_triangle_mesh("1LXtFkjw3qL.obj", sg.nav_mesh.mesh)
    # pickle.dump(region_bbox_dict, open("1LXtFkjw3qL_region_bbox_dict.pkl", "wb"))
    
    ############# Visualization ##################
    
    # visualize_scene_graph(
    #     scene_graph=sg,
    #     scene_o3d=sg.gt_point_cloud,
    #     vis_region_bbox=True,
    #     vis_object_bbox=False,
    #     vis_navmesh=True,
    #     navmesh_shift=[0, 0, -8.0],
    #     vis_region_graph=True,
    #     region_graph_shift=[0, 0, 10.0],
    #     mp3d_coord=True
    # )

    ############ Generate scene description ###########

    # Generate scene descriptions
    objects_description = generate_objects_description(env.sim, sg.object_layer)
    agent_description = generate_agents_description(sg.agent_layer, sg.region_layer, sg.nav_mesh)
        
    # print(region_scene_graph_description)
    print("\n")
    print(agent_description)
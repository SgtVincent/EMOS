import os
import copy
import numpy as np
import open3d as o3d
import quaternion as qt
import networkx as nx

from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
# from scene_graph.scene_graph_base import SceneGraphBase

###################### grid map ###############################


def project_points_to_grid_xz(bounds, points, meters_per_pixel):
    """
    @bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    @points: numpy array with shape (N, 3)
    return: points_grid, numpy array (N,2), dtype=int
    """

    # convert 3D x,z to topdown x,y
    pxs = (points[:, 0] - bounds[0][0]) / meters_per_pixel
    pys = (points[:, 2] - bounds[0][2]) / meters_per_pixel
    points_grid = np.stack([pxs, pys]).astype(int).T
    return points_grid


def grid_xz_to_points(bounds, grids, height, meters_per_pixel):
    """
    @bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    @grids: numpy array with shape (N, 2)
    @height: fixed height (y-axis) of points
    return: points, numpy array (N,3)
    """

    points_x = grids[:, 0] * meters_per_pixel + bounds[0][0]
    points_z = grids[:, 1] * meters_per_pixel + bounds[0][2]
    points_y = np.full_like(points_x, height)
    return np.stack([points_x, points_y, points_z], axis=1)


################ bounding box ######################################

# vertices order of box for visualization
"""
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
"""

BBOX_EDGES = np.asarray(
    [
        [0, 1],
        [0, 3],
        [0, 4],
        [2, 1],
        [2, 3],
        [2, 6],
        [5, 1],
        [5, 4],
        [5, 6],
        [7, 3],
        [7, 4],
        [7, 6],
    ]
)


def rotate_in_xy(pt, theta):
    # pt: (3,) numpy vector (x,y,z)
    # theta: in radius
    r_mat = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return r_mat @ pt


def rotate(pt, r_mat):
    # pt: (3,) numpy vector (x,y,z)
    # mat: rotation mat
    return r_mat @ pt


def get_corners(center, size, quat=np.array([0, 0, 0, 1])):

    l, w, h = size

    # calculate 8 bbox vertices in 3D space
    local_corners = np.array(
        [
            [l / 2, w / 2, h / 2],  # (+x,+y,+z)
            [-l / 2, w / 2, h / 2],  # (-x,+y,+z)
            [-l / 2, -w / 2, h / 2],  # (-x,-y,+z)
            [l / 2, -w / 2, h / 2],  # (+x,-y,+z)
            [l / 2, w / 2, -h / 2],  # (+x,+y,-z)
            [-l / 2, w / 2, -h / 2],  # (-x,+y,-z)
            [-l / 2, -w / 2, -h / 2],  # (-x,-y,-z)
            [l / 2, -w / 2, -h / 2],  # (+x,-y,-z)
        ]
    )
    r_mat = R.from_quat(quat).as_matrix()
    rotated_corners = local_corners @ r_mat.T
    world_corners = rotated_corners + center

    return world_corners

def aggregate_bboxes(bboxes):
    '''
    Aggregate list of bounding boxes into a single bounding box
    Args:
        bboxes: list of bounding boxes, each bounding box is a numpy array with shape (2, 3)
    Returns: 
        aggregated bounding box, numpy array with shape (2, 3)
    '''
    min_bound = np.min([bbox[0] for bbox in bboxes], axis=0)
    max_bound = np.max([bbox[1] for bbox in bboxes], axis=0)
    return np.array([min_bound, max_bound])

########### Scene to text ###################

def generate_region_adjacency_description(reion_layer):
    num_regions = len(reion_layer.nodes)
    description = "There are {} regions in the scene.".format(num_regions)    
    visited = set()
    connected_components = []
    
    def dfs(region_node, component):
        if region_node in visited:
            return
        visited.add(region_node)
        component.append(region_node)
        for neighbor_node in reion_layer.neighbors(region_node):
            if neighbor_node not in visited:
                dfs(neighbor_node, component)
    
    # use dfs to find connected components
    for region_node in reion_layer.nodes:
        if region_node not in visited:
            component = []
            dfs(region_node, component)
            connected_components.append(component)

    description += " The regions are grouped into {} connected components. Within each component, all regions are connected, while regions from different components have no connections to each other. \n ".format(len(connected_components))
    
    for i, component in enumerate(connected_components):
        region_names = []
        for region_node in component:
            region_id = region_node.region_id
            region_name = region_node.class_name + "_" + str(region_id)
            region_names.append(region_name)
        
        description += "Component {}: [{}] are all connected. \n".format(i + 1, ", ".join(region_names))

    return description

def generate_region_objects_description(region_layer, region_id):
    """
    Generate description of objects in each region
    """
    region = region_layer.region_dict[region_id]
    region_full_name = region.class_name + str(region_id)
    description = "Region {} contains the following objects:\n".format(region_full_name)
    for obj in region.objects:
        description += f"{obj.class_name}_{obj.id}; "
    description += "\n"
        
    return description

def generate_objects_description(sim, object_layer):
    """
    Generate description of objects, used when region layer is empty 
    """
    description = "There are {} objects in the scene.\n".format(len(object_layer.obj_ids))
    for obj_id in object_layer.obj_ids:
        obj = object_layer.obj_dict[obj_id]

        if obj.full_name in sim._handle_to_goal_name:
            obj_name = sim._handle_to_goal_name[obj.full_name]
        elif obj.label:
            obj_name = obj.label
        if obj.full_name is None:
            obj_name = "any_targets|" + str(obj_id)
        else:
            obj_name = obj.full_name
        
        position = np.array(obj.center)
        # Remove quotes in list 
        # position_str = ', '.join([f'{p:.1f}' for p in position])
        # description += f'Object "{obj_name}" is at the position [{position_str}]. '
        # add height and distance information
        description += f'The height of "{obj_name}" is {position[1]:.1f}. '
        island_areas = [
            (island_ix, sim.pathfinder.island_area(island_index=island_ix))
            for island_ix in range(sim.pathfinder.num_islands)
        ]
        horizontal_dist = np.inf
        for nav_island in range(len(island_areas)):
            snap_pos = sim.pathfinder.snap_point(obj.center, nav_island)
            if not np.isnan(snap_pos).any():
                horizontal_dist = np.linalg.norm(
                    np.array(snap_pos)[[0, 2]] - np.array(obj.center)[[0, 2]]
                )
                break
        description += f'The horizontal distance of "{obj_name}" to the nearest navigable point is {horizontal_dist:.1f}. '
        if horizontal_dist == np.inf:
            description += f'"inf" means the robot cannot navigate to a position near "{obj_name}".\n'
        else:
            description += "\n"

    return description

def generate_mp3d_objects_description(object_layer):
    """
    Generate description of objects, used when region layer is not empty
    """
    description = "There are {} objects in the scene.\n".format(len(object_layer.obj_ids))
    for obj_id in object_layer.obj_ids:
        obj = object_layer.obj_dict[obj_id]
        assert obj.parent_region, f"{obj.full_name} has no parent region"
        parent_region = obj.parent_region
        region_name = parent_region.class_name + "_" + str(parent_region.region_id)
        level = parent_region.parent_level
        obj_name = obj.label
        obj_height = obj.height
        if not obj.label:
            obj_name = obj.full_name

        # Remove quotes in list
        # position = np.array(obj.center)
        # position_str = ', '.join([f'{p:.1f}' for p in position])
        description += (
            f'The object "{obj_name}" is located in {region_name} on {level} floor. '
            f'The height of object "{obj_name}" from the floor is {obj_height:.1f}. \n'
        )

    return description

def generate_agents_description(agent_layer, region_layer, nav_mesh):
    agent_description = ""
    
    if len(region_layer) == 0 or nav_mesh is None:
        #  when no region layer, just list the agent positions
        agent_description = "There are {} agents in the scene.\n".format(len(agent_layer.agent_ids))
        for agent_id in agent_layer.agent_ids:
            agent = agent_layer.agent_dict[agent_id]
            agent_name = agent.agent_name
            agent_pos = agent.position
            agent_pos_str = ', '.join([f'{p:.1f}' for p in agent_pos])
            # agent_description += f'"{agent_name}" is at position [{agent_pos_str}].\n'

    else:
        agents_region_ids = agent_layer.get_agents_region_ids(nav_mesh)
        for agent_id, region_id in agents_region_ids.items():
            agent_name = agent_layer.agent_dict[agent_id].agent_name
            region_name = region_layer.region_dict[region_id].class_name + "_" + str(region_id)
            agent_description += f'"{agent_name}" is in {region_name}.\n'
    
    return agent_description

def generate_mp3d_agents_description(agent_layer, region_layer):
    agent_description = "There are {} agents in the scene.\n".format(len(agent_layer.agent_ids))
    for agent_id in agent_layer.agent_ids:
        agent = agent_layer.agent_dict[agent_id]
        agent_name = agent.agent_name
        agent_pos = agent.position

        find_agent_in_region = False
        for region_id in region_layer.region_ids:
            region_node = region_layer.region_dict[region_id]
            region_bb = region_node.bbox

            if np.all(agent_pos >= region_bb[0]) and np.all(agent_pos <= region_bb[1]):
                region_name = region_node.class_name + "_" + str(region_id)
                parent_level = region_node.parent_level
                find_agent_in_region = True
                break
        
        agent_pos_str = ', '.join([f'{p:.1f}' for p in agent_pos])
        if not find_agent_in_region:
            agent_description += f'"{agent_name}" is at position [{agent_pos_str}].\n'
        else:
            agent_description += f'"{agent_name}" is in {region_name} on {parent_level} floor.\n'
        
    return agent_description


############ Visualization ############################

def visualize_scene_graph(
    scene_graph,
    scene_o3d=None,
    vis_coords=False,
    vis_region_bbox=False,
    house_file_vis=False,
    house_file_path=None,
    vis_object_bbox=True,
    bbox_color=(0, 1, 0),
    vis_navmesh=False,
    navmesh_shift=[0, 0, 0],
    vis_region_graph=False,
    region_graph_shift=[0, 0, 0],
    save_file=None,
    use_o3d_bbox=False,
    mp3d_coord=True,
):

    o3d_vis_list = []
    if vis_coords:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5, origin=[0, 0, 0]
        )
        o3d_vis_list.append(axis_pcd)

    # transformation: from habitat (-y) to mp3d (-z)
    quat_alignGravity = quat_from_two_vectors(
        np.array([0, 0, -1]), np.array([0, -1, 0])
    )
    r_m2h = qt.as_rotation_matrix(quat_alignGravity)
    r_h2m = np.linalg.inv(r_m2h)

    # render scene geometry
    if scene_o3d is not None:
        if not mp3d_coord:
            scene_o3d.rotate(r_m2h, (0, 0, 0))
        o3d_vis_list.append(scene_o3d)

    # render region bounding box
    if vis_region_bbox:
        for region_id in scene_graph.region_layer.region_ids:
            region_node = scene_graph.region_layer.region_dict[region_id]
            r_mat = R.from_quat([0,0,0,1]).as_matrix()
            region_center = region_node.bbox.mean(axis=0)
            region_size = region_node.bbox[1] - region_node.bbox[0]
            region_bbox = o3d.geometry.OrientedBoundingBox(
                region_center, r_mat, region_size
            )
            if mp3d_coord:
                region_bbox.rotate(r_h2m, (0, 0, 0))
            region_bbox.color = (1, 0, 0)
            o3d_vis_list.append(region_bbox)

    # render object bounding box
    if vis_object_bbox:
        if use_o3d_bbox:
            for obj_id in scene_graph.object_layer.obj_ids:
                obj_node = scene_graph.object_layer.obj_dict[obj_id]
                try:
                    o3d_pcl = o3d.geometry.PointCloud()
                    o3d_pcl.points = o3d.utility.Vector3dVector(
                        obj_node.vertices
                    )
                    obb = o3d_pcl.get_oriented_bounding_box()
                    if mp3d_coord:
                        obb.rotate(r_h2m)

                    obb.color = bbox_color
                    o3d_vis_list.append(obb)
                except:
                    print(
                        f"WARNING: object with id{obj_id}, class name {obj_node.class_name} does not have point clouds"
                    )
        else:
            for obj_id in scene_graph.object_layer.obj_ids:
                obj_node = scene_graph.object_layer.obj_dict[obj_id]
                r_mat = R.from_quat(obj_node.rotation).as_matrix()
                obb = o3d.geometry.OrientedBoundingBox(
                    obj_node.center, r_mat, obj_node.size
                )
                if mp3d_coord:
                    obb.rotate(r_h2m, (0, 0, 0))

                obb.color = bbox_color
                o3d_vis_list.append(obb)

    ############### visualize gt free space segmentation ###############
    if vis_navmesh:
        if scene_graph.nav_mesh is not None:
            mesh = scene_graph.nav_mesh.mesh
            # deep copy and rotate
            mesh = copy.deepcopy(mesh)
            if mp3d_coord:
                mesh.rotate(r_h2m, (0, 0, 0))
                
            mesh.translate(navmesh_shift)
            o3d_vis_list.append(mesh)

    ############# visualize region graph ##################
    if vis_region_graph:
        # Add nodes as bbox to the scene
        for region_node in scene_graph.region_layer.nodes:
            bbox_center = region_node.bbox.mean(axis=0)
            if mp3d_coord:
                bbox_center = bbox_center @ r_h2m.T
            
            
            bbox_size = np.array([0.1, 0.1, 0.1])
            bbox = o3d.geometry.OrientedBoundingBox(
                bbox_center+np.array(region_graph_shift), np.eye(3), bbox_size
            )
            bbox.color = (0, 0, 1)
            o3d_vis_list.append(bbox)
        
        # Add edges as lines to the scene
        for edge in scene_graph.region_layer.edges:
            node1 = edge[0]
            node2 = edge[1]
            line = o3d.geometry.LineSet()
            if mp3d_coord:
                start_point = node1.bbox.mean(axis=0) @ r_h2m.T + np.array(region_graph_shift)
                end_point = node2.bbox.mean(axis=0) @ r_h2m.T + np.array(region_graph_shift)
            else:
                start_point = node1.bbox.mean(axis=0) + np.array(region_graph_shift)
                end_point = node2.bbox.mean(axis=0) + np.array(region_graph_shift)
                
            line.points = o3d.utility.Vector3dVector([start_point, end_point])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
            o3d_vis_list.append(line)


    o3d.visualization.draw_geometries(o3d_vis_list, mesh_show_back_face=True)
    # o3d.visualization.draw_geometries(o3d_vis_list,
    #                                 zoom=0.455,
    #                                 front=[0.4999, 0.1659, 0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])

    # if save_file:
    #     o3d.io.write_point_cloud(save_file, scene_o3d, write_ascii=True)


def visualize_point_clouds(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    o3d.visualization.draw_geometries([pcd])


def display_image_arr(sizes, img_arr, block=True):
    row, col = sizes
    f, axarr = plt.subplots(row, col)
    if row * col > 1:
        for i in range(row):
            for j in range(col):
                pos = axarr[i, j].imshow(img_arr[i * col + j])
                f.colorbar(pos, ax=axarr[i, j])
    else:
        pos = axarr.imshow(img_arr[0])
        f.colorbar(pos, ax=axarr)
    plt.show(block=block)


def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed on the entire row.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.0
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def inNd(a, b, assume_unique=False):
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)


if __name__ == "__main__":
    # test get_corners()
    center = np.array([0, 0, 0])
    size = np.array([1, 1, 1])
    quat_1 = np.array([0, 0, 0, 1])
    print(get_corners(center, size, quat_1))
    quat_2 = np.array([1, 0, 0, 1])
    print(get_corners(center, size, quat_2))

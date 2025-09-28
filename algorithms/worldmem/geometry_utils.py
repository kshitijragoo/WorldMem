# algorithms/worldmem/geometry_utils.py

"""
Geometry utilities that import from VMem's original implementation.
This avoids code duplication and ensures consistency with VMem's methods.
"""

import os
import sys

# Add vmem to path
vmem_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vmem"))
if vmem_path not in sys.path:
    sys.path.insert(0, vmem_path)

# Import geometry utilities from VMem
from utils.util import (
    get_default_intrinsics,
    load_img_and_K, 
    transform_img_and_K,
    tensor_to_pil,
    average_camera_pose,
    geodesic_distance,
    inverse_geodesic_distance,
    Surfel,
    Octree,
    visualize_surfels,
    visualize_pointcloud,
    visualize_depth,
    unproject_depth_map_to_point_map
)

# Import VGGT geometry utilities
from vggt.utils.geometry import (
    unproject_depth_map_to_point_map as vggt_unproject,
    depth_to_world_coords_points,
    depth_to_cam_coords_points,
    project_world_points_to_camera_points_batch,
    project_world_points_to_cam,
    img_from_cam,
    cam_from_img
)

# Re-export commonly used functions for backward compatibility
__all__ = [
    'get_default_intrinsics',
    'load_img_and_K',
    'transform_img_and_K', 
    'tensor_to_pil',
    'average_camera_pose',
    'geodesic_distance',
    'inverse_geodesic_distance',
    'Surfel',
    'Octree',
    'visualize_surfels',
    'visualize_pointcloud',
    'visualize_depth',
    'unproject_depth_map_to_point_map',
    'vggt_unproject',
    'depth_to_world_coords_points',
    'depth_to_cam_coords_points',
    'project_world_points_to_camera_points_batch',
    'project_world_points_to_cam',
    'img_from_cam',
    'cam_from_img'
]

# Convenience functions that combine VMem and VGGT functionality
def unproject_depth_to_pointcloud(depth_map, pose_enc, image_size_hw):
    """
    Unprojects a depth map to a 3D point cloud using VGGT's camera outputs.
    This uses VGGT's original implementation for consistency.
    """
    return vggt_unproject(depth_map, pose_enc, image_size_hw)

def pointcloud_to_surfels(pointcloud, camera_params, downsample_factor=4, alpha=0.2):
    """
    Converts a dense point cloud into surfels using VMem's approach.
    This is now handled internally by VMem's pipeline.
    """
    # This functionality is now integrated into VMem's pipeline
    # Users should use VMemAdapter.pipeline methods instead
    raise NotImplementedError(
        "This function is now handled by VMem's pipeline. "
        "Use VMemAdapter.pipeline.pointmap_to_surfels() instead."
    )
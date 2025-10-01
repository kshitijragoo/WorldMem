# worldmem/algorithms/worldmem/geometry_utils.py

import torch
import numpy as np


# Import core geometric functions directly from the vggt library
# This ensures that all transformations are consistent with the VGGT model's native conventions.
from vggt.utils.rotation import quat_to_mat, mat_to_quat
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import (
    unproject_depth_map_to_point_map,
    project_world_points_to_cam,
    depth_to_world_coords_points,
)

def get_camera_matrices_from_pose_encoding(pose_encoding, image_size_hw):
    """
    Wrapper function to decode a 9D pose vector into extrinsic and intrinsic matrices.

    Args:
        pose_encoding (torch.Tensor): VGGT's 9D pose encoding.
                                      Shape can be (B, S, 9) or (S, 9) or (9,).
        image_size_hw (tuple): Tuple of (height, width) of the image.

    Returns:
        tuple: (extrinsics, intrinsics) as torch.Tensors.
    """
    # Ensure at least 2 dimensions (Sequence, Dims) for the library function
    original_shape = pose_encoding.shape
    if pose_encoding.dim() == 1:
        pose_encoding = pose_encoding.unsqueeze(0)
    if pose_encoding.dim() == 2:
        pose_encoding = pose_encoding.unsqueeze(0) # Add a batch dimension

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding,
        image_size_hw=image_size_hw,
        pose_encoding_type="absT_quaR_FoV"
    )

    # Restore original batch/sequence dimensions
    if len(original_shape) == 1:
        extrinsics = extrinsics.squeeze(0).squeeze(0)
        intrinsics = intrinsics.squeeze(0).squeeze(0)
    elif len(original_shape) == 2:
        extrinsics = extrinsics.squeeze(0)
        intrinsics = intrinsics.squeeze(0)
        
    return extrinsics, intrinsics

def unproject_depth_to_points(depth_map, pose_encoding, image_size_hw):
    """
    Unprojects a depth map to a 3D point map in the world coordinate frame.
    This function is critical for creating the surfel memory bank and uses the principle
    that unprojecting from VGGT's predicted depth and camera is more accurate than
    using its direct point map head.

    Args:
        depth_map (torch.Tensor): The depth map of shape (H, W) or (S, H, W).
        pose_encoding (torch.Tensor): The 9D pose encoding for the camera(s).
        image_size_hw (tuple): The (height, width) of the image.

    Returns:
        torch.Tensor: The 3D point map in world coordinates, shape (S, H, W, 3).
    """
    extrinsics, intrinsics = get_camera_matrices_from_pose_encoding(pose_encoding, image_size_hw)

    # The library function expects numpy arrays.
    depth_np = depth_map.cpu().numpy()
    extrinsics_np = extrinsics.cpu().numpy()
    intrinsics_np = intrinsics.cpu().numpy()

    # Ensure batch dimension for the library function
    if depth_np.ndim == 2:
        depth_np = depth_np[np.newaxis, :, :]
    if extrinsics_np.ndim == 2:
         extrinsics_np = extrinsics_np[np.newaxis, :, :]
    if intrinsics_np.ndim == 2:
         intrinsics_np = intrinsics_np[np.newaxis, :, :]


    # Call the validated library function for unprojection.
    world_points_np = unproject_depth_map_to_point_map(
        depth_map=depth_np,
        extrinsics_cam=extrinsics_np,
        intrinsics_cam=intrinsics_np
    )

    return torch.from_numpy(world_points_np).to(depth_map.device)

def project_points_to_camera(world_points, pose_encoding, image_size_hw):
    """
    Projects 3D world points into a camera's 2D image plane and returns their
    camera-space coordinates.

    Args:
        world_points (torch.Tensor): 3D points of shape (N, 3).
        pose_encoding (torch.Tensor): The 9D pose encoding for the target camera.
        image_size_hw (tuple): The (height, width) of the image.

    Returns:
        tuple:
            - torch.Tensor: 2D pixel coordinates of shape (N, 2).
            - torch.Tensor: 3D camera-space coordinates of shape (N, 3).
    """
    extrinsics, intrinsics = get_camera_matrices_from_pose_encoding(pose_encoding, image_size_hw)

    # The library function expects batch dimension for camera parameters
    if extrinsics.dim() == 2:
        extrinsics = extrinsics.unsqueeze(0)
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)

    # Call the validated library function for projection.
    image_points, cam_points = project_world_points_to_cam(
        world_points=world_points,
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics
    )
    
    # cam_points is (B, 3, N), transpose to (B, N, 3) for easier indexing
    cam_points = cam_points.transpose(1, 2)

    # Squeeze batch dimension from result
    return image_points.squeeze(0), cam_points.squeeze(0)

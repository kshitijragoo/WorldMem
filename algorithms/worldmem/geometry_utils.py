# algorithms/worldmem/geometry_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def unproject_depth_to_pointcloud(depth_map, pose_enc, image_size_hw):
    """
    Unprojects a depth map to a 3D point cloud using VGGT's camera outputs.
    This is the more accurate method for geometry acquisition, as recommended by the VGGT paper.[1, 1]

    Args:
        depth_map (torch.Tensor): Depth map of shape (B, 1, H, W).
        pose_enc (torch.Tensor): VGGT's 9D pose encoding of shape (B, 9).
        image_size_hw (tuple): The (height, width) of the image.

    Returns:
        torch.Tensor: Point cloud of shape (B, H, W, 3) in world coordinates.
    """
    B, _, H, W = depth_map.shape
    device = depth_map.device

    # Get camera extrinsics (world-to-camera) and intrinsics from VGGT's output [2, 1]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
    
    # Create a grid of pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).expand(B, -1, -1, -1) # Shape: (B, H, W, 3)

    # Unproject: 2D pixel -> 3D camera coordinates
    # P_cam = K_inv * [u, v, 1] * Z [3, 1, 4]
    cam_coords = torch.matmul(torch.linalg.inv(intrinsics).unsqueeze(1).unsqueeze(1), pixels.unsqueeze(-1)).squeeze(-1)
    cam_coords = cam_coords * depth_map.permute(0, 2, 3, 1) # Shape: (B, H, W, 3)

    # Transform: 3D camera coordinates -> 3D world coordinates
    # P_world = R_inv * (P_cam - t)
    R_inv = torch.linalg.inv(extrinsics[:, :3, :3])
    t = extrinsics[:, :3, 3]
    world_coords = torch.matmul(R_inv.unsqueeze(1).unsqueeze(1), (cam_coords - t.view(B, 1, 1, 3)).unsqueeze(-1)).squeeze(-1)

    return world_coords

def pointcloud_to_surfels(pointcloud, camera_params, downsample_factor=4, alpha=0.2):
    """
    Converts a dense point cloud into a set of surfels, calculating position, normal, and radius.[1, 1]

    Args:
        pointcloud (torch.Tensor): Point cloud of shape (H, W, 3).
        camera_params (dict): Dictionary containing 'extrinsics' and 'intrinsics'.
        downsample_factor (int): Factor to downsample the point cloud for efficiency.
        alpha (float): Constant for radius calculation heuristic from VMem.[1]

    Returns:
        tuple: A tuple of (positions, normals, radii) for the surfels.
    """
    # 1. Downsample the point cloud
    points_down = pointcloud[::downsample_factor, ::downsample_factor, :]
    
    # 2. Calculate normals using the grid structure for efficiency [5, 1]
    # This avoids a costly k-NN search [6, 7]
    dx = points_down[1:-1, 2:, :] - points_down[1:-1, :-2, :]
    dy = points_down[2:, 1:-1, :] - points_down[:-2, 1:-1, :]
    normals = F.normalize(torch.cross(dx[:, :-2], dy[:-2, :], dim=-1), p=2, dim=-1)
    
    # Pad to match original downsampled dimensions
    positions = points_down[1:-1, 1:-1, :]
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1))

    # 3. Calculate radii using the VMem heuristic [1, 1]
    extrinsics = camera_params['extrinsics'] # Assuming batch size of 1 for this operation
    intrinsics = camera_params['intrinsics']
    cam_center = -torch.matmul(extrinsics[:3, :3].T, extrinsics[:3, 3])
    focal_length = (intrinsics + intrinsics[1, 1]) / 2
    
    view_dirs = F.normalize(positions - cam_center.view(1, 1, 3), p=2, dim=-1)
    depths = torch.linalg.norm(positions - cam_center.view(1, 1, 3), dim=-1)
    
    cos_angle = torch.sum(normals * view_dirs, dim=-1).abs()
    radii = (0.5 * depths / focal_length) / (alpha + (1 - alpha) * cos_angle)

    return positions.reshape(-1, 3), normals.reshape(-1, 3), radii.reshape(-1)
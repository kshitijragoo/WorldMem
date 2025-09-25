"""
Memory Visualization Module for VGGT Memory Retriever

This module provides functionality to export the VGGT memory retriever's 
surfel-based world representation as 3D files (.glb, .ply) for visualization
and debugging purposes.
"""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple
import colorsys
import matplotlib


class VGGTMemoryVisualizer:
    """
    Visualizer for VGGT Memory Retriever's surfel-based world representation.
    """
    
    def __init__(self, memory_retriever):
        """
        Initialize the visualizer with a VGGT memory retriever.
        
        Args:
            memory_retriever: VGGTMemoryRetriever instance
        """
        self.memory_retriever = memory_retriever
        
    def export_world_as_glb(self, output_path: str, 
                           include_cameras: bool = True,
                           include_surfels: bool = True,
                           surfel_scale: float = 1.0,
                           camera_scale: float = 0.5,
                           include_axes: bool = True) -> None:
        """
        Export the current world representation as a GLB file.
        
        Args:
            output_path: Path to save the GLB file
            include_cameras: Whether to include camera positions
            include_surfels: Whether to include surfel points
            surfel_scale: Scale factor for surfel visualization
            camera_scale: Scale factor for camera frustums
        """
        # Debug: Print memory retriever state
        print(f"=== MEMORY RETRIEVER DEBUG ===")
        print(f"Surfels: {self.memory_retriever.surfels}")
        print(f"View database length: {len(self.memory_retriever.view_database)}")
        print(f"Surfel to views length: {len(self.memory_retriever.surfel_to_views)}")
        
        if self.memory_retriever.surfels is not None:
            print(f"Surfel positions shape: {self.memory_retriever.surfels['pos'].shape}")
            print(f"Surfel normals shape: {self.memory_retriever.surfels['norm'].shape}")
            print(f"Surfel radii shape: {self.memory_retriever.surfels['rad'].shape}")
        print(f"==============================")
        
        # Always create a scene, even if no surfels
        scene = trimesh.Scene()
        
        # Add coordinate axes for reference
        if include_axes:
            axes_meshes = self._create_coordinate_axes()
            for i, mesh in enumerate(axes_meshes):
                scene.add_geometry(mesh, node_name=f"axis_{i}")
        
        # Add surfels as colored shapes
        if include_surfels:
            if self.memory_retriever.surfels is not None:
                print(f"Creating surfel meshes for {len(self.memory_retriever.surfels['pos'])} surfels")
                surfel_meshes = self._create_surfel_meshes(surfel_scale)
                print(f"Generated {len(surfel_meshes)} surfel meshes")
                for i, mesh in enumerate(surfel_meshes):
                    if hasattr(mesh, 'vertices'):  # Check if it's a proper mesh
                        scene.add_geometry(mesh, node_name=f"surfel_{i}")
                        print(f"Added surfel mesh {i} with {len(mesh.vertices)} vertices")
            else:
                print("No surfels available - creating enhanced placeholder visualization")
                # Create enhanced placeholder visualization
                placeholder_viz = self._create_enhanced_placeholder_visualization()
                if placeholder_viz is not None:
                    scene.add_geometry(placeholder_viz, node_name="placeholder_points")
                    print("Added enhanced placeholder visualization")
        
        # Add camera positions and orientations
        if include_cameras:
            camera_meshes = self._create_camera_meshes(camera_scale)
            for i, mesh in enumerate(camera_meshes):
                scene.add_geometry(mesh, node_name=f"camera_{i}")
        
        # Add scene statistics as a text note in the GLB metadata
        if self.memory_retriever.surfels is not None:
            scene.metadata = {
                'total_surfels': len(self.memory_retriever.surfels['pos']),
                'total_cameras': len(self.memory_retriever.view_database),
                'scene_bounds': self._get_scene_bounds()
            }
        
        # Skip VGGT reconstruction for now due to performance issues
        # Instead, focus on surfel and trajectory visualization
        print("Skipping VGGT reconstruction to avoid hanging - using surfel data instead")
        
        # Try enhanced surfel visualization first
        enhanced_surfel_cloud = self._create_enhanced_surfel_cloud()
        if enhanced_surfel_cloud is not None:
            scene.add_geometry(enhanced_surfel_cloud, node_name="enhanced_surfels")
            print("Added enhanced surfel point cloud")
        else:
            # Create trajectory visualization as fallback
            print("Creating trajectory visualization from camera positions")
            trajectory_viz = self._create_trajectory_visualization()
            if trajectory_viz is not None:
                scene.add_geometry(trajectory_viz, node_name="camera_trajectory")
                print("Added camera trajectory visualization")
        
        # Export as GLB
        scene.export(output_path)
        print(f"World representation exported to: {output_path}")
        print(f"Scene contains {len(scene.geometry)} objects")
        
    def export_world_as_ply(self, output_path: str) -> None:
        """
        Export surfels as a PLY point cloud file.
        
        Args:
            output_path: Path to save the PLY file
        """
        if self.memory_retriever.surfels is None:
            print("No surfels in memory to export.")
            return
            
        positions = self.memory_retriever.surfels['pos'].cpu().numpy()
        normals = self.memory_retriever.surfels['norm'].cpu().numpy()
        radii = self.memory_retriever.surfels['rad'].cpu().numpy()
        
        # Create colors based on view associations
        colors = self._generate_surfel_colors()
        
        # Create point cloud
        point_cloud = trimesh.PointCloud(
            vertices=positions,
            colors=colors
        )
        
        # Export as PLY
        point_cloud.export(output_path)
        print(f"Point cloud exported to: {output_path}")
        
    def export_memory_analysis(self, output_path: str) -> None:
        """
        Export detailed analysis of the memory state as JSON.
        
        Args:
            output_path: Path to save the JSON file
        """
        if self.memory_retriever.surfels is None:
            print("No memory data to analyze.")
            return
            
        analysis = {
            "total_surfels": len(self.memory_retriever.surfels['pos']),
            "total_views": len(self.memory_retriever.view_database),
            "surfel_statistics": self._compute_surfel_statistics(),
            "view_associations": self._analyze_view_associations(),
            "spatial_distribution": self._analyze_spatial_distribution(),
            "memory_coverage": self._analyze_memory_coverage()
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=self._json_serializer)
            
        print(f"Memory analysis exported to: {output_path}")
        
    def visualize_retrieval_for_pose(self, target_c2w: torch.Tensor, 
                                   k: int = 4,
                                   output_path: str = None) -> Dict:
        """
        Visualize which views would be retrieved for a given target pose.
        
        Args:
            target_c2w: Target camera-to-world matrix
            k: Number of views to retrieve
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary with retrieval information
        """
        # Get retrieval results
        retrieved_indices = self.memory_retriever.retrieve_relevant_views(
            target_c2w, k=k
        )
        
        # Create visualization
        scene = trimesh.Scene()
        
        # Add all surfels (dimmed)
        if self.memory_retriever.surfels is not None:
            surfel_meshes = self._create_surfel_meshes(scale=0.5, alpha=0.3)
            for i, mesh in enumerate(surfel_meshes):
                scene.add_geometry(mesh, node_name=f"surfel_{i}")
        
        # Add target camera (bright red)
        target_mesh = self._create_camera_mesh(target_c2w, color=[1, 0, 0, 1], scale=1.0)
        scene.add_geometry(target_mesh, node_name="target_camera")
        
        # Add retrieved cameras (bright green)
        for i, view_idx in enumerate(retrieved_indices):
            if view_idx < len(self.memory_retriever.view_database):
                _, c2w = self.memory_retriever.view_database[view_idx]
                retrieved_mesh = self._create_camera_mesh(c2w, color=[0, 1, 0, 1], scale=0.8)
                scene.add_geometry(retrieved_mesh, node_name=f"retrieved_camera_{i}")
        
        # Add other cameras (dimmed blue)
        for view_idx in range(len(self.memory_retriever.view_database)):
            if view_idx not in retrieved_indices:
                _, c2w = self.memory_retriever.view_database[view_idx]
                other_mesh = self._create_camera_mesh(c2w, color=[0, 0, 1, 0.3], scale=0.4)
                scene.add_geometry(other_mesh, node_name=f"other_camera_{view_idx}")
        
        if output_path:
            scene.export(output_path)
            print(f"Retrieval visualization saved to: {output_path}")
        
        return {
            "target_pose": target_c2w.cpu().numpy().tolist(),
            "retrieved_indices": retrieved_indices,
            "total_views": len(self.memory_retriever.view_database),
            "total_surfels": len(self.memory_retriever.surfels['pos']) if self.memory_retriever.surfels else 0
        }
    
    def _create_surfel_meshes(self, scale: float = 1.0, alpha: float = 1.0) -> List[trimesh.Trimesh]:
        """Create mesh representations of surfels with better visualization."""
        positions = self.memory_retriever.surfels['pos'].cpu().numpy()
        normals = self.memory_retriever.surfels['norm'].cpu().numpy()
        radii = self.memory_retriever.surfels['rad'].cpu().numpy()
        
        colors = self._generate_surfel_colors(alpha=alpha)
        
        meshes = []
        
        # Create a single point cloud instead of individual spheres for better performance
        if len(positions) > 1000:
            # For large numbers of surfels, use a point cloud approach
            point_cloud = trimesh.PointCloud(
                vertices=positions,
                colors=colors[:, :3]  # RGB only
            )
            return [point_cloud]
        
        # For smaller numbers, create oriented discs instead of spheres
        for i in range(len(positions)):
            # Create a small disc oriented along the normal
            if np.linalg.norm(normals[i]) > 0:
                # Create a disc (flattened cylinder)
                disc = trimesh.creation.cylinder(
                    radius=max(radii[i] * scale, 0.01), 
                    height=max(radii[i] * scale * 0.1, 0.001),
                    sections=8
                )
                
                # Orient the disc along the normal
                normal = normals[i] / np.linalg.norm(normals[i])
                z_axis = np.array([0, 0, 1])
                
                if not np.allclose(normal, z_axis):
                    rotation_axis = np.cross(z_axis, normal)
                    if np.linalg.norm(rotation_axis) > 1e-6:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        rotation_angle = np.arccos(np.clip(np.dot(z_axis, normal), -1, 1))
                        rotation_matrix = trimesh.transformations.rotation_matrix(
                            rotation_angle, rotation_axis
                        )
                        disc.apply_transform(rotation_matrix)
                
                disc.apply_translation(positions[i])
                disc.visual.face_colors = colors[i]
                meshes.append(disc)
            else:
                # Fallback to small sphere if normal is zero
                sphere = trimesh.creation.icosphere(radius=max(radii[i] * scale, 0.01), subdivisions=1)
                sphere.apply_translation(positions[i])
                sphere.visual.face_colors = colors[i]
                meshes.append(sphere)
            
        return meshes
    
    def _create_camera_meshes(self, scale: float = 1.0) -> List[trimesh.Trimesh]:
        """Create mesh representations of camera poses."""
        meshes = []
        
        for i, (_, c2w) in enumerate(self.memory_retriever.view_database):
            # Generate color based on view index
            hue = (i * 137.5) % 360  # Golden angle for good color distribution
            color = list(colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)) + [1.0]
            
            mesh = self._create_camera_mesh(c2w, color, scale)
            meshes.append(mesh)
            
        return meshes
    
    def _create_camera_mesh(self, c2w: torch.Tensor, color: List[float], scale: float = 1.0) -> trimesh.Trimesh:
        """Create a single camera frustum mesh with better visibility."""
        # Convert to numpy if needed
        if isinstance(c2w, torch.Tensor):
            c2w = c2w.cpu().numpy()
        
        # Create a more visible camera representation
        center = c2w[:3, 3]
        
        # Camera coordinate system (make it larger and more visible)
        right = c2w[:3, 0] * scale * 0.3
        up = c2w[:3, 1] * scale * 0.3
        forward = -c2w[:3, 2] * scale * 0.5  # Negative Z is forward in camera coords
        
        # Create a camera body (small box at center)
        body_size = scale * 0.05
        body_vertices = np.array([
            center + np.array([-body_size, -body_size, -body_size]),
            center + np.array([body_size, -body_size, -body_size]),
            center + np.array([body_size, body_size, -body_size]),
            center + np.array([-body_size, body_size, -body_size]),
            center + np.array([-body_size, -body_size, body_size]),
            center + np.array([body_size, -body_size, body_size]),
            center + np.array([body_size, body_size, body_size]),
            center + np.array([-body_size, body_size, body_size]),
        ])
        
        # Create frustum vertices
        frustum_vertices = np.array([
            center,  # Camera center
            center + forward + right * 0.5 + up * 0.5,    # Top-right-far
            center + forward - right * 0.5 + up * 0.5,    # Top-left-far
            center + forward - right * 0.5 - up * 0.5,    # Bottom-left-far
            center + forward + right * 0.5 - up * 0.5,    # Bottom-right-far
        ])
        
        # Combine all vertices
        all_vertices = np.vstack([body_vertices, frustum_vertices])
        
        # Define faces for the camera body (cube)
        body_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Front face
            [4, 7, 6], [4, 6, 5],  # Back face
            [0, 4, 5], [0, 5, 1],  # Bottom face
            [2, 6, 7], [2, 7, 3],  # Top face
            [0, 3, 7], [0, 7, 4],  # Left face
            [1, 5, 6], [1, 6, 2],  # Right face
        ])
        
        # Define faces for the frustum (offset indices by 8 for body vertices)
        frustum_faces = np.array([
            [8, 9, 10],   # Top face
            [8, 10, 11],  # Left face
            [8, 11, 12],  # Bottom face
            [8, 12, 9],   # Right face
            [9, 10, 11],  # Far face (triangle 1)
            [9, 11, 12],  # Far face (triangle 2)
        ])
        
        # Combine all faces
        all_faces = np.vstack([body_faces, frustum_faces])
        
        mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        
        # Set colors - make camera body slightly different from frustum
        body_color = np.array(color)
        frustum_color = body_color * 0.7  # Darker frustum
        
        # Create face colors
        face_colors = np.zeros((len(all_faces), 4))
        face_colors[:len(body_faces)] = body_color  # Body faces
        face_colors[len(body_faces):] = frustum_color  # Frustum faces
        
        mesh.visual.face_colors = face_colors
        
        return mesh
    
    def _create_coordinate_axes(self, scale: float = 1.0) -> List[trimesh.Trimesh]:
        """Create coordinate axes for reference."""
        axes = []
        
        # X-axis (Red)
        x_axis = trimesh.creation.cylinder(radius=0.02 * scale, height=2.0 * scale)
        x_axis.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        x_axis.apply_translation([1.0 * scale, 0, 0])
        x_axis.visual.face_colors = [255, 0, 0, 255]  # Red
        axes.append(x_axis)
        
        # Y-axis (Green)
        y_axis = trimesh.creation.cylinder(radius=0.02 * scale, height=2.0 * scale)
        y_axis.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        y_axis.apply_translation([0, 1.0 * scale, 0])
        y_axis.visual.face_colors = [0, 255, 0, 255]  # Green
        axes.append(y_axis)
        
        # Z-axis (Blue)
        z_axis = trimesh.creation.cylinder(radius=0.02 * scale, height=2.0 * scale)
        z_axis.apply_translation([0, 0, 1.0 * scale])
        z_axis.visual.face_colors = [0, 0, 255, 255]  # Blue
        axes.append(z_axis)
        
        # Origin sphere
        origin = trimesh.creation.icosphere(radius=0.05 * scale)
        origin.visual.face_colors = [255, 255, 255, 255]  # White
        axes.append(origin)
        
        return axes
    
    def _get_scene_bounds(self) -> dict:
        """Get the bounding box of the scene."""
        if self.memory_retriever.surfels is None:
            return {}
        
        positions = self.memory_retriever.surfels['pos'].cpu().numpy()
        
        # Get camera positions too
        camera_positions = []
        for _, c2w in self.memory_retriever.view_database:
            if isinstance(c2w, torch.Tensor):
                camera_positions.append(c2w[:3, 3].cpu().numpy())
            else:
                camera_positions.append(c2w[:3, 3])
        
        if camera_positions:
            all_positions = np.vstack([positions, np.array(camera_positions)])
        else:
            all_positions = positions
        
        return {
            'min': all_positions.min(axis=0).tolist(),
            'max': all_positions.max(axis=0).tolist(),
            'center': all_positions.mean(axis=0).tolist(),
            'extent': (all_positions.max(axis=0) - all_positions.min(axis=0)).tolist()
        }
    
    def _generate_surfel_colors(self, alpha: float = 1.0) -> np.ndarray:
        """Generate colors for surfels based on their view associations and quality."""
        num_surfels = len(self.memory_retriever.surfels['pos'])
        colors = np.zeros((num_surfels, 4))
        
        # Get view count statistics for better color mapping
        view_counts = [len(self.memory_retriever.surfel_to_views[i]) for i in range(num_surfels)]
        max_views = max(view_counts) if view_counts else 1
        
        for i in range(num_surfels):
            num_views = len(self.memory_retriever.surfel_to_views[i])
            
            # Create a more nuanced color scheme
            if num_views == 1:
                # Single view: Bright red (uncertain)
                colors[i] = [1.0, 0.2, 0.2, alpha]
            elif num_views == 2:
                # Two views: Orange (low confidence)
                colors[i] = [1.0, 0.6, 0.0, alpha]
            elif num_views <= 4:
                # Few views: Yellow-green (moderate confidence)
                colors[i] = [0.8, 1.0, 0.2, alpha]
            elif num_views <= 6:
                # Good views: Green (high confidence)
                colors[i] = [0.2, 1.0, 0.2, alpha]
            else:
                # Many views: Blue-green (very high confidence)
                colors[i] = [0.0, 0.8, 1.0, alpha]
        
        # Convert to 0-255 range for trimesh
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        return colors_uint8
    
    def _create_vggt_style_reconstruction(self) -> Optional[trimesh.PointCloud]:
        """
        Create a VGGT-style point cloud reconstruction from the stored view database.
        This attempts to reconstruct the actual world geometry from the stored frames and poses.
        """
        if not self.memory_retriever.view_database or not hasattr(self.memory_retriever, 'vggt_model'):
            return None
            
        print("Creating VGGT-style reconstruction from stored views...")
        
        try:
            # Collect all stored frames and poses
            stored_frames = []
            stored_poses = []
            
            for frame_tensor, c2w_matrix in self.memory_retriever.view_database:
                # Convert frame to numpy if needed
                if isinstance(frame_tensor, torch.Tensor):
                    frame = frame_tensor.cpu().numpy()
                else:
                    frame = frame_tensor
                    
                # Convert pose to numpy if needed  
                if isinstance(c2w_matrix, torch.Tensor):
                    pose = c2w_matrix.cpu().numpy()
                else:
                    pose = c2w_matrix
                
                stored_frames.append(frame)
                stored_poses.append(pose)
            
            if not stored_frames:
                return None
                
            # Convert to numpy arrays
            frames_array = np.array(stored_frames)  # Shape: (N, C, H, W)
            poses_array = np.array(stored_poses)    # Shape: (N, 4, 4)
            
            # Use VGGT to get depth and 3D points for stored frames
            world_points_all = []
            colors_all = []
            
            for i, (frame, c2w) in enumerate(zip(frames_array, poses_array)):
                try:
                    # Prepare frame for VGGT (ensure correct format and size)
                    if frame.shape[0] == 3:  # CHW format
                        frame_tensor = torch.from_numpy(frame).to(self.memory_retriever.device)
                    else:  # HWC format, need to transpose
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(self.memory_retriever.device)
                    
                    # Resize to VGGT-compatible dimensions (multiple of 14)
                    h, w = frame_tensor.shape[-2:]
                    new_h = ((h + 13) // 14) * 14
                    new_w = ((w + 13) // 14) * 14
                    
                    frame_input = F.interpolate(
                        frame_tensor.unsqueeze(0), 
                        size=(new_h, new_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).to(self.memory_retriever.dtype)
                    
                    print(f"DEBUG: Frame {i} resized from {h}x{w} to {new_h}x{new_w}")
                    print(f"DEBUG: About to run VGGT inference for frame {i}")
                    
                    # Get VGGT predictions with timeout handling
                    try:
                        with torch.amp.autocast(device_type='cuda', dtype=self.memory_retriever.dtype):
                            predictions = self.memory_retriever.vggt_model(frame_input)
                        print(f"DEBUG: VGGT inference completed for frame {i}")
                    except Exception as e:
                        print(f"ERROR: VGGT inference failed for frame {i}: {e}")
                        continue  # Skip this frame
                    
                    depth = predictions["depth"].cpu().numpy()
                    pose_enc = predictions["pose_enc"].cpu().numpy()
                    
                    # Convert pose encoding to camera matrices
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    extrinsics, intrinsics = pose_encoding_to_extri_intri(
                        torch.from_numpy(pose_enc), 
                        (frame.shape[-2], frame.shape[-1])
                    )
                    
                    # Unproject depth to world coordinates
                    world_points = self._unproject_depth_to_world_points(
                        depth[0], extrinsics[0].numpy(), intrinsics[0].numpy()
                    )
                    
                    # Get frame colors
                    if frame.shape[0] == 3:  # CHW format
                        frame_colors = frame.transpose(1, 2, 0)  # Convert to HWC
                    else:
                        frame_colors = frame
                    
                    # Flatten points and colors
                    points_flat = world_points.reshape(-1, 3)
                    colors_flat = frame_colors.reshape(-1, 3)
                    
                    # Filter out invalid points (depth <= 0)
                    depth_flat = depth[0].flatten()
                    valid_mask = depth_flat > 0.01  # Minimum depth threshold
                    
                    if valid_mask.any():
                        world_points_all.append(points_flat[valid_mask])
                        colors_all.append((colors_flat[valid_mask] * 255).astype(np.uint8))
                    
                except Exception as e:
                    print(f"Error processing frame {i}: {e}")
                    continue
            
            if not world_points_all:
                print("No valid 3D points generated")
                return None
            
            # Combine all points and colors
            all_points = np.vstack(world_points_all)
            all_colors = np.vstack(colors_all)
            
            # Apply confidence-based filtering (keep top 80% of points by some quality metric)
            # For now, we'll use a simple distance-based filter
            if len(all_points) > 10000:  # Subsample if too many points
                center = np.mean(all_points, axis=0)
                distances = np.linalg.norm(all_points - center, axis=1)
                distance_threshold = np.percentile(distances, 90)  # Keep 90% of points
                keep_mask = distances <= distance_threshold
                all_points = all_points[keep_mask]
                all_colors = all_colors[keep_mask]
            
            print(f"Created point cloud with {len(all_points)} points")
            
            # Create trimesh PointCloud
            point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
            return point_cloud
            
        except Exception as e:
            print(f"Error creating VGGT-style reconstruction: {e}")
            return None
    
    def _unproject_depth_to_world_points(self, depth_map: np.ndarray, 
                                       extrinsics: np.ndarray, 
                                       intrinsics: np.ndarray) -> np.ndarray:
        """
        Unproject a depth map to 3D world coordinates using camera parameters.
        """
        H, W = depth_map.shape
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Get intrinsic parameters
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Unproject to camera coordinates
        x_cam = (u - cx) * depth_map / fx
        y_cam = (v - cy) * depth_map / fy
        z_cam = depth_map
        
        # Stack to form camera coordinates
        cam_coords = np.stack([x_cam, y_cam, z_cam], axis=-1)  # Shape: (H, W, 3)
        
        # Convert to homogeneous coordinates
        ones = np.ones((H, W, 1))
        cam_coords_homo = np.concatenate([cam_coords, ones], axis=-1)  # Shape: (H, W, 4)
        
        # Transform to world coordinates using inverse extrinsics
        # extrinsics is world-to-camera, so we need camera-to-world
        extrinsics_4x4 = np.eye(4)
        extrinsics_4x4[:3, :4] = extrinsics
        
        # Compute camera-to-world transformation
        cam_to_world = np.linalg.inv(extrinsics_4x4)
        
        # Apply transformation
        world_coords = np.dot(cam_coords_homo, cam_to_world.T)
        
        return world_coords[:, :, :3]  # Return only XYZ coordinates
    
    def _create_enhanced_surfel_cloud(self) -> Optional[trimesh.PointCloud]:
        """
        Create an enhanced point cloud from surfel data with better density and colors.
        """
        if self.memory_retriever.surfels is None:
            return None
            
        try:
            positions = self.memory_retriever.surfels['pos'].cpu().numpy()
            normals = self.memory_retriever.surfels['norm'].cpu().numpy()
            radii = self.memory_retriever.surfels['rad'].cpu().numpy()
            
            # Generate multiple points per surfel to create denser point cloud
            enhanced_points = []
            enhanced_colors = []
            
            surfel_colors = self._generate_surfel_colors()
            
            for i in range(len(positions)):
                pos = positions[i]
                normal = normals[i]
                radius = radii[i]
                color = surfel_colors[i]
                
                # Create multiple points around each surfel position
                # Number of points based on radius (larger surfels get more points)
                num_points = max(1, min(10, int(radius * 100)))  # 1-10 points per surfel
                
                if num_points == 1:
                    enhanced_points.append(pos)
                    enhanced_colors.append(color)
                else:
                    # Create points in a small sphere around the surfel
                    for _ in range(num_points):
                        # Random offset within the surfel radius
                        offset = np.random.randn(3) * radius * 0.5
                        # Project offset perpendicular to normal to stay on surface
                        if np.linalg.norm(normal) > 0:
                            normal_unit = normal / np.linalg.norm(normal)
                            offset = offset - np.dot(offset, normal_unit) * normal_unit
                        
                        enhanced_points.append(pos + offset)
                        enhanced_colors.append(color)
            
            if enhanced_points:
                enhanced_points = np.array(enhanced_points)
                enhanced_colors = np.array(enhanced_colors)
                
                print(f"Created enhanced surfel cloud with {len(enhanced_points)} points")
                return trimesh.PointCloud(vertices=enhanced_points, colors=enhanced_colors)
            
        except Exception as e:
            print(f"Error creating enhanced surfel cloud: {e}")
            
        return None
    
    def _create_trajectory_visualization(self) -> Optional[trimesh.PointCloud]:
        """
        Create a visualization showing the camera trajectory as a connected line with points.
        """
        if not self.memory_retriever.view_database:
            return None
            
        try:
            # Extract camera positions
            positions = []
            for _, c2w in self.memory_retriever.view_database:
                if isinstance(c2w, torch.Tensor):
                    pos = c2w[:3, 3].cpu().numpy()
                else:
                    pos = c2w[:3, 3]
                positions.append(pos)
            
            if len(positions) < 2:
                return None
                
            positions = np.array(positions)
            
            # Create points along the trajectory with interpolation for smoother visualization
            trajectory_points = []
            trajectory_colors = []
            
            # Add original camera positions
            for i, pos in enumerate(positions):
                trajectory_points.append(pos)
                # Color gradient from blue (start) to red (end)
                t = i / max(1, len(positions) - 1)
                color = [int(255 * t), int(255 * (1-t)), 100, 255]  # Red to blue gradient
                trajectory_colors.append(color)
            
            # Add interpolated points between cameras for smoother trajectory
            for i in range(len(positions) - 1):
                start_pos = positions[i]
                end_pos = positions[i + 1]
                
                # Add 5 intermediate points between each pair of cameras
                for j in range(1, 6):
                    t = j / 6.0
                    interp_pos = start_pos * (1 - t) + end_pos * t
                    trajectory_points.append(interp_pos)
                    
                    # Interpolate color as well
                    t_global = (i + t) / max(1, len(positions) - 1)
                    color = [int(255 * t_global), int(255 * (1-t_global)), 150, 255]
                    trajectory_colors.append(color)
            
            if trajectory_points:
                trajectory_points = np.array(trajectory_points)
                trajectory_colors = np.array(trajectory_colors)
                
                print(f"Created trajectory visualization with {len(trajectory_points)} points")
                return trimesh.PointCloud(vertices=trajectory_points, colors=trajectory_colors)
                
        except Exception as e:
            print(f"Error creating trajectory visualization: {e}")
            
        return None
    
    def _create_enhanced_placeholder_visualization(self) -> Optional[trimesh.PointCloud]:
        """
        Create an enhanced placeholder visualization when no surfels are available.
        This creates a more interesting visualization using camera positions and synthetic geometry.
        """
        if not self.memory_retriever.view_database:
            return None
            
        try:
            all_points = []
            all_colors = []
            
            # Extract camera positions and create a grid around each one
            for i, (_, c2w) in enumerate(self.memory_retriever.view_database):
                if isinstance(c2w, torch.Tensor):
                    cam_pos = c2w[:3, 3].cpu().numpy()
                    cam_forward = -c2w[:3, 2].cpu().numpy()  # -Z is forward
                    cam_right = c2w[:3, 0].cpu().numpy()
                    cam_up = c2w[:3, 1].cpu().numpy()
                else:
                    cam_pos = c2w[:3, 3]
                    cam_forward = -c2w[:3, 2]
                    cam_right = c2w[:3, 0]
                    cam_up = c2w[:3, 1]
                
                # Create a small grid of points in front of the camera to simulate scene geometry
                grid_size = 0.5
                grid_distance = 2.0
                grid_points_per_side = 5
                
                for x in range(-grid_points_per_side, grid_points_per_side + 1):
                    for y in range(-grid_points_per_side, grid_points_per_side + 1):
                        for z in range(1, 4):  # Points at different distances
                            # Calculate point position relative to camera
                            offset_x = (x / grid_points_per_side) * grid_size
                            offset_y = (y / grid_points_per_side) * grid_size
                            offset_z = z * grid_distance
                            
                            # Transform to world coordinates
                            world_point = (cam_pos + 
                                         offset_x * cam_right + 
                                         offset_y * cam_up + 
                                         offset_z * cam_forward)
                            
                            all_points.append(world_point)
                            
                            # Color based on distance and camera index
                            distance_factor = z / 3.0
                            camera_factor = i / max(1, len(self.memory_retriever.view_database) - 1)
                            
                            # Create a nice color gradient
                            r = int(255 * (1 - distance_factor) * (1 - camera_factor))
                            g = int(255 * distance_factor)
                            b = int(255 * camera_factor)
                            
                            all_colors.append([r, g, b, 255])
            
            # Add camera positions themselves as larger points
            for i, (_, c2w) in enumerate(self.memory_retriever.view_database):
                if isinstance(c2w, torch.Tensor):
                    pos = c2w[:3, 3].cpu().numpy()
                else:
                    pos = c2w[:3, 3]
                
                all_points.append(pos)
                all_colors.append([255, 255, 0, 255])  # Yellow for camera positions
            
            if all_points:
                all_points = np.array(all_points)
                all_colors = np.array(all_colors)
                
                print(f"Created enhanced placeholder with {len(all_points)} points")
                return trimesh.PointCloud(vertices=all_points, colors=all_colors)
                
        except Exception as e:
            print(f"Error creating enhanced placeholder: {e}")
            
        return None
    
    def _compute_surfel_statistics(self) -> Dict:
        """Compute statistics about surfels."""
        positions = self.memory_retriever.surfels['pos'].cpu().numpy()
        radii = self.memory_retriever.surfels['rad'].cpu().numpy()
        
        return {
            "count": len(positions),
            "position_bounds": {
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
                "mean": positions.mean(axis=0).tolist(),
                "std": positions.std(axis=0).tolist()
            },
            "radius_stats": {
                "min": float(radii.min()),
                "max": float(radii.max()),
                "mean": float(radii.mean()),
                "std": float(radii.std())
            }
        }
    
    def _analyze_view_associations(self) -> Dict:
        """Analyze how views are associated with surfels."""
        view_counts = [len(views) for views in self.memory_retriever.surfel_to_views]
        
        return {
            "min_views_per_surfel": min(view_counts) if view_counts else 0,
            "max_views_per_surfel": max(view_counts) if view_counts else 0,
            "mean_views_per_surfel": np.mean(view_counts) if view_counts else 0,
            "view_count_distribution": {
                "1_view": sum(1 for c in view_counts if c == 1),
                "2_views": sum(1 for c in view_counts if c == 2),
                "3_4_views": sum(1 for c in view_counts if 3 <= c <= 4),
                "5_plus_views": sum(1 for c in view_counts if c >= 5)
            }
        }
    
    def _analyze_spatial_distribution(self) -> Dict:
        """Analyze spatial distribution of surfels."""
        if self.memory_retriever.kdtree is None:
            return {"error": "No KD-tree available"}
        
        positions = self.memory_retriever.surfels['pos'].cpu().numpy()
        
        # Sample some points and compute nearest neighbor distances
        sample_size = min(1000, len(positions))
        sample_indices = np.random.choice(len(positions), sample_size, replace=False)
        sample_positions = positions[sample_indices]
        
        distances, _ = self.memory_retriever.kdtree.query(sample_positions, k=2)
        nn_distances = distances[:, 1]  # Distance to nearest neighbor (excluding self)
        
        return {
            "nearest_neighbor_distances": {
                "min": float(nn_distances.min()),
                "max": float(nn_distances.max()),
                "mean": float(nn_distances.mean()),
                "std": float(nn_distances.std())
            },
            "density_estimate": float(1.0 / nn_distances.mean()) if nn_distances.mean() > 0 else 0
        }
    
    def _analyze_memory_coverage(self) -> Dict:
        """Analyze memory coverage and efficiency."""
        total_views = len(self.memory_retriever.view_database)
        total_surfels = len(self.memory_retriever.surfels['pos']) if self.memory_retriever.surfels else 0
        
        return {
            "total_views": total_views,
            "total_surfels": total_surfels,
            "surfels_per_view": total_surfels / total_views if total_views > 0 else 0,
            "memory_efficiency": {
                "description": "Lower is more efficient (fewer surfels per view)",
                "ratio": total_surfels / total_views if total_views > 0 else 0
            }
        }
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_memory_visualization(memory_retriever, output_dir: str = "memory_viz"):
    """
    Convenience function to export comprehensive memory visualization.
    
    Args:
        memory_retriever: VGGTMemoryRetriever instance
        output_dir: Directory to save visualization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    visualizer = VGGTMemoryVisualizer(memory_retriever)
    
    # Export 3D representations
    visualizer.export_world_as_glb(str(output_path / "world_representation.glb"))
    visualizer.export_world_as_ply(str(output_path / "surfels_pointcloud.ply"))
    
    # Export analysis
    visualizer.export_memory_analysis(str(output_path / "memory_analysis.json"))
    
    print(f"Complete memory visualization exported to: {output_path}")
    
    return visualizer

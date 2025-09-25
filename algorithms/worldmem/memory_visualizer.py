"""
Memory Visualization Module for VGGT Memory Retriever

This module provides functionality to export the VGGT memory retriever's 
surfel-based world representation as 3D files (.glb, .ply) for visualization
and debugging purposes.
"""

import numpy as np
import torch
import trimesh
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple
import colorsys


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
        if self.memory_retriever.surfels is None:
            print("No surfels in memory to export.")
            return
            
        scene = trimesh.Scene()
        
        # Add coordinate axes for reference
        if include_axes:
            axes_meshes = self._create_coordinate_axes()
            for i, mesh in enumerate(axes_meshes):
                scene.add_geometry(mesh, node_name=f"axis_{i}")
        
        # Add surfels as colored shapes
        if include_surfels:
            surfel_meshes = self._create_surfel_meshes(surfel_scale)
            for i, mesh in enumerate(surfel_meshes):
                if hasattr(mesh, 'vertices'):  # Check if it's a proper mesh
                    scene.add_geometry(mesh, node_name=f"surfel_{i}")
        
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

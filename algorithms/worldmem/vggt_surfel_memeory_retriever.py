# algorithms/worldmem/vggt_memory_retriever.py

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Add VGGT to path - using vggt directory
sys.path.append("../../vggt")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

@dataclass
class Surfel:
    """
    Surface element (surfel) representation for memory indexing.
    """
    position: np.ndarray  # 3D position (x, y, z)
    normal: np.ndarray    # Surface normal (nx, ny, nz)
    radius: float         # Surfel radius
    view_indices: List[int]  # Indices of views that observed this surfel
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float32)
        self.normal = np.array(self.normal, dtype=np.float32)
        # Normalize the normal vector
        norm = np.linalg.norm(self.normal)
        if norm > 1e-8:
            self.normal = self.normal / norm

class Octree:
    """
    Simple octree implementation for efficient surfel spatial queries.
    """
    def __init__(self, positions: np.ndarray, max_points: int = 10, max_depth: int = 10):
        self.max_points = max_points
        self.max_depth = max_depth
        self.positions = positions
        self.indices = np.arange(len(positions))
        
        # Calculate bounding box
        if len(positions) > 0:
            self.min_bound = np.min(positions, axis=0)
            self.max_bound = np.max(positions, axis=0)
            self.center = (self.min_bound + self.max_bound) / 2
            self.size = np.max(self.max_bound - self.min_bound)
        else:
            self.min_bound = np.zeros(3)
            self.max_bound = np.ones(3)
            self.center = np.array([0.5, 0.5, 0.5])
            self.size = 1.0
        
        self.children = None
        self.is_leaf = True
        
        if len(positions) > max_points and max_depth > 0:
            self._subdivide(max_depth - 1)
    
    def _subdivide(self, remaining_depth):
        if remaining_depth <= 0:
            return
            
        self.is_leaf = False
        self.children = []
        
        # Create 8 children
        half_size = self.size / 2
        for i in range(8):
            child_center = self.center.copy()
            child_center[0] += half_size / 2 * (1 if (i & 1) else -1)
            child_center[1] += half_size / 2 * (1 if (i & 2) else -1)
            child_center[2] += half_size / 2 * (1 if (i & 4) else -1)
            
            # Find points in this child
            child_min = child_center - half_size / 2
            child_max = child_center + half_size / 2
            
            mask = np.all((self.positions >= child_min) & (self.positions <= child_max), axis=1)
            child_positions = self.positions[mask]
            child_indices = self.indices[mask]
            
            if len(child_positions) > 0:
                child = Octree.__new__(Octree)
                child.max_points = self.max_points
                child.max_depth = remaining_depth
                child.positions = child_positions
                child.indices = child_indices
                child.min_bound = child_min
                child.max_bound = child_max
                child.center = child_center
                child.size = half_size
                child.children = None
                child.is_leaf = True
                
                if len(child_positions) > self.max_points and remaining_depth > 0:
                    child._subdivide(remaining_depth - 1)
                    
                self.children.append(child)
    
    def query_ball_point(self, center: np.ndarray, radius: float) -> List[int]:
        """Query points within a ball around the center."""
        result = []
        self._query_ball_recursive(center, radius, result)
        return result
    
    def _query_ball_recursive(self, center: np.ndarray, radius: float, result: List[int]):
        # Check if sphere intersects with this node's bounding box
        closest_point = np.clip(center, self.min_bound, self.max_bound)
        dist_to_box = np.linalg.norm(center - closest_point)
        
        if dist_to_box > radius:
            return
            
        if self.is_leaf:
            # Check all points in this leaf
            distances = np.linalg.norm(self.positions - center, axis=1)
            mask = distances <= radius
            result.extend(self.indices[mask].tolist())
        else:
            # Recurse to children
            if self.children:
                for child in self.children:
                    child._query_ball_recursive(center, radius, result)

class VGGTSurfelMemoryRetriever:
    """
    Surfel-based memory retrieval system using VGGT for geometry estimation.
    This implements the VMem approach for view indexing and retrieval, but with VGGT's faster inference.
    """
    
    def __init__(self, device: str = "cuda", model_path: Optional[str] = None):
        self.device = device
        self.surfels: List[Surfel] = []
        self.view_to_surfel_map: Dict[int, List[int]] = {}  # view_id -> surfel_indices
        self.octree: Optional[Octree] = None
        
        # VGGT model loading
        print("Loading VGGT model...")
        if model_path is None:
            # Use HuggingFace pretrained model
            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        else:
            # Load from local path
            self.vggt_model = VGGT().to(device)
            self.vggt_model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.vggt_model.eval()
        
        # Determine dtype based on GPU capability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        
        # Parameters for surfel creation and merging
        self.position_threshold = 0.025  # Distance threshold for merging surfels
        self.normal_threshold = 0.7      # Cosine similarity threshold for normals
        self.radius_scale = 0.5          # Scale factor for surfel radius calculation
        self.downsample_factor = 4       # Downsample factor for efficiency
        
        # Threading for asynchronous operations
        self.memory_update_executor = ThreadPoolExecutor(max_workers=1)
    
    def __del__(self):
        """Clean up the executor on deletion."""
        if hasattr(self, 'memory_update_executor'):
            self.memory_update_executor.shutdown(wait=True)
    
    def add_view_to_memory(self, image: torch.Tensor, camera_pose: torch.Tensor, view_index: int):
        """
        Add a new view to the surfel-based memory.
        
        Args:
            image: RGB image tensor of shape (3, H, W) or (H, W, 3)
            camera_pose: Camera-to-world transformation matrix (4, 4)
            view_index: Unique identifier for this view
        """
        # Prepare image for VGGT processing
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                # VGGT expects (3, H, W) format
                processed_image = image.clone()
            else:
                # Convert from (H, W, 3) to (3, H, W)
                processed_image = image.permute(2, 0, 1)
            
            # Ensure values are in [0, 1] range
            if processed_image.max() > 1.0:
                processed_image = processed_image / 255.0
        else:
            raise ValueError("Expected torch.Tensor input for image")
        
        # Convert camera pose to numpy if needed
        if isinstance(camera_pose, torch.Tensor):
            camera_pose = camera_pose.cpu().numpy()
        
        # Run VGGT inference to get depth, point maps, and camera parameters
        try:
            # Add batch dimension and move to device
            images = processed_image.unsqueeze(0).to(self.device)  # (1, 3, H, W)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # VGGT processes single images efficiently
                    predictions = self.vggt_model(images)
            
            # Extract predictions
            depth_map = predictions['depth'][0, 0]  # (H, W)
            depth_conf = predictions['depth_conf'][0, 0]  # (H, W)
            
            # Get camera parameters and convert to extrinsics/intrinsics
            pose_enc = predictions['pose_enc'][0, 0]  # (9,)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc.unsqueeze(0).unsqueeze(0), 
                images.shape[-2:]
            )
            
            # Use VGGT's world points if available, otherwise unproject depth
            if 'world_points' in predictions:
                world_points = predictions['world_points'][0, 0]  # (H, W, 3)
            else:
                # Unproject depth to get world points
                world_points = unproject_depth_map_to_point_map(
                    depth_map.unsqueeze(0).unsqueeze(-1).cpu().numpy(),
                    extrinsic.cpu().numpy(),
                    intrinsic.cpu().numpy()
                )[0]  # (H, W, 3)
                world_points = torch.from_numpy(world_points).to(self.device)
            
            # Convert to surfels
            new_surfels = self._vggt_to_surfels(
                world_points, depth_map, depth_conf, camera_pose, view_index
            )
            
            # Merge with existing surfels
            self._merge_surfels_into_memory(new_surfels, view_index)
            
        except Exception as e:
            print(f"Warning: Failed to process view {view_index} with VGGT: {e}")
    
    def _vggt_to_surfels(self, world_points: torch.Tensor, depth: torch.Tensor, 
                        confidence: torch.Tensor, camera_pose: np.ndarray, 
                        view_index: int) -> List[Surfel]:
        """
        Convert VGGT world points to surfels.
        
        Args:
            world_points: World points tensor (H, W, 3)
            depth: Depth map tensor (H, W)
            confidence: Confidence map tensor (H, W)
            camera_pose: Camera pose matrix (4, 4)
            view_index: View index for this surfel
            
        Returns:
            List of Surfel objects
        """
        if isinstance(world_points, torch.Tensor):
            world_points = world_points.cpu().numpy()
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
        
        H, W = world_points.shape[:2]
        
        # Downsample for efficiency
        step = self.downsample_factor
        H_ds, W_ds = H // step, W // step
        
        surfels = []
        camera_center = camera_pose[:3, 3]
        
        # Create confidence threshold
        conf_threshold = np.percentile(confidence, 50)  # Use median as threshold
        
        for v in range(0, H_ds * step, step):
            for u in range(0, W_ds * step, step):
                if v >= H or u >= W:
                    continue
                
                # Check confidence
                if confidence[v, u] < conf_threshold:
                    continue
                
                # Get 3D position
                position = world_points[v, u]
                
                # Skip invalid points
                if np.any(np.isnan(position)) or np.any(np.isinf(position)):
                    continue
                
                # Estimate normal using neighboring points
                normal = self._estimate_normal(world_points, v, u, H, W)
                if normal is None:
                    continue
                
                # Calculate surfel radius based on depth and viewing angle
                depth_val = depth[v, u]
                view_dir = position - camera_center
                view_dir_norm = view_dir / (np.linalg.norm(view_dir) + 1e-8)
                
                # Angle between view direction and normal
                cos_angle = np.abs(np.dot(normal, view_dir_norm))
                cos_angle = max(cos_angle, 0.1)  # Avoid division by very small numbers
                
                # Radius proportional to depth and inversely proportional to viewing angle
                radius = self.radius_scale * depth_val / (100.0 * cos_angle)  # Assuming focal length ~100
                radius = max(radius, 0.001)  # Minimum radius
                
                # Create surfel
                surfel = Surfel(
                    position=position,
                    normal=normal,
                    radius=radius,
                    view_indices=[view_index]
                )
                surfels.append(surfel)
        
        return surfels
    
    def _estimate_normal(self, pointmap: np.ndarray, v: int, u: int, H: int, W: int) -> Optional[np.ndarray]:
        """Estimate surface normal at a pixel using neighboring points."""
        # Get neighboring points
        neighbors = []
        for dv in [-1, 0, 1]:
            for du in [-1, 0, 1]:
                nv, nu = v + dv, u + du
                if 0 <= nv < H and 0 <= nu < W:
                    point = pointmap[nv, nu]
                    if not (np.any(np.isnan(point)) or np.any(np.isinf(point))):
                        neighbors.append(point)
        
        if len(neighbors) < 4:
            return None
        
        # Use cross product of displacement vectors
        center = pointmap[v, u]
        
        # Try different neighbor pairs to get stable normal
        normals = []
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                v1 = neighbors[i] - center
                v2 = neighbors[j] - center
                
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    normal = np.cross(v1, v2)
                    norm_len = np.linalg.norm(normal)
                    if norm_len > 1e-6:
                        normals.append(normal / norm_len)
        
        if len(normals) == 0:
            return None
        
        # Average the normals
        avg_normal = np.mean(normals, axis=0)
        norm_len = np.linalg.norm(avg_normal)
        if norm_len < 1e-6:
            return None
        
        return avg_normal / norm_len
    
    def _merge_surfels_into_memory(self, new_surfels: List[Surfel], view_index: int):
        """
        Merge new surfels into the existing memory, combining similar surfels.
        
        Args:
            new_surfels: List of new surfels to add
            view_index: Index of the view these surfels came from
        """
        if len(self.surfels) == 0:
            # First surfels, just add them
            self.surfels.extend(new_surfels)
            self.view_to_surfel_map[view_index] = list(range(len(new_surfels)))
            self._rebuild_octree()
            return
        
        # Build octree if needed
        if self.octree is None:
            self._rebuild_octree()
        
        merged_count = 0
        new_surfel_indices = []
        
        for new_surfel in new_surfels:
            # Find nearby existing surfels
            nearby_indices = self.octree.query_ball_point(new_surfel.position, self.position_threshold)
            
            merged = False
            for idx in nearby_indices:
                existing_surfel = self.surfels[idx]
                
                # Check normal similarity
                normal_sim = np.dot(existing_surfel.normal, new_surfel.normal)
                if normal_sim > self.normal_threshold:
                    # Merge with existing surfel
                    if view_index not in existing_surfel.view_indices:
                        existing_surfel.view_indices.append(view_index)
                    merged = True
                    merged_count += 1
                    new_surfel_indices.append(idx)
                    break
            
            if not merged:
                # Add as new surfel
                new_idx = len(self.surfels)
                self.surfels.append(new_surfel)
                new_surfel_indices.append(new_idx)
        
        # Update view to surfel mapping
        self.view_to_surfel_map[view_index] = new_surfel_indices
        
        # Rebuild octree with new surfels
        self._rebuild_octree()
        
        print(f"Added {len(new_surfels)} surfels, merged {merged_count}, total surfels: {len(self.surfels)}")
    
    def _rebuild_octree(self):
        """Rebuild the octree with current surfels."""
        if len(self.surfels) > 0:
            positions = np.array([surfel.position for surfel in self.surfels])
            self.octree = Octree(positions, max_points=10)
        else:
            self.octree = None
    
    def retrieve_relevant_views(self, target_pose: torch.Tensor, k: int = 8, 
                              image_size: Tuple[int, int] = (64, 64)) -> List[int]:
        """
        Retrieve the most relevant views for a target camera pose.
        
        Args:
            target_pose: Target camera pose (4, 4)
            k: Number of views to retrieve
            image_size: Size for surfel rendering (height, width)
            
        Returns:
            List of view indices sorted by relevance
        """
        if len(self.surfels) == 0:
            return []
        
        if isinstance(target_pose, torch.Tensor):
            target_pose = target_pose.cpu().numpy()
        
        # Render surfels from target pose to get visibility
        view_votes = self._render_surfels_for_retrieval(target_pose, image_size)
        
        # Sort views by vote count and return top k
        sorted_views = sorted(view_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Apply non-maximum suppression to avoid similar views
        selected_views = self._apply_nms_to_views(sorted_views, k)
        
        return [view_idx for view_idx, _ in selected_views[:k]]
    
    def _render_surfels_for_retrieval(self, target_pose: np.ndarray, 
                                    image_size: Tuple[int, int]) -> Dict[int, float]:
        """
        Render surfels from target pose to determine view relevance.
        
        Args:
            target_pose: Camera pose matrix (4, 4)
            image_size: Rendering resolution (height, width)
            
        Returns:
            Dictionary mapping view indices to relevance scores
        """
        H, W = image_size
        view_votes = {}
        
        # Camera parameters
        focal_length = min(H, W) * 0.8  # Reasonable focal length
        cx, cy = W / 2, H / 2
        
        # Camera center and rotation
        camera_center = target_pose[:3, 3]
        rotation = target_pose[:3, :3]
        
        # Create depth buffer for occlusion handling
        depth_buffer = np.full((H, W), np.inf)
        vote_buffer = np.zeros((H, W), dtype=object)
        
        # Initialize vote buffer
        for i in range(H):
            for j in range(W):
                vote_buffer[i, j] = {}
        
        for surfel in self.surfels:
            # Transform surfel to camera coordinates
            surfel_cam = rotation.T @ (surfel.position - camera_center)
            
            # Skip surfels behind camera
            if surfel_cam[2] <= 0:
                continue
            
            # Project to image plane
            x_proj = focal_length * surfel_cam[0] / surfel_cam[2] + cx
            y_proj = focal_length * surfel_cam[1] / surfel_cam[2] + cy
            
            # Check if projection is within image bounds
            if x_proj < 0 or x_proj >= W or y_proj < 0 or y_proj >= H:
                continue
            
            # Calculate surfel coverage in image
            depth = surfel_cam[2]
            pixel_radius = max(1, int(focal_length * surfel.radius / depth))
            
            # Render surfel as a circle
            u_center, v_center = int(x_proj), int(y_proj)
            
            for v in range(max(0, v_center - pixel_radius), min(H, v_center + pixel_radius + 1)):
                for u in range(max(0, u_center - pixel_radius), min(W, u_center + pixel_radius + 1)):
                    # Check if pixel is within surfel radius
                    dist = np.sqrt((u - x_proj)**2 + (v - y_proj)**2)
                    if dist <= pixel_radius:
                        # Check depth buffer for occlusion
                        if depth < depth_buffer[v, u]:
                            depth_buffer[v, u] = depth
                            
                            # Vote for views that observed this surfel
                            for view_idx in surfel.view_indices:
                                if view_idx not in vote_buffer[v, u]:
                                    vote_buffer[v, u][view_idx] = 0
                                # Weight by inverse distance and surfel size
                                weight = (pixel_radius - dist + 1) / (depth + 1)
                                vote_buffer[v, u][view_idx] += weight
        
        # Aggregate votes across all pixels
        for i in range(H):
            for j in range(W):
                for view_idx, weight in vote_buffer[i, j].items():
                    if view_idx not in view_votes:
                        view_votes[view_idx] = 0
                    view_votes[view_idx] += weight
        
        return view_votes
    
    def _apply_nms_to_views(self, sorted_views: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        """
        Apply non-maximum suppression to avoid selecting very similar views.
        This is a placeholder - in practice, you'd compare camera poses.
        
        Args:
            sorted_views: List of (view_index, score) tuples sorted by score
            k: Maximum number of views to select
            
        Returns:
            List of selected (view_index, score) tuples
        """
        # For now, just return the top k views
        # In a full implementation, you'd compare camera poses to avoid similar viewpoints
        return sorted_views[:k]
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about the current memory state."""
        total_views = len(self.view_to_surfel_map)
        total_surfels = len(self.surfels)
        
        # Calculate average views per surfel
        view_counts = [len(surfel.view_indices) for surfel in self.surfels]
        avg_views_per_surfel = np.mean(view_counts) if view_counts else 0
        
        return {
            "total_views": total_views,
            "total_surfels": total_surfels,
            "avg_views_per_surfel": avg_views_per_surfel
        }

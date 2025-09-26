# algorithms/worldmem/surfel_memory_retriever.py

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from copy import deepcopy

# Add the vmem path to use CUT3R
sys.path.append("/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/vmem")
sys.path.append("/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/vmem/extern/CUT3R")

from extern.CUT3R.surfel_inference import run_inference_from_pil
from extern.CUT3R.add_ckpt_path import add_path_to_dust3r
from extern.CUT3R.src.dust3r.model import ARCroco3DStereo
try:
    from utils import tensor_to_pil
except ImportError:
    # Fallback tensor_to_pil implementation
    import torchvision.transforms.functional as TF
    from PIL import Image
    
    def tensor_to_pil(tensor):
        """Convert tensor to PIL Image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            # Already in (C, H, W) format
            pass
        elif tensor.dim() == 3 and tensor.shape[2] == 3:
            # Convert from (H, W, C) to (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        
        # Ensure values are in [0, 1] range
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        # Convert to PIL
        return TF.to_pil_image(tensor)


@dataclass
class Surfel:
    """
    A surfel (surface element) representing a 3D surface point with associated metadata.
    """
    position: np.ndarray  # 3D position (x, y, z)
    normal: np.ndarray    # Surface normal (nx, ny, nz)
    radius: float         # Surfel radius
    view_indices: List[int] = None  # Indices of views that observed this surfel
    
    def __post_init__(self):
        if self.view_indices is None:
            self.view_indices = []


class Octree:
    """
    Simple octree implementation for efficient spatial queries of surfels.
    """
    def __init__(self, surfels: List[Surfel], max_points: int = 10):
        self.surfels = surfels
        self.max_points = max_points
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build a simple spatial index using positions"""
        if len(self.surfels) == 0:
            return
        
        positions = np.array([s.position for s in self.surfels])
        self.positions = positions
        
    def query_ball_point(self, center: np.ndarray, radius: float) -> List[int]:
        """Find all surfel indices within radius of center point"""
        if len(self.surfels) == 0:
            return []
        
        distances = np.linalg.norm(self.positions - center, axis=1)
        return np.where(distances <= radius)[0].tolist()


class SurfelMemoryRetriever:
    """
    Surfel-based memory retrieval system using CUT3R for 3D scene reconstruction.
    This implements the VMem approach adapted for WorldMem.
    """
    
    def __init__(self, device: str = "cuda", model_path: str = None):
        self.device = device
        self.surfels: List[Surfel] = []
        self.view_database: Dict[int, Dict] = {}  # Store view data: {idx: {image, pose, etc}}
        self.octree: Optional[Octree] = None
        self.current_view_index = 0
        
        # Initialize CUT3R model
        self._initialize_cut3r_model(model_path)
        
        # Asynchronous memory writer
        self.memory_update_executor = ThreadPoolExecutor(max_workers=1)
        
        # Configuration parameters
        self.position_threshold = 0.025  # Distance threshold for surfel merging
        self.normal_threshold = 0.7      # Cosine similarity threshold for normal alignment
        self.downsample_factor = 4       # Downsample factor for point maps
        self.alpha = 0.2                 # Factor for surfel radius calculation
    
    def _initialize_cut3r_model(self, model_path: str = None):
        """Initialize the CUT3R model for 3D reconstruction"""
        try:
            # Use default model path if not provided
            if model_path is None:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="liguang0115/vmem", 
                    filename="cut3r_512_dpt_4_64.pth"
                )
            
            print(f"Loading CUT3R model from {model_path}...")
            add_path_to_dust3r(model_path)
            self.surfel_model = ARCroco3DStereo.from_pretrained(model_path).to(self.device)
            self.surfel_model.eval()
            
        except Exception as e:
            print(f"Warning: Could not load CUT3R model: {e}")
            self.surfel_model = None
    
    def add_view_to_memory(self, image: torch.Tensor, pose: torch.Tensor):
        """
        Add a new view to the memory by extracting surfels and updating the spatial index.
        
        Args:
            image: RGB image tensor of shape (3, H, W) or (H, W, 3)
            pose: Camera pose matrix of shape (4, 4) - camera-to-world transformation
        """
        if self.surfel_model is None:
            print("Warning: CUT3R model not available, skipping memory update")
            return
        
        # Ensure image is in the correct format
        if image.dim() == 3 and image.shape[0] == 3:
            # Convert from (3, H, W) to (H, W, 3)
            image = image.permute(1, 2, 0)
        
        # Convert to PIL image for CUT3R processing
        pil_image = tensor_to_pil(image.unsqueeze(0) if image.dim() == 3 else image)
        
        # Store view in database
        view_idx = self.current_view_index
        self.view_database[view_idx] = {
            'image': image.clone(),
            'pose': pose.clone(),
            'pil_image': pil_image
        }
        
        # Extract surfels from the image
        try:
            self._extract_and_merge_surfels([pil_image], [pose.cpu().numpy()], [view_idx])
        except Exception as e:
            print(f"Warning: Failed to extract surfels: {e}")
        
        self.current_view_index += 1
    
    def _extract_and_merge_surfels(self, pil_images: List, poses: List[np.ndarray], view_indices: List[int]):
        """
        Extract surfels from images using CUT3R and merge with existing surfels.
        """
        if len(pil_images) == 0:
            return
        
        try:
            # Run CUT3R inference
            scene_result = run_inference_from_pil(
                pil_images=pil_images,
                model=self.surfel_model,
                poses=poses,
                depths=None,
                lr=0.01,
                niter=100,  # Reduced iterations for speed
                device=self.device,
                size=512,
                visualize=False,
                save_flag=False
            )
            
            # Extract point clouds and convert to surfels
            point_clouds = scene_result['point_clouds']
            depths = scene_result['depths']
            confidences = scene_result['confidences']
            camera_info = scene_result['camera_info']
            
            for i, (pc, depth, conf) in enumerate(zip(point_clouds, depths, confidences)):
                if i >= len(view_indices):
                    break
                    
                view_idx = view_indices[i]
                pose = poses[i]
                
                # Convert point cloud to surfels
                new_surfels = self._pointcloud_to_surfels(
                    pc.squeeze(0), depth.squeeze(0), conf.squeeze(0), 
                    pose, view_idx
                )
                
                # Merge with existing surfels
                self._merge_surfels(new_surfels, view_idx)
                
        except Exception as e:
            print(f"Warning: CUT3R processing failed: {e}")
    
    def _pointcloud_to_surfels(self, pointcloud: torch.Tensor, depth: torch.Tensor, 
                              confidence: torch.Tensor, pose: np.ndarray, view_idx: int) -> List[Surfel]:
        """
        Convert a point cloud to surfels.
        
        Args:
            pointcloud: Point cloud tensor of shape (H, W, 3)
            depth: Depth map tensor of shape (H, W)
            confidence: Confidence map tensor of shape (H, W)
            pose: Camera pose matrix (4, 4)
            view_idx: Index of the view
            
        Returns:
            List of Surfel objects
        """
        surfels = []
        
        # Downsample for efficiency
        H, W = pointcloud.shape[:2]
        step = self.downsample_factor
        
        # Extract camera parameters
        camera_center = pose[:3, 3]
        
        # Process downsampled points
        for y in range(0, H, step):
            for x in range(0, W, step):
                if confidence[y, x] < 0.5:  # Skip low-confidence points
                    continue
                
                point = pointcloud[y, x].cpu().numpy()
                depth_val = depth[y, x].item()
                
                # Skip invalid points
                if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                    continue
                
                # Estimate normal from neighboring points
                normal = self._estimate_normal(pointcloud, y, x, step)
                if normal is None:
                    continue
                
                # Calculate surfel radius based on depth and viewing angle
                view_dir = point - camera_center
                view_dir = view_dir / np.linalg.norm(view_dir)
                cos_angle = np.abs(np.dot(normal, view_dir))
                
                # Radius calculation (similar to VMem)
                focal_length = 256  # Approximate focal length
                radius = (depth_val / focal_length) / (self.alpha + (1 - self.alpha) * cos_angle)
                radius = max(0.01, min(radius, 1.0))  # Clamp radius
                
                # Create surfel
                surfel = Surfel(
                    position=point.copy(),
                    normal=normal.copy(),
                    radius=radius,
                    view_indices=[view_idx]
                )
                surfels.append(surfel)
        
        return surfels
    
    def _estimate_normal(self, pointcloud: torch.Tensor, y: int, x: int, step: int) -> Optional[np.ndarray]:
        """Estimate surface normal at a point using neighboring points"""
        H, W = pointcloud.shape[:2]
        
        # Get neighboring points
        neighbors = []
        for dy, dx in [(0, step), (step, 0), (0, -step), (-step, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                neighbor = pointcloud[ny, nx].cpu().numpy()
                if not (np.any(np.isnan(neighbor)) or np.any(np.isinf(neighbor))):
                    neighbors.append(neighbor)
        
        if len(neighbors) < 2:
            return None
        
        # Use first two valid neighbors to compute normal
        center = pointcloud[y, x].cpu().numpy()
        v1 = neighbors[0] - center
        v2 = neighbors[1] - center
        
        # Cross product to get normal
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        
        if norm_length < 1e-8:
            return None
        
        return normal / norm_length
    
    def _merge_surfels(self, new_surfels: List[Surfel], view_idx: int):
        """
        Merge new surfels with existing ones based on position and normal similarity.
        """
        if len(self.surfels) == 0:
            self.surfels.extend(new_surfels)
            self._rebuild_octree()
            return
        
        merged_count = 0
        for new_surfel in new_surfels:
            merged = False
            
            # Find nearby existing surfels
            if self.octree is not None:
                nearby_indices = self.octree.query_ball_point(
                    new_surfel.position, self.position_threshold
                )
                
                for idx in nearby_indices:
                    if idx >= len(self.surfels):
                        continue
                        
                    existing_surfel = self.surfels[idx]
                    
                    # Check normal similarity
                    normal_similarity = np.dot(existing_surfel.normal, new_surfel.normal)
                    
                    if normal_similarity > self.normal_threshold:
                        # Merge: add view index to existing surfel
                        if view_idx not in existing_surfel.view_indices:
                            existing_surfel.view_indices.append(view_idx)
                        merged = True
                        merged_count += 1
                        break
            
            if not merged:
                # Add as new surfel
                self.surfels.append(new_surfel)
        
        # Rebuild spatial index
        self._rebuild_octree()
        print(f"Merged {merged_count} surfels, added {len(new_surfels) - merged_count} new surfels")
    
    def _rebuild_octree(self):
        """Rebuild the octree spatial index"""
        if len(self.surfels) > 0:
            self.octree = Octree(self.surfels)
    
    def retrieve_relevant_views(self, target_pose: torch.Tensor, k: int = 4, 
                               image_size: Tuple[int, int] = (256, 256)) -> List[int]:
        """
        Retrieve the most relevant past views for a target camera pose using surfel-based indexing.
        
        Args:
            target_pose: Target camera pose matrix (4, 4)
            k: Number of views to retrieve
            image_size: Size of the rendered image for vote counting
            
        Returns:
            List of view indices sorted by relevance
        """
        if len(self.surfels) == 0 or len(self.view_database) == 0:
            return []
        
        # Render surfels from target viewpoint to get visibility votes
        view_votes = self._render_surfel_votes(target_pose.cpu().numpy(), image_size)
        
        # Sort views by vote count and return top-k
        if len(view_votes) == 0:
            # Fallback: return most recent views
            available_indices = list(self.view_database.keys())
            return available_indices[-k:] if len(available_indices) >= k else available_indices
        
        sorted_views = sorted(view_votes.items(), key=lambda x: x[1], reverse=True)
        top_k_views = [view_idx for view_idx, _ in sorted_views[:k]]
        
        return top_k_views
    
    def _render_surfel_votes(self, target_pose: np.ndarray, image_size: Tuple[int, int]) -> Dict[int, int]:
        """
        Render surfels from target viewpoint and count votes for each view.
        
        Args:
            target_pose: Target camera pose matrix (4, 4)
            image_size: Size of the rendered image
            
        Returns:
            Dictionary mapping view indices to vote counts
        """
        if len(self.surfels) == 0:
            return {}
        
        H, W = image_size
        view_votes = {}
        
        # Extract camera parameters
        camera_center = target_pose[:3, 3]
        camera_rotation = target_pose[:3, :3]
        
        # Simple projection (assuming fixed focal length)
        focal_length = min(H, W) * 0.8
        cx, cy = W // 2, H // 2
        
        for surfel in self.surfels:
            # Transform point to camera coordinate system
            world_to_camera = np.linalg.inv(target_pose)
            point_cam = world_to_camera @ np.append(surfel.position, 1.0)
            
            # Skip points behind camera
            if point_cam[2] <= 0:
                continue
            
            # Project to image plane
            x_proj = int(focal_length * point_cam[0] / point_cam[2] + cx)
            y_proj = int(focal_length * point_cam[1] / point_cam[2] + cy)
            
            # Check if projection is within image bounds
            if 0 <= x_proj < W and 0 <= y_proj < H:
                # Check if surfel is facing the camera (backface culling)
                view_dir = surfel.position - camera_center
                view_dir = view_dir / np.linalg.norm(view_dir)
                
                if np.dot(surfel.normal, view_dir) > 0:  # Facing camera
                    # Add votes for all views that observed this surfel
                    for view_idx in surfel.view_indices:
                        if view_idx in self.view_database:
                            view_votes[view_idx] = view_votes.get(view_idx, 0) + 1
        
        return view_votes
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the current memory state"""
        return {
            'num_surfels': len(self.surfels),
            'num_views': len(self.view_database),
            'memory_size_mb': self._estimate_memory_size()
        }
    
    def _estimate_memory_size(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimation
        surfel_size = len(self.surfels) * (3 * 8 + 3 * 8 + 8 + 4 * 4) / (1024 * 1024)  # positions, normals, radius, indices
        view_size = len(self.view_database) * 0.1  # Rough estimate for view data
        return surfel_size + view_size
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'memory_update_executor'):
            self.memory_update_executor.shutdown(wait=True)

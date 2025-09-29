# algorithms/worldmem/memory_adapter.py

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from omegaconf import OmegaConf

# Add vmem to path
vmem_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vmem"))
# if vmem_path not in sys.path:
#     sys.path.insert(0, vmem_path)

# Import VMem components

# the utils.util is in the folder CITS4010-4011vmem/utils/util.py
# we are in the folder CITS4010-4011
# so we need to go up one level to get to the vmem folder and then import the utils.util
# Add the project root to the path so 'vmem' can be found as a package

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vmem.modeling.pipeline import VMemPipeline


from vmem.utils.util import (
    tensor_to_pil, 
    get_default_intrinsics, 
    load_img_and_K, 
    transform_img_and_K,
    average_camera_pose
)

# same for this from modeling.pipeline import VMemPipeline


class VMemAdapter:
    """
    Adapter class that bridges WorldMem's data structures with VMem's VMemPipeline.
    This allows us to reuse VMem's complete surfel memory system without reimplementation.
    """
    
    def __init__(self, device: str = "cuda", config_path: Optional[str] = None):
        self.device = device
        
        # Load VMem config
        if config_path is None:
            config_path = os.path.join(vmem_path, "configs/inference/inference.yaml")
        
        self.config = OmegaConf.load(config_path)
        
        # Initialize VMem pipeline
        self.pipeline = VMemPipeline(self.config, device=device)
        
        # Track current state
        self.is_initialized = False
        self.current_frames = []
        
    def initialize_with_frame(self, image: torch.Tensor, camera_pose: torch.Tensor, 
                             K: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Initialize the VMem pipeline with the first frame.
        
        Args:
            image: Initial RGB image tensor [3, H, W] or [1, 3, H, W] in range [0, 1] or [-1, 1]
            camera_pose: Camera-to-world transformation matrix [4, 4]
            K: Camera intrinsic matrix [3, 3]. If None, uses default intrinsics
            
        Returns:
            PIL-converted image tensor for consistency
        """
        # Ensure image is in correct format
        if image.dim() == 4:
            image = image.squeeze(0)  # Remove batch dimension
        
        # Ensure image is in [0, 1] range
        if image.min() < -0.1:  # Detect [-1, 1] range
            image = (image + 1.0) / 2.0
        
        # Convert camera pose to numpy if needed
        if isinstance(camera_pose, torch.Tensor):
            camera_pose = camera_pose.cpu().numpy()
        
        # Use default intrinsics if not provided
        if K is None:
            K = get_default_intrinsics()[0].cpu().numpy()
        elif isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        
        # Initialize the pipeline
        pil_frame = self.pipeline.initialize(image.unsqueeze(0), camera_pose, K)
        
        self.is_initialized = True
        self.current_frames = [pil_frame]
        
        return image
    
    def generate_trajectory_frames(self, c2ws: List[np.ndarray], Ks: List[np.ndarray],
                                 use_non_maximum_suppression: Optional[bool] = None) -> List:
        """
        Generate frames for a trajectory using VMem's pipeline.
        
        Args:
            c2ws: List of camera-to-world matrices
            Ks: List of camera intrinsic matrices
            use_non_maximum_suppression: Whether to use NMS for frame selection
            
        Returns:
            List of PIL images for the generated frames
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline must be initialized first")
        
        # Use VMem's trajectory generation
        new_frames = self.pipeline.generate_trajectory_frames(c2ws, Ks, use_non_maximum_suppression)
        
        # Update current frames
        self.current_frames.extend(new_frames)
        
        return new_frames
    
    def undo_latest_move(self) -> bool:
        """
        Undo the latest move using VMem's undo functionality.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            return False
        
        success = self.pipeline.undo_latest_move()
        
        if success:
            # Update current frames to match pipeline state
            self.current_frames = [tensor_to_pil(frame) for frame in self.pipeline.pil_frames]
        
        return success
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about the current memory state."""
        if not self.is_initialized:
            return {"total_frames": 0, "total_surfels": 0}
        
        return {
            "total_frames": len(self.pipeline.pil_frames),
            "total_surfels": len(self.pipeline.surfels),
            "memory_condition_length": self.pipeline.memory_condition_length
        }
    
    def get_context_info(self, target_c2ws: List[np.ndarray], 
                        use_non_maximum_suppression: Optional[bool] = None) -> Dict:
        """
        Get context information for novel view synthesis using VMem's retrieval.
        
        Args:
            target_c2ws: Target camera-to-world matrices
            use_non_maximum_suppression: Whether to use NMS
            
        Returns:
            Dictionary containing context information
        """
        if not self.is_initialized:
            return {}
        
        # Convert to tensor format expected by VMem
        target_c2ws_tensor = torch.from_numpy(np.array(target_c2ws)).to(self.device)
        
        return self.pipeline.get_context_info(target_c2ws_tensor, use_non_maximum_suppression)
    
    def reset(self):
        """Reset the pipeline to initial state."""
        self.pipeline.reset()
        self.is_initialized = False
        self.current_frames = []
    
    @property
    def frames(self) -> List:
        """Get current frames."""
        return self.current_frames
    
    @property
    def surfels(self) -> List:
        """Get current surfels from the pipeline."""
        if not self.is_initialized:
            return []
        return self.pipeline.surfels
    
    @property
    def poses(self) -> List[np.ndarray]:
        """Get current camera poses."""
        if not self.is_initialized:
            return []
        return self.pipeline.c2ws


def convert_worldmem_pose_to_vmem(pose: torch.Tensor) -> np.ndarray:
    """
    Convert WorldMem pose format to VMem camera-to-world format.
    
    Args:
        pose: Pose tensor from WorldMem (could be various formats)
        
    Returns:
        4x4 camera-to-world transformation matrix
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    
    # Ensure it's a 4x4 matrix
    if pose.shape == (4, 4):
        return pose
    elif pose.shape == (3, 4):
        # Add bottom row [0, 0, 0, 1]
        bottom_row = np.array([[0, 0, 0, 1]], dtype=pose.dtype)
        return np.vstack([pose, bottom_row])
    else:
        raise ValueError(f"Unsupported pose shape: {pose.shape}")


def convert_worldmem_image_to_vmem(image: torch.Tensor) -> torch.Tensor:
    """
    Convert WorldMem image format to VMem format.
    
    Args:
        image: Image tensor from WorldMem
        
    Returns:
        Image tensor in VMem format [3, H, W] in range [0, 1]
    """
    # Handle different input formats
    if image.dim() == 4:
        image = image.squeeze(0)  # Remove batch dimension
    
    if image.dim() == 3 and image.shape[0] != 3:
        # Assume [H, W, C] format, convert to [C, H, W]
        image = image.permute(2, 0, 1)
    
    # Normalize to [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    elif image.min() < -0.1:
        # Convert from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
    
    return image

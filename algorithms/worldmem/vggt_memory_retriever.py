import torch
import numpy as np
from scipy.spatial import KDTree
from vggt.models.vggt import VGGT
from.geometry_utils import unproject_depth_to_pointcloud, pointcloud_to_surfels
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import torch.nn.functional as F

class VGGTMemoryRetriever:
    """
    Manages a geometrically-grounded memory using VGGT and a surfel index,
    inspired by the VMem architecture.[1]
    """
    def __init__(self, device, d_thresh=0.1, n_thresh=0.9, downsample_factor=4):
        self.device = device
        self.dtype = torch.float16
        
        print("Loading VGGT model...")
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device, dtype=self.dtype)
        self.vggt_model.eval()
        print("VGGT model loaded.")

        # Memory stores
        self.view_database =  []# Stores (raw_frame_tensor, c2w_matrix)
        self.surfels = None      # Dict storing all surfel positions, normals, radii
        self.surfel_to_views = []# List of lists, mapping surfel index to view indices
        self.kdtree = None       # KD-Tree for efficient spatial queries.
                                 # NOTE: Research recommends an Octree for better scalability in merging operations.[1]
                                 # A KD-Tree is a valid and efficient alternative for nearest-neighbor searches.

        # Hyperparameters for surfel merging [1, 1]
        self.d_thresh = d_thresh
        self.n_thresh = n_thresh
        self.downsample_factor = downsample_factor

    @torch.no_grad()
    def add_view_to_memory(self, new_frame_tensor, new_c2w_matrix):
        """
        The "Write to Memory" pipeline. Processes a new frame to update the geometric index.
        """
        frame_index = len(self.view_database)
        self.view_database.append((new_frame_tensor.cpu(), new_c2w_matrix.cpu()))
        
        print(f"Adding view {frame_index} to geometric memory...")
        
        # 1. Geometry Acquisition via VGGT [1, 1]
        image_size_hw = new_frame_tensor.shape[-2:]
        
        # Resize to make dimensions compatible with VGGT patch size (14)
        h, w = image_size_hw
        new_h = ((h + 13) // 14) * 14
        new_w = ((w + 13) // 14) * 14
        
        print(f"DEBUG: Original image size: {h}x{w}")
        print(f"DEBUG: Resizing to VGGT-compatible size: {new_h}x{new_w}")
        
        # Ensure we have the right tensor format for interpolation
        if new_frame_tensor.dim() == 3:  # CHW
            frame_for_resize = new_frame_tensor.unsqueeze(0)  # Add batch dimension
        else:  # Already has batch dimension
            frame_for_resize = new_frame_tensor
            
        resized_frame = F.interpolate(
            frame_for_resize, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        print(f"DEBUG: Resized frame shape: {resized_frame.shape}")
            
        vggt_input = resized_frame.to(self.device, dtype=self.dtype)
        
        # FIX: Updated to use the recommended torch.amp.autocast syntax
        print(f"DEBUG: About to run VGGT inference on shape {vggt_input.shape}")
        try:
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                predictions = self.vggt_model(vggt_input)
            print(f"DEBUG: VGGT inference completed successfully")
        except Exception as e:
            print(f"ERROR: VGGT inference failed: {e}")
            return  # Skip this frame if VGGT fails

        depth = predictions["depth"]
        pose_enc = predictions["pose_enc"]

        print(f"DEBUG: Depth shape: {depth.shape}, min/max: {depth.min():.6f}/{depth.max():.6f}")
        print(f"DEBUG: Pose encoding shape: {pose_enc.shape}")
        
        # Check if depth is valid
        valid_depth_pixels = (depth > 0.01).sum().item()
        total_pixels = depth.numel()
        print(f"DEBUG: Valid depth pixels: {valid_depth_pixels}/{total_pixels} ({100*valid_depth_pixels/total_pixels:.1f}%)")
        
        if valid_depth_pixels == 0:
            print("WARNING: No valid depth pixels found, skipping frame")
            return

        # 2. Unproject depth to get accurate point cloud [1, 1]
        try:
            pointcloud = unproject_depth_to_pointcloud(depth, pose_enc, (new_h, new_w))
            print(f"DEBUG: Pointcloud unprojection successful")
        except Exception as e:
            print(f"ERROR: Pointcloud unprojection failed: {e}")
            return
        
        # 3. Convert point cloud to surfels [1, 1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (new_h, new_w))
        camera_params = {'extrinsics': extrinsics, 'intrinsics': intrinsics}
        
        print(f"DEBUG: Pointcloud shape: {pointcloud.shape}")
        print(f"DEBUG: Pointcloud min/max: {pointcloud.min():.3f} / {pointcloud.max():.3f}")
        print(f"DEBUG: Depth min/max: {depth.min():.3f} / {depth.max():.3f}")
        
        new_positions, new_normals, new_radii = pointcloud_to_surfels(
            pointcloud, camera_params, self.downsample_factor
        )
        
        print(f"DEBUG: Generated {len(new_positions)} surfels")
        print(f"DEBUG: Surfel positions shape: {new_positions.shape}")
        if len(new_positions) > 0:
            print(f"DEBUG: Surfel position range: {new_positions.min(dim=0)[0]} to {new_positions.max(dim=0)[0]}")
            print(f"DEBUG: Surfel radii range: {new_radii.min():.6f} to {new_radii.max():.6f}")

        # 4. Merge new surfels into the global index [1, 1]
        if self.surfels is None:
            self.surfels = {'pos': new_positions, 'norm': new_normals, 'rad': new_radii}
            self.surfel_to_views = [[frame_index] for _ in range(len(new_positions))]
        else:
            if self.kdtree is not None and len(new_positions) > 0:
                # Query the 5 nearest neighbors for each new surfel
                distances, indices = self.kdtree.query(new_positions.cpu().numpy(), k=5, workers=-1)
                
                # FIX: Convert numpy distances to a tensor for consistent operations
                distances = torch.from_numpy(distances).to(new_normals.device)

                # Get existing normals for queried indices
                existing_normals = self.surfels['norm'][indices] # Shape: (num_new_surfels, 5, 3)
                
                # Compute dot products in a batch: (num_new, 1, 3) dot (num_new, 5, 3) -> (num_new, 5)
                normal_similarity = torch.sum(new_normals.unsqueeze(1) * existing_normals, dim=-1)

                # Find the first match for each new surfel that satisfies both distance and normal thresholds
                match_mask = (distances < self.d_thresh) & (normal_similarity > self.n_thresh)
                
                # Use find_first_true or equivalent logic to get the index of the first match
                first_match_indices = torch.argmax(match_mask.int(), dim=1)
                has_match = match_mask.any(dim=1)

                # Update existing surfels for matched new surfels
                if has_match.any():
                    matched_new_indices = torch.arange(len(indices))[has_match]
                    matched_global_indices = indices[matched_new_indices, first_match_indices[has_match]]
                    for idx in matched_global_indices:
                        if frame_index not in self.surfel_to_views[idx]:
                            self.surfel_to_views[idx].append(frame_index)

                # Add unmatched new surfels to the global index
                unmatched_mask = ~has_match
                if unmatched_mask.any():
                    self.surfels['pos'] = torch.cat([self.surfels['pos'], new_positions[unmatched_mask]])
                    self.surfels['norm'] = torch.cat([self.surfels['norm'], new_normals[unmatched_mask]])
                    self.surfels['rad'] = torch.cat([self.surfels['rad'], new_radii[unmatched_mask]])
                    for _ in range(unmatched_mask.sum()):
                        self.surfel_to_views.append([frame_index])

        # Rebuild KD-Tree after updates
        if self.surfels and len(self.surfels['pos']) > 0:
            self.kdtree = KDTree(self.surfels['pos'].cpu().numpy())
            print(f"Memory updated. Total surfels: {len(self.surfels['pos'])}")
            print(f"DEBUG: Final surfel memory state:")
            print(f"  - Total surfels: {len(self.surfels['pos'])}")
            print(f"  - Surfel positions shape: {self.surfels['pos'].shape}")
            print(f"  - Surfel normals shape: {self.surfels['norm'].shape}")
            print(f"  - Surfel radii shape: {self.surfels['rad'].shape}")
            print(f"  - Position range: {self.surfels['pos'].min(dim=0)[0]} to {self.surfels['pos'].max(dim=0)[0]}")
        else:
            print("DEBUG: No surfels in memory after update!")

    @torch.no_grad()
    def retrieve_relevant_views(self, target_c2w, k=4, intrinsics=None, image_size=(256, 256)):
        """
        The "Read from Memory" pipeline. Retrieves top-K views using an occlusion-aware mechanism.
        This is a simplified but more robust simulation of the GPU rendering pipeline described in the research.[1, 1]
        """
        if self.surfels is None or len(self.view_database) <= k:
            # Fallback for early stages: return most recent frames
            num_views = len(self.view_database)
            indices = list(range(max(0, num_views - k), num_views))
            while len(indices) < k: indices.append(0) # Pad if necessary
            return indices

        print(f"Retrieving from memory for target pose...")
        
        # 1. Broad-phase Culling (Simplified to radius search)
        cam_center = target_c2w[:3, 3]
        # Increased radius for better coverage
        candidate_indices = self.kdtree.query_ball_point(cam_center.cpu().numpy(), r=100.0)
        
        if not candidate_indices:
            return list(range(max(0, len(self.view_database) - k), len(self.view_database)))

        culled_pos = self.surfels['pos'][candidate_indices].to(self.device)
        culled_rad = self.surfels['rad'][candidate_indices].to(self.device)
        
        # 2. Visibility Check with Simplified Z-Buffering [1, 1]
        # This is the core improvement: it approximates occlusion handling.
        w2c = torch.linalg.inv(target_c2w.to(self.device))
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # Transform surfel positions to camera space
        cam_space_pos = torch.matmul(culled_pos, R.T) + t
        
        # Frustum check: only consider points in front of the camera
        front_mask = cam_space_pos[:, 2] > 0.1
        if not torch.any(front_mask):
            return list(range(max(0, len(self.view_database) - k), len(self.view_database)))

        cam_space_pos = cam_space_pos[front_mask]
        culled_rad = culled_rad[front_mask]
        original_indices = np.array(candidate_indices)[front_mask.cpu().numpy()]

        # Project points to screen space (assuming simple intrinsics if not provided)
        if intrinsics is None:
            H, W = image_size
            focal = 1.2 * W
            K = torch.tensor([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], device=self.device, dtype=self.dtype)
        else:
            K = intrinsics.to(self.device, dtype=self.dtype)

        # Perspective projection: u = K * p_cam
        uvz = torch.matmul(cam_space_pos, K.T)
        uv = uvz[:, :2] / (uvz[:, 2].unsqueeze(-1) + 1e-6)

        # Create a simplified Z-buffer (depth buffer)
        z_buffer = torch.full(image_size, float('inf'), device=self.device, dtype=self.dtype)
        index_buffer = torch.full(image_size, -1, dtype=torch.long, device=self.device)

        # Sort surfels by depth (front-to-back) for correct occlusion
        depths = cam_space_pos[:, 2]
        sorted_depth_indices = torch.argsort(depths)

        # Rasterize splats (simplified as squares for efficiency)
        for idx in sorted_depth_indices:
            u, v = uv[idx]
            depth = depths[idx]
            # FIX: Correctly calculate projected radius using focal length from K matrix
            focal_length = K[0, 0]  # Extract focal length from intrinsics matrix
            radius_proj = (culled_rad[idx] * focal_length / (depth + 1e-6)).int()
            
            # Define pixel bounds for the splat
            # FIX: Correctly use image_size tuple indices
            u_min, u_max = max(0, int(u - radius_proj)), min(image_size[1] - 1, int(u + radius_proj))
            v_min, v_max = max(0, int(v - radius_proj)), min(image_size[0] - 1, int(v + radius_proj))

            if u_min >= u_max or v_min >= v_max: continue

            # Z-buffer test: update pixels if the current surfel is closer
            patch = z_buffer[v_min:v_max, u_min:u_max]
            visible_pixels = patch > depth
            
            z_buffer[v_min:v_max, u_min:u_max][visible_pixels] = depth
            index_buffer[v_min:v_max, u_min:u_max][visible_pixels] = original_indices[idx]

        # 3. Voting based on the visible surfels in the index buffer [1, 1]
        visible_surfel_indices = index_buffer[index_buffer!= -1].cpu().numpy()
        if len(visible_surfel_indices) == 0:
            return list(range(max(0, len(self.view_database) - k), len(self.view_database)))

        vote_counts = {}
        for surfel_idx in visible_surfel_indices:
            for view_idx in self.surfel_to_views[surfel_idx]:
                vote_counts[view_idx] = vote_counts.get(view_idx, 0) + 1
        
        if not vote_counts:
            return list(range(max(0, len(self.view_database) - k), len(self.view_database)))

        # 4. Filtering and Selection (Non-Maximum Suppression on poses) [1, 1]
        sorted_views = sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)
        
        selected_indices = []
        selected_poses = []
        pose_dist_thresh = 2.0 # Heuristic distance threshold

        for view_idx, _ in sorted_views:
            if len(selected_indices) >= k: break
            
            is_redundant = False
            current_pose_c2w = self.view_database[view_idx][1]
            for sel_pose in selected_poses:
                if torch.linalg.norm(current_pose_c2w[:3, 3] - sel_pose[:3, 3]) < pose_dist_thresh:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_indices.append(view_idx)
                selected_poses.append(current_pose_c2w)

        # Pad if not enough unique views were found
        if len(selected_indices) < k:
            recent_indices = reversed(range(len(self.view_database)))
            for idx in recent_indices:
                if len(selected_indices) >= k: break
                if idx not in selected_indices:
                    selected_indices.append(idx)
        
        print(f"Retrieved indices: {selected_indices}")
        return selected_indices
    
    def export_world_visualization(self, output_dir: str = "memory_viz"):
        """
        Export the current world representation for visualization.
        
        Args:
            output_dir: Directory to save visualization files
        """
        try:
            from .memory_visualizer import export_memory_visualization
            return export_memory_visualization(self, output_dir)
        except ImportError:
            print("Warning: trimesh not available. Install with: pip install trimesh")
            return None
    
    def visualize_retrieval(self, target_c2w: torch.Tensor, k: int = 4, output_path: str = None):
        """
        Visualize which views would be retrieved for a given target pose.
        
        Args:
            target_c2w: Target camera-to-world matrix
            k: Number of views to retrieve
            output_path: Optional path to save visualization
        """
        try:
            from .memory_visualizer import VGGTMemoryVisualizer
            visualizer = VGGTMemoryVisualizer(self)
            return visualizer.visualize_retrieval_for_pose(target_c2w, k, output_path)
        except ImportError:
            print("Warning: trimesh not available. Install with: pip install trimesh")
            return None
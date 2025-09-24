# algorithms/worldmem/vggt_memory_retriever.py

import torch
import numpy as np
from scipy.spatial import KDTree
from vggt.models.vggt import VGGT
from.geometry_utils import unproject_depth_to_pointcloud, pointcloud_to_surfels
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

class VGGTMemoryRetriever:
    """
    Manages a geometrically-grounded memory using VGGT and a surfel index,
    inspired by the VMem architecture.[1, 1]
    """
    def __init__(self, device, d_thresh=0.1, n_thresh=0.9, downsample_factor=4):
        self.device = device
        self.dtype = torch.float16
        
        # Load the pre-trained VGGT model [9]
        print("Loading VGGT model...")
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device, dtype=self.dtype)
        self.vggt_model.eval()
        print("VGGT model loaded.")

        # Memory stores
        self.view_database = []  # Stores (raw_frame_tensor, c2w_matrix)
        self.surfels = None      # Dict storing all surfel positions, normals, radii
        self.surfel_to_views = [] # List of lists, mapping surfel index to view indices
        self.kdtree = None       # KD-Tree for efficient spatial queries [8]

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
        new_h = ((h + 13) // 14) * 14  # Round up to nearest multiple of 14
        new_w = ((w + 13) // 14) * 14  # Round up to nearest multiple of 14
        
        if h != new_h or w != new_w:
            import torch.nn.functional as F
            resized_frame = F.interpolate(
                new_frame_tensor.unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            resized_frame = new_frame_tensor
            
        vggt_input = resized_frame.unsqueeze(0).to(self.device, dtype=self.dtype)
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.vggt_model(vggt_input)

        depth = predictions["depth"]
        pose_enc = predictions["pose_enc"]

        # 2. Unproject depth to get accurate point cloud [1, 1]
        pointcloud = unproject_depth_to_pointcloud(depth, pose_enc, image_size_hw)
        
        # 3. Convert point cloud to surfels [1, 1]
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        camera_params = {'extrinsics': extrinsics, 'intrinsics': intrinsics}
        new_positions, new_normals, new_radii = pointcloud_to_surfels(
            pointcloud, camera_params, self.downsample_factor
        )

        # 4. Merge new surfels into the global index [1, 1]
        if self.surfels is None:
            self.surfels = {'pos': new_positions, 'norm': new_normals, 'rad': new_radii}
            self.surfel_to_views = [[frame_index] for _ in range(len(new_positions))]
        else:
            if self.kdtree is not None:
                distances, indices = self.kdtree.query(new_positions.cpu().numpy(), k=5)
                
                for i in range(len(new_positions)):
                    match_found = False
                    for dist, idx in zip(distances[i], indices[i]):
                        if dist < self.d_thresh and torch.dot(new_normals[i], self.surfels['norm'][idx]) > self.n_thresh:
                            if frame_index not in self.surfel_to_views[idx]:
                                self.surfel_to_views[idx].append(frame_index)
                            match_found = True
                            break
                    
                    if not match_found:
                        self.surfels['pos'] = torch.cat([self.surfels['pos'], new_positions[i].unsqueeze(0)])
                        self.surfels['norm'] = torch.cat([self.surfels['norm'], new_normals[i].unsqueeze(0)])
                        self.surfels['rad'] = torch.cat([self.surfels['rad'], new_radii[i].unsqueeze(0)])
                        self.surfel_to_views.append([frame_index])

        # Rebuild KD-Tree after updates
        self.kdtree = KDTree(self.surfels['pos'].cpu().numpy())
        print(f"Memory updated. Total surfels: {len(self.surfels['pos'])}")

    @torch.no_grad()
    def retrieve_relevant_views(self, target_c2w, k=4):
        """
        The "Read from Memory" pipeline. Retrieves top-K views using occlusion-aware rendering.
        """
        if self.surfels is None or len(self.view_database) <= k:
            # Fallback for early stages: return most recent frames
            indices = list(range(max(0, len(self.view_database) - k), len(self.view_database)))
            while len(indices) < k: indices.append(0) # Pad if necessary
            return indices

        print(f"Retrieving from memory for target pose...")
        
        # Simplified simulation of the GPU rendering pipeline described in the research.[1]
        
        # 1. Broad-phase culling (simplified to radius search)
        cam_center = target_c2w[:3, 3]
        candidate_indices = self.kdtree.query_ball_point(cam_center.cpu().numpy(), r=50.0)
        
        if not candidate_indices:
            return list(range(max(0, len(self.view_database) - k), len(self.view_database)))

        culled_pos = self.surfels['pos'][candidate_indices]
        
        # 2. Visibility Check (Simplified Z-buffering) [1, 1]
        w2c = torch.linalg.inv(target_c2w.to(self.device))
        cam_space_pos = torch.matmul(w2c[:3, :3], culled_pos.T).T + w2c[:3, 3]
        visible_mask = cam_space_pos[:, 2] > 0.1 # Check if in front of camera
        
        if not torch.any(visible_mask):
            return list(range(max(0, len(self.view_database) - k), len(self.view_database)))

        # 3. Voting [1, 1]
        visible_surfel_indices = np.array(candidate_indices)[visible_mask.cpu().numpy()]
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
        pose_dist_thresh = 2.0

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
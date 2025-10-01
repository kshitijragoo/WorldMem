# worldmem/algorithms/worldmem/vggt_memory_retriever.py

import torch
import numpy as np
from collections import namedtuple, Counter
from scipy.spatial import cKDTree # Using cKDTree as a stand-in for an octree for efficient search

# Import from our newly refactored, reliable geometry utility file
from. import geometry_utils as geo

# Define a simple structure for a surfel
Surfel = namedtuple('Surfel', ['position', 'normal', 'radius', 'view_indices'])

class VGGTSurfelMemory:
    """
    Implements the Surfel-Indexed View Memory (VMem) strategy.
    This class manages the "Writing" of new views into a surfel-based geometric index
    and the "Reading" of relevant past views to condition future frame generation.
    """
    def __init__(self, downsample_factor=4, merge_dist_threshold=0.1, merge_norm_threshold=0.9):
        self.surfels = []
        self.view_memory = {} # Stores {view_index: (image, pose_encoding)}
        self.downsample_factor = downsample_factor
        self.merge_dist_threshold = merge_dist_threshold
        self.merge_norm_threshold = merge_norm_threshold
        self.surfel_kdtree = None

    def _update_kdtree(self):
        """Rebuilds the k-d tree for fast nearest-neighbor search of surfels."""
        if not self.surfels:
            self.surfel_kdtree = None
            return
        positions = np.array([s.position for s in self.surfels])
        self.surfel_kdtree = cKDTree(positions)

    def add_view(self, view_index, image, pose_encoding, depth_map, image_size_hw):
        """
        The "Writing" module. Processes a new view, generates surfels,
        and updates the memory bank.
        """
        try:
            print(f"[add_view] view_index={view_index}")
            print("[add_view] image:",
                  f"type={type(image)}, tensor_shape={tuple(image.shape) if hasattr(image,'shape') else None}")
            print("[add_view] pose_encoding:",
                  f"shape={tuple(pose_encoding.shape) if hasattr(pose_encoding,'shape') else None}, dtype={getattr(pose_encoding,'dtype',None)}")
            print("[add_view] depth_map:",
                  f"shape={tuple(depth_map.shape) if hasattr(depth_map,'shape') else None}, dtype={getattr(depth_map,'dtype',None)}")
            print("[add_view] image_size_hw:", image_size_hw)
        except Exception as _e:
            print("[add_view][debug] input print failed:", _e)
        # 1. Store the raw view data
        self.view_memory[view_index] = (image, pose_encoding)

        # 2. Derive the accurate point map using our validated geometry utility
        point_map = geo.unproject_depth_to_points(depth_map, pose_encoding, image_size_hw).squeeze()
        try:
            pm = point_map
            print("[add_view] point_map:", f"shape={tuple(pm.shape)}, dtype={pm.dtype}, device={pm.device}")
            flat = pm.reshape(-1, 3)
            print("[add_view] point_map stats:",
                  f"min={flat.min(dim=0).values.detach().cpu().tolist()}",
                  f"max={flat.max(dim=0).values.detach().cpu().tolist()}")
        except Exception as _e:
            print("[add_view][debug] point_map stats failed:", _e)
        H, W, _ = point_map.shape
        
        # 3. Downsample the point map to create surfels
        points = point_map[::self.downsample_factor, ::self.downsample_factor, :].reshape(-1, 3)
        try:
            print("[add_view] downsampled points:", f"count={points.shape[0]}")
        except Exception:
            pass
        
        # 4. Create and merge new surfels
        new_surfels_to_add = []
        skipped_invalid = 0
        for r in range(0, H - 1, self.downsample_factor):
            for c in range(0, W - 1, self.downsample_factor):
                p_center = point_map[r, c]
                if torch.all(p_center == 0):
                    skipped_invalid += 1
                    continue # Skip invalid points

                # Calculate normal using cross-product of neighbors (VMem heuristic)
                p_right = point_map[r, c + 1]
                p_down = point_map[r + 1, c]
                vec_h = p_right - p_center
                vec_v = p_down - p_center
                normal = torch.cross(vec_h, vec_v)
                normal = torch.nn.functional.normalize(normal, dim=0)

                # Calculate radius (VMem heuristic)
                # This is a simplified version; a full implementation would use focal length from intrinsics
                radius = torch.norm(p_center) * 0.05 

                # 5. Merge with existing surfels
                if self.surfel_kdtree:
                    dist, idx = self.surfel_kdtree.query(p_center.cpu().numpy())
                    if dist < self.merge_dist_threshold:
                        existing_surfel = self.surfels[idx]
                        cos_sim = torch.dot(normal, torch.tensor(existing_surfel.normal, device=normal.device, dtype=normal.dtype))
                        if cos_sim > self.merge_norm_threshold:
                            # Add current view index to the existing surfel
                            existing_surfel.view_indices.add(view_index)
                            continue # Skip adding a new surfel

                # If no similar surfel found, create a new one
                new_surfels_to_add.append(Surfel(
                    position=p_center.cpu().numpy(),
                    normal=normal.cpu().numpy(),
                    radius=radius.item(),
                    view_indices={view_index}
                ))
        
        self.surfels.extend(new_surfels_to_add)
        self._update_kdtree()
        try:
            print(f"[add_view] new_surfels_added={len(new_surfels_to_add)}, skipped_invalid={skipped_invalid}")
        except Exception:
            pass
        print(f"Memory updated. View {view_index} added. Total surfels: {len(self.surfels)}")

    def retrieve_relevant_views(self, target_pose_encoding, image_size_hw, top_k=4):
        """
        The "Reading" module. Renders surfels from a target pose to find the most
        relevant past views for conditioning, correctly handling occlusions with a Z-buffer.
        """
        print(f"[retrieve] total_surfels={len(self.surfels)}")
        if not self.surfels:
            return

        device = target_pose_encoding.device
        H, W = image_size_hw

        # 1. Project all surfel positions into the target camera's view to get pixel coords and depth
        surfel_positions = torch.tensor(np.array([s.position for s in self.surfels]), dtype=torch.float32, device=device)
        pixel_coords, cam_points = geo.project_points_to_camera(surfel_positions, target_pose_encoding, image_size_hw)
        surfel_depths = cam_points[:, 2] # Z-coordinate in camera space is depth
        try:
            print("[retrieve] pixel_coords:", f"shape={tuple(pixel_coords.shape)}")
            print("[retrieve] depth stats:", float(surfel_depths.min().item()), float(surfel_depths.max().item()))
        except Exception:
            pass

        # 2. Initialize Z-buffer and an index buffer for rendering
        z_buffer = torch.full((H, W), float('inf'), device=device)
        index_buffer = torch.full((H, W), -1, dtype=torch.long, device=device)
        
        # Get camera focal length to project surfel radius into pixel space
        _, intrinsics = geo.get_camera_matrices_from_pose_encoding(target_pose_encoding, image_size_hw)
        focal_length = (intrinsics[0, 0] + intrinsics[1, 1]) / 2.0
        try:
            print("[retrieve] focal_length:", float(focal_length))
        except Exception:
            pass

        # 3. Render surfels via splatting with depth testing
        for i, surfel in enumerate(self.surfels):
            depth = surfel_depths[i]
            if depth <= 0: # Cull surfels behind the camera
                continue

            px, py = int(round(pixel_coords[i, 0].item())), int(round(pixel_coords[i, 1].item()))
            
            # Project surfel radius from world units to pixel units
            pixel_radius = int(round((surfel.radius * focal_length / depth).item()))
            if pixel_radius < 1: pixel_radius = 1

            # Define the bounding box for the splat
            x_min, x_max = max(0, px - pixel_radius), min(W, px + pixel_radius)
            y_min, y_max = max(0, py - pixel_radius), min(H, py + pixel_radius)

            # Iterate over pixels in the splat's bounding box and perform depth test
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Use a simple circular splat shape
                    if (x - px)**2 + (y - py)**2 <= pixel_radius**2:
                        if depth < z_buffer[y, x]:
                            z_buffer[y, x] = depth
                            index_buffer[y, x] = i
        
        # 4. Vote for views based on which surfels are visible
        vote_counter = Counter()
        visible_surfel_indices = torch.unique(index_buffer[index_buffer!= -1])
        print("[retrieve] visible_surfel_count:", int(visible_surfel_indices.numel()))
        for idx in visible_surfel_indices:
            vote_counter.update(self.surfels[idx.item()].view_indices)

        # 5. Select top-K most frequent views
        if not vote_counter:
            return
            
        most_common_views = vote_counter.most_common(top_k)
        retrieved_indices = [view_idx for view_idx, count in most_common_views]

        # 6. Retrieve the actual view data
        # A full implementation would also apply non-maximum suppression on poses here.
        retrieved_views = [self.view_memory[idx] for idx in retrieved_indices if idx in self.view_memory]

        print(f"Retrieved {len(retrieved_views)} views for conditioning: {retrieved_indices}")
        return retrieved_views
# WorldMem Memory Visualization

This document describes the memory visualization capabilities added to WorldMem, specifically for the VGGT-based geometric memory retrieval system.

## Overview

The memory visualization system allows you to export and visualize the 3D world representation built by the VGGT memory retriever during inference. This includes:

- **3D Surfel Representation**: The point cloud of surfels (surface elements) that represent the geometric memory
- **Camera Trajectories**: The positions and orientations of all camera views in memory
- **Memory Retrieval Visualization**: Which views are retrieved for specific target poses
- **Memory Evolution**: How the memory grows and changes over time

## Installation

Install the additional visualization dependencies:

```bash
pip install trimesh[easy]
```

This will install trimesh with support for GLB export and various 3D file formats.

## Usage

### Interactive Inference with Visualization

The `interactive()` method in `WorldMemMinecraft` now supports memory visualization:

```python
result = model.interactive(
    first_frame=first_frame,
    new_actions=new_actions,
    first_pose=first_pose,
    device=device,
    memory_latent_frames=None,
    memory_actions=None,
    memory_poses=None,
    memory_c2w=None,
    memory_frame_idx=None,
    memory_raw_frames=None,
    # NEW VISUALIZATION PARAMETERS:
    enable_memory_viz=True,                    # Enable visualization
    viz_output_dir="memory_viz",               # Output directory
    viz_interval=10                            # Export every N frames
)
```

### Parameters

- `enable_memory_viz` (bool): Enable/disable memory visualization exports
- `viz_output_dir` (str): Directory to save visualization files (default: "interactive_memory_viz")
- `viz_interval` (int): Export visualization every N frames (default: 10)

### Direct Visualization API

You can also directly export visualizations from a VGGT memory retriever:

```python
# Export complete world representation
visualizer = memory_retriever.export_world_visualization("output_dir")

# Visualize retrieval for a specific pose
memory_retriever.visualize_retrieval(target_c2w, k=4, output_path="retrieval.glb")
```

## Output Files

The visualization system creates several types of files:

### Directory Structure
```
memory_viz/
├── initial_state/           # Memory after first frame
│   ├── world_representation.glb
│   ├── surfels_pointcloud.ply
│   └── memory_analysis.json
├── step_0010/              # Periodic snapshots
│   ├── world_representation.glb
│   ├── retrieval_visualization.glb
│   └── memory_analysis.json
├── step_0020/
│   └── ...
└── final_state/            # Complete final memory
    ├── world_representation.glb
    ├── final_retrieval.glb
    ├── memory_analysis.json
    └── session_summary.json
```

### File Types

#### GLB Files (3D Scenes)
- `world_representation.glb`: Complete 3D scene with surfels and cameras
- `retrieval_visualization.glb`: Shows which views are retrieved for a target pose
- `final_retrieval.glb`: Retrieval visualization for the final pose

#### PLY Files (Point Clouds)
- `surfels_pointcloud.ply`: Raw surfel point cloud data

#### JSON Files (Analysis)
- `memory_analysis.json`: Detailed statistics about the memory state
- `session_summary.json`: Summary of the entire interactive session

## Visualization Features

### 3D World Representation

The GLB files contain:

- **Surfels**: Colored spheres representing surface elements
  - Red: Surfels seen by only 1 view
  - Orange: Surfels seen by 2 views  
  - Yellow: Surfels seen by 3-4 views
  - Green: Surfels seen by 5+ views (well-observed)

- **Camera Frustums**: Pyramid shapes showing camera positions and orientations
  - Different colors for each camera view
  - Size indicates relative importance

### Retrieval Visualization

Shows the memory retrieval process:
- **Target Camera**: Bright red frustum (the pose you're querying)
- **Retrieved Views**: Bright green frustums (selected by the retrieval algorithm)
- **Other Views**: Dimmed blue frustums (available but not selected)
- **Relevant Surfels**: Highlighted surfels that influenced the selection

## Viewing the Visualizations

### Option 1: Blender (Recommended)
1. Install [Blender](https://www.blender.org/)
2. Open Blender and go to File → Import → glTF 2.0 (.glb/.gltf)
3. Select your GLB file
4. Navigate the 3D scene to explore the memory representation

### Option 2: Online Viewers
- [glTF Viewer](https://gltf-viewer.donmccurdy.com/)
- [Three.js Editor](https://threejs.org/editor/)
- Upload your GLB file and explore interactively

### Option 3: Point Cloud Viewers (for PLY files)
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://cloudcompare.org/)

## Analysis Tools

### Memory Statistics

The `memory_analysis.json` files contain detailed statistics:

```json
{
  "total_surfels": 1250,
  "total_views": 15,
  "surfel_statistics": {
    "count": 1250,
    "position_bounds": {...},
    "radius_stats": {...}
  },
  "view_associations": {
    "min_views_per_surfel": 1,
    "max_views_per_surfel": 8,
    "mean_views_per_surfel": 2.3
  },
  "spatial_distribution": {...},
  "memory_coverage": {...}
}
```

### Session Summary

The `session_summary.json` provides an overview:

```json
{
  "total_frames_generated": 50,
  "total_memory_views": 25,
  "total_surfels": 2100,
  "visualization_interval": 10,
  "condition_index_method": "vggt_surfel"
}
```

## Standalone Visualization Script

Use the provided script for standalone visualization:

```bash
# Create demo visualization
python visualize_memory.py --demo --output_dir demo_viz

# Visualize from checkpoint (when implemented)
python visualize_memory.py --checkpoint model.ckpt --output_dir viz_output

# Include retrieval visualization for specific pose
python visualize_memory.py --demo --target_pose "0,0,5,0,90" --output_dir viz_with_retrieval
```

## Examples

See the example scripts:
- `interactive_with_viz_example.py`: How to use visualization in interactive sessions
- `example_memory_viz_integration.py`: Integration patterns for different use cases

## Performance Considerations

- Visualization export can be computationally expensive for large memories
- Use appropriate `viz_interval` values (10-20 frames) to balance detail vs. performance
- GLB files can become large with many surfels; consider the storage requirements
- The visualization is most useful with the VGGT surfel memory retrieval method

## Troubleshooting

### Common Issues

1. **"trimesh not available" warning**
   ```bash
   pip install trimesh[easy]
   ```

2. **Large file sizes**
   - Increase `viz_interval` to export less frequently
   - The surfel representation grows with scene complexity

3. **Visualization only works with VGGT**
   - Set `condition_index_method: "vggt_surfel"` in your config
   - Other methods (dinov3, knn, fov) don't have geometric representations to visualize

4. **Empty visualizations**
   - Ensure the memory retriever has been used (run some inference first)
   - Check that views have been added to memory

### Debug Information

The system prints debug information during export:
```
Memory visualization enabled. Output directory: memory_viz
Adding view 1 to geometric memory...
Memory updated. Total surfels: 156
Memory visualization exported at step 10 to: memory_viz/step_0010
```

This helps track the memory growth and export process.

## Future Enhancements

Potential future improvements:
- Real-time visualization during inference
- Interactive 3D exploration tools
- Comparison visualizations between different retrieval methods
- Animation of memory evolution over time
- Integration with other memory retrieval methods

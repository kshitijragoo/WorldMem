# VMem Surfel Memory Integration

This document describes the integration of VMem's surfel-based memory system into the WorldMem project, replacing custom implementations with the original VMem components.

## Overview

The integration replaces custom surfel memory implementations with VMem's complete pipeline, ensuring:
- **Consistency**: Uses VMem's proven surfel storage and retrieval algorithms
- **Reliability**: Avoids implementation errors by reusing tested code
- **Completeness**: Includes all VMem features like undo functionality and geometric optimization

## Key Components

### 1. VMemAdapter (`memory_adapter.py`)
- **Purpose**: Bridges WorldMem data structures with VMem's VMemPipeline
- **Key Methods**:
  - `initialize_with_frame()`: Initialize VMem with first frame
  - `generate_trajectory_frames()`: Generate new frames using VMem
  - `get_context_info()`: Retrieve relevant past views
  - `undo_latest_move()`: Undo last navigation step
  - `get_memory_stats()`: Get memory statistics

### 2. Updated df_video.py
- **Changes**: Replaced custom `VGGTMemoryRetriever` with `VMemAdapter`
- **Memory Handling**: VMem now handles all surfel creation, merging, and retrieval internally
- **Asynchronous Operations**: Removed custom threading since VMem handles updates efficiently

### 3. Geometry Utils (`geometry_utils.py`)
- **Purpose**: Import geometry utilities from VMem/VGGT instead of reimplementing
- **Imports**: 
  - VMem utilities: `tensor_to_pil`, `Surfel`, `Octree`, visualization functions
  - VGGT utilities: depth unprojection, camera projection functions

## Usage

### Basic Setup
```python
from memory_adapter import VMemAdapter

# Initialize adapter
adapter = VMemAdapter(device="cuda")

# Initialize with first frame
first_frame = convert_worldmem_image_to_vmem(image_tensor)
first_pose = convert_worldmem_pose_to_vmem(pose_matrix)
adapter.initialize_with_frame(first_frame, first_pose)
```

### Navigation
```python
# Generate trajectory frames
new_poses = [convert_worldmem_pose_to_vmem(pose) for pose in trajectory_poses]
new_Ks = [default_intrinsics] * len(new_poses)
new_frames = adapter.generate_trajectory_frames(new_poses, new_Ks)
```

### Memory Retrieval
```python
# Get relevant context for new viewpoint
target_poses = [target_pose_matrix]
context_info = adapter.get_context_info(target_poses)
relevant_frame_indices = context_info['context_time_indices']
```

## Configuration

The adapter uses VMem's configuration system. Key parameters:
- `memory_condition_length`: Number of context frames to retrieve
- `context_num_frames`: Maximum context window size
- `target_num_frames`: Number of frames generated per step
- Surfel parameters: merge thresholds, radius scaling, etc.

## Dependencies

### Required Packages
- VMem repository in `../../../vmem/`
- VGGT repository in `../../../vggt/`
- PyTorch, NumPy, OmegaConf
- CUT3R for point map estimation (handled by VMem)

### Path Setup
The adapter automatically adds VMem and VGGT to the Python path:
```python
vmem_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vmem"))
sys.path.insert(0, vmem_path)
```

## Removed Files

The following custom implementations were replaced:
- `vggt_surfel_memory_retriever.py`: Replaced by `VMemAdapter`
- Custom geometry functions in `geometry_utils.py`: Now import from VMem/VGGT
- Asynchronous memory update logic: VMem handles this internally

## Testing

Run the integration test:
```bash
cd algorithms/worldmem
python test_vmem_integration.py
```

This tests:
- VMemAdapter initialization and basic operations
- Geometry utilities imports and functionality
- VMem pipeline creation (may fail without model weights)

## Benefits

1. **Reduced Code Duplication**: ~600 lines of custom surfel code removed
2. **Improved Reliability**: Uses VMem's tested implementations
3. **Feature Completeness**: Access to all VMem features (undo, optimization, etc.)
4. **Easier Maintenance**: Updates to VMem automatically benefit WorldMem
5. **Consistency**: Identical behavior to VMem's proven system

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure VMem and VGGT repositories are in correct locations
2. **Model Loading**: VMem requires model weights from HuggingFace
3. **CUDA Errors**: Ensure compatible PyTorch/CUDA versions
4. **Path Issues**: Check that relative paths to VMem/VGGT are correct

### Configuration Issues
- Verify `vmem/configs/inference/inference.yaml` exists
- Check that all required model paths are accessible
- Ensure CUT3R dependencies are properly installed

## Migration Guide

For existing WorldMem code:

1. **Replace Custom Retrievers**:
   ```python
   # OLD
   self.vggt_retriever = VGGTMemoryRetriever(device=device)
   
   # NEW
   self.vmem_adapter = VMemAdapter(device=device)
   ```

2. **Update Memory Operations**:
   ```python
   # OLD
   self.vggt_retriever.add_view_to_memory(image, pose)
   indices = self.vggt_retriever.retrieve_relevant_views(target_pose, k=8)
   
   # NEW
   self.vmem_adapter.generate_trajectory_frames([pose], [K])
   context_info = self.vmem_adapter.get_context_info([target_pose])
   ```

3. **Update Geometry Imports**:
   ```python
   # OLD
   from .geometry_utils import unproject_depth_to_pointcloud
   
   # NEW
   from .geometry_utils import vggt_unproject as unproject_depth_to_pointcloud
   ```

This integration provides a robust foundation for surfel-based memory in WorldMem while leveraging VMem's proven implementation.

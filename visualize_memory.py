#!/usr/bin/env python3
"""
Script to visualize VGGT memory retriever's world representation.

This script can be used to export and visualize the 3D world representation
built by the VGGT memory retriever during inference or interactive sessions.

Usage:
    python visualize_memory.py --checkpoint path/to/model.ckpt --output_dir viz_output
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from algorithms.worldmem.vggt_memory_retriever import VGGTMemoryRetriever
from algorithms.worldmem.memory_visualizer import VGGTMemoryVisualizer, export_memory_visualization


def create_demo_memory():
    """Create a demo memory retriever with some sample data for testing."""
    retriever = VGGTMemoryRetriever(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add some dummy frames and poses for demonstration
    print("Creating demo memory with synthetic data...")
    
    # Generate some synthetic camera poses in a circle
    num_views = 10
    radius = 5.0
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Create synthetic camera pose
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.0
        
        # Create camera-to-world matrix
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[0, 3] = x  # X position
        c2w[1, 3] = y  # Y position
        c2w[2, 3] = z  # Z position
        
        # Look towards center
        forward = -torch.tensor([x, y, z], dtype=torch.float32)
        forward = forward / torch.norm(forward)
        right = torch.cross(forward, torch.tensor([0, 1, 0], dtype=torch.float32))
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # Negative Z is forward in camera coordinates
        
        # Create a synthetic RGB frame (random for demo)
        frame = torch.rand(3, 64, 64, dtype=torch.float32)
        
        # Add to memory (this will create synthetic surfels)
        try:
            retriever.add_view_to_memory(frame, c2w)
            print(f"Added view {i+1}/{num_views}")
        except Exception as e:
            print(f"Error adding view {i}: {e}")
            continue
    
    return retriever


def visualize_from_checkpoint(checkpoint_path: str, output_dir: str):
    """Load a trained model and visualize its memory state."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # This would need to be adapted based on your actual model loading code
    # For now, we'll create a demo memory
    print("Note: Checkpoint loading not implemented. Creating demo memory instead.")
    return create_demo_memory()


def main():
    parser = argparse.ArgumentParser(description='Visualize VGGT Memory Retriever')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='memory_viz', 
                       help='Output directory for visualization files')
    parser.add_argument('--demo', action='store_true', 
                       help='Create demo memory with synthetic data')
    parser.add_argument('--target_pose', type=str, 
                       help='Target pose for retrieval visualization (comma-separated: x,y,z,pitch,yaw)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get memory retriever
    if args.demo or not args.checkpoint:
        print("Creating demo memory retriever...")
        memory_retriever = create_demo_memory()
    else:
        memory_retriever = visualize_from_checkpoint(args.checkpoint, args.output_dir)
    
    if memory_retriever is None:
        print("Failed to create memory retriever.")
        return
    
    # Export comprehensive visualization
    print("Exporting world visualization...")
    try:
        visualizer = export_memory_visualization(memory_retriever, args.output_dir)
        print(f"World visualization exported to: {args.output_dir}")
        
        # Print summary
        if memory_retriever.surfels is not None:
            print(f"Total surfels: {len(memory_retriever.surfels['pos'])}")
            print(f"Total views: {len(memory_retriever.view_database)}")
        
        # If target pose is specified, visualize retrieval
        if args.target_pose:
            try:
                pose_values = [float(x.strip()) for x in args.target_pose.split(',')]
                if len(pose_values) != 5:
                    print("Target pose must have 5 values: x,y,z,pitch,yaw")
                    return
                
                # Create target camera-to-world matrix
                from algorithms.worldmem.df_video import euler_to_camera_to_world_matrix
                target_pose_tensor = torch.tensor(pose_values, dtype=torch.float32)
                target_c2w = euler_to_camera_to_world_matrix(target_pose_tensor)
                
                print(f"Visualizing retrieval for pose: {pose_values}")
                retrieval_output = str(output_path / "retrieval_visualization.glb")
                visualizer.visualize_retrieval_for_pose(
                    target_c2w, k=4, output_path=retrieval_output
                )
                
            except Exception as e:
                print(f"Error creating retrieval visualization: {e}")
        
        print("\nVisualization files created:")
        print(f"  - {args.output_dir}/world_representation.glb")
        print(f"  - {args.output_dir}/surfels_pointcloud.ply")
        print(f"  - {args.output_dir}/memory_analysis.json")
        if args.target_pose:
            print(f"  - {args.output_dir}/retrieval_visualization.glb")
        
        print("\nTo view GLB files, you can:")
        print("  1. Open them in Blender")
        print("  2. Use online viewers like https://gltf-viewer.donmccurdy.com/")
        print("  3. Use Three.js or other WebGL libraries")
        
    except ImportError:
        print("Error: trimesh package not found.")
        print("Install with: pip install trimesh")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()

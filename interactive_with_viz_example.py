#!/usr/bin/env python3
"""
Example: Using Interactive WorldMem with Memory Visualization

This script demonstrates how to use the new memory visualization features
during interactive inference sessions.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Your existing imports (adapt these to your actual setup)
from algorithms.worldmem.df_video import WorldMemMinecraft
from omegaconf import DictConfig


def create_sample_config():
    """Create a sample configuration for WorldMem."""
    # This is a minimal config - adapt to match your actual configuration structure
    cfg = DictConfig({
        'n_frames': 16,
        'frame_stack': 1,
        'memory_condition_length': 8,
        'pose_cond_dim': 5,
        'use_plucker': True,
        'relative_embedding': True,
        'add_timestamp_embedding': True,
        'condition_index_method': 'vggt_surfel',  # Enable VGGT for visualization
        'log_video': True,
        'focal_length': 0.35,
        'require_pose_prediction': True,
        'next_frame_length': 1,
        'diffusion': {},  # Add your diffusion config here
        'noise_level': 'random_all'
    })
    return cfg


def run_interactive_with_visualization():
    """
    Example of running interactive inference with memory visualization enabled.
    """
    print("Setting up WorldMem with memory visualization...")
    
    # Create model (you'll need to adapt this to your actual model loading)
    cfg = create_sample_config()
    # model = WorldMemMinecraft(cfg)  # Uncomment when you have proper model loading
    
    # For this example, we'll create dummy data
    print("Creating sample data for demonstration...")
    
    # Sample first frame (3, H, W) - RGB image
    first_frame = np.random.rand(3, 256, 256).astype(np.float32)
    
    # Sample first pose (x, y, z, pitch, yaw)
    first_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Sample actions for the sequence (N, action_dim)
    num_steps = 20
    action_dim = 7  # Adapt to your action space
    new_actions = torch.randn(num_steps, action_dim)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running interactive inference on {device}...")
    print(f"Generating {num_steps} frames with visualization enabled")
    
    # This is where you would call your model's interactive method
    # with visualization enabled:
    
    """
    result = model.interactive(
        first_frame=first_frame,
        new_actions=new_actions,
        first_pose=first_pose,
        device=device,
        memory_latent_frames=None,  # Start fresh
        memory_actions=None,
        memory_poses=None,
        memory_c2w=None,
        memory_frame_idx=None,
        memory_raw_frames=None,
        # NEW VISUALIZATION PARAMETERS:
        enable_memory_viz=True,                    # Enable visualization
        viz_output_dir="interactive_viz_demo",     # Output directory
        viz_interval=5                             # Export every 5 frames
    )
    
    print("Interactive inference completed!")
    print("Visualization files saved to: interactive_viz_demo/")
    print("\nGenerated visualization files:")
    print("  - initial_state/: Memory state after first frame")
    print("  - step_XXXX/: Memory state at regular intervals")
    print("  - final_state/: Complete final memory representation")
    print("  - final_state/session_summary.json: Session statistics")
    
    # The result contains the same data as before:
    generated_frames, memory_latent_frames, memory_actions, memory_poses, memory_c2w, memory_frame_idx, memory_raw_frames = result
    
    print(f"Generated {len(generated_frames)} frames")
    print(f"Final memory contains {len(memory_poses)} total poses")
    """
    
    print("Example setup complete. Uncomment the model.interactive() call to run actual inference.")


def analyze_visualization_results(viz_dir="interactive_viz_demo"):
    """
    Example of how to analyze the visualization results after inference.
    """
    viz_path = Path(viz_dir)
    
    if not viz_path.exists():
        print(f"Visualization directory {viz_dir} not found.")
        return
    
    print(f"Analyzing visualization results in: {viz_path}")
    
    # Check what visualization files were created
    subdirs = [d for d in viz_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} visualization snapshots:")
    for subdir in sorted(subdirs):
        print(f"  - {subdir.name}")
        
        # Check contents of each snapshot
        glb_files = list(subdir.glob("*.glb"))
        ply_files = list(subdir.glob("*.ply"))
        json_files = list(subdir.glob("*.json"))
        
        if glb_files:
            print(f"    GLB files: {[f.name for f in glb_files]}")
        if ply_files:
            print(f"    PLY files: {[f.name for f in ply_files]}")
        if json_files:
            print(f"    JSON files: {[f.name for f in json_files]}")
    
    # Load and display session summary if available
    summary_path = viz_path / "final_state" / "session_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print("\nSession Summary:")
        print(f"  Total frames generated: {summary.get('total_frames_generated', 'N/A')}")
        print(f"  Total memory views: {summary.get('total_memory_views', 'N/A')}")
        print(f"  Total surfels: {summary.get('total_surfels', 'N/A')}")
        print(f"  Visualization interval: {summary.get('visualization_interval', 'N/A')}")
        print(f"  Condition index method: {summary.get('condition_index_method', 'N/A')}")
    
    print("\nTo view the 3D visualizations:")
    print("1. Install Blender and open the .glb files")
    print("2. Use online viewers like https://gltf-viewer.donmccurdy.com/")
    print("3. Use MeshLab or CloudCompare for .ply point clouds")


def compare_memory_evolution(viz_dir="interactive_viz_demo"):
    """
    Example of how to compare memory evolution over time.
    """
    viz_path = Path(viz_dir)
    
    if not viz_path.exists():
        print(f"Visualization directory {viz_dir} not found.")
        return
    
    # Find all step directories
    step_dirs = [d for d in viz_path.iterdir() if d.is_dir() and d.name.startswith('step_')]
    step_dirs.sort()
    
    print(f"Analyzing memory evolution across {len(step_dirs)} steps:")
    
    for step_dir in step_dirs:
        analysis_file = step_dir / "memory_analysis.json"
        if analysis_file.exists():
            import json
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            step_num = step_dir.name.replace('step_', '')
            total_surfels = analysis.get('surfel_statistics', {}).get('count', 0)
            total_views = analysis.get('total_views', 0)
            
            print(f"  Step {step_num}: {total_views} views, {total_surfels} surfels")
    
    print("\nThis shows how the memory representation grows over time.")
    print("You can open the corresponding .glb files to see the 3D evolution.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive WorldMem with Visualization Demo')
    parser.add_argument('--run', action='store_true', help='Run the interactive inference demo')
    parser.add_argument('--analyze', type=str, help='Analyze visualization results from directory')
    parser.add_argument('--compare', type=str, help='Compare memory evolution from directory')
    
    args = parser.parse_args()
    
    if args.run:
        run_interactive_with_visualization()
    elif args.analyze:
        analyze_visualization_results(args.analyze)
    elif args.compare:
        compare_memory_evolution(args.compare)
    else:
        print("Usage:")
        print("  python interactive_with_viz_example.py --run")
        print("  python interactive_with_viz_example.py --analyze interactive_viz_demo")
        print("  python interactive_with_viz_example.py --compare interactive_viz_demo")

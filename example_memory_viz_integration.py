"""
Example: Integrating Memory Visualization into WorldMem Inference

This example shows how to add memory visualization capabilities to your 
existing WorldMem inference pipeline.
"""

import torch
import numpy as np
from pathlib import Path

# Your existing imports
from algorithms.worldmem.df_video import WorldMemMinecraft
from algorithms.worldmem.vggt_memory_retriever import VGGTMemoryRetriever


def enhanced_validation_step_with_visualization(model, batch, batch_idx, viz_output_dir="memory_viz"):
    """
    Enhanced validation step that includes memory visualization.
    
    This is an example of how you could modify your validation_step method
    to include periodic memory visualization exports.
    """
    # Run the normal validation step
    result = model.validation_step(batch, batch_idx)
    
    # Add memory visualization every N batches
    if batch_idx % 10 == 0 and hasattr(model, 'vggt_retriever'):
        print(f"Exporting memory visualization for batch {batch_idx}...")
        
        # Create batch-specific output directory
        batch_viz_dir = Path(viz_output_dir) / f"batch_{batch_idx}"
        batch_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Export the current state of the memory
        try:
            model.vggt_retriever.export_world_visualization(str(batch_viz_dir))
            print(f"Memory visualization saved to: {batch_viz_dir}")
        except Exception as e:
            print(f"Warning: Could not export memory visualization: {e}")
    
    return result


def interactive_session_with_visualization(model, output_dir="interactive_viz"):
    """
    Example of an interactive session with memory visualization.
    
    This shows how you could add visualization capabilities to your
    interactive inference sessions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize your interactive session
    first_frame = np.random.rand(3, 256, 256).astype(np.float32)  # Example frame
    first_pose = np.array([0, 0, 0, 0, 0], dtype=np.float32)     # Example pose
    new_actions = torch.rand(10, 7)  # Example actions
    device = next(model.parameters()).device
    
    # Run interactive inference
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
        memory_raw_frames=None
    )
    
    # Export memory visualization after interactive session
    if hasattr(model, 'vggt_retriever'):
        print("Exporting final memory state...")
        model.vggt_retriever.export_world_visualization(str(output_path / "final_memory"))
        
        # Visualize retrieval for the last pose
        if len(result) >= 5 and result[4] is not None:  # memory_c2w
            memory_c2w = torch.from_numpy(result[4])
            if len(memory_c2w) > 0:
                last_c2w = memory_c2w[-1, 0]  # Last pose, first batch item
                model.vggt_retriever.visualize_retrieval(
                    last_c2w, 
                    k=4, 
                    output_path=str(output_path / "last_retrieval.glb")
                )
                print(f"Retrieval visualization saved to: {output_path / 'last_retrieval.glb'}")
    
    return result


def analyze_memory_evolution(model, sequence_data, output_dir="memory_evolution"):
    """
    Analyze how the memory evolves over a sequence.
    
    This function processes a sequence and exports memory visualizations
    at different time steps to show how the world representation grows.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not hasattr(model, 'vggt_retriever'):
        print("Model does not have VGGT retriever. Skipping memory analysis.")
        return
    
    print("Analyzing memory evolution...")
    
    # Process sequence in chunks and visualize memory at key points
    checkpoint_intervals = [0.25, 0.5, 0.75, 1.0]  # Export at 25%, 50%, 75%, 100%
    sequence_length = len(sequence_data)
    
    for i, checkpoint in enumerate(checkpoint_intervals):
        frame_idx = int(checkpoint * sequence_length) - 1
        if frame_idx < 0:
            continue
            
        # Process up to this frame (this is pseudocode - adapt to your data format)
        # for j in range(frame_idx + 1):
        #     process_frame(model, sequence_data[j])
        
        # Export memory state
        checkpoint_dir = output_path / f"checkpoint_{i+1}_{int(checkpoint*100)}percent"
        checkpoint_dir.mkdir(exist_ok=True)
        
        try:
            visualizer = model.vggt_retriever.export_world_visualization(str(checkpoint_dir))
            
            # Also export analysis
            if visualizer:
                analysis_path = checkpoint_dir / "analysis.json"
                visualizer.export_memory_analysis(str(analysis_path))
                
            print(f"Checkpoint {i+1} visualization saved to: {checkpoint_dir}")
            
        except Exception as e:
            print(f"Error at checkpoint {i+1}: {e}")
    
    print(f"Memory evolution analysis complete. Results in: {output_path}")


def debug_retrieval_quality(model, target_poses, output_dir="retrieval_debug"):
    """
    Debug the quality of memory retrieval for specific target poses.
    
    This function helps you understand which views are being retrieved
    for different target poses and whether the retrieval makes sense.
    """
    if not hasattr(model, 'vggt_retriever'):
        print("Model does not have VGGT retriever. Cannot debug retrieval.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Debugging retrieval quality...")
    
    for i, target_pose in enumerate(target_poses):
        # Convert pose to camera-to-world matrix
        if isinstance(target_pose, (list, tuple, np.ndarray)):
            from algorithms.worldmem.df_video import euler_to_camera_to_world_matrix
            target_pose_tensor = torch.tensor(target_pose, dtype=torch.float32)
            target_c2w = euler_to_camera_to_world_matrix(target_pose_tensor)
        else:
            target_c2w = target_pose
        
        # Visualize retrieval
        output_file = output_path / f"retrieval_debug_{i}.glb"
        retrieval_info = model.vggt_retriever.visualize_retrieval(
            target_c2w, k=4, output_path=str(output_file)
        )
        
        if retrieval_info:
            print(f"Target pose {i}: Retrieved views {retrieval_info['retrieved_indices']}")
            
            # Save retrieval info as JSON
            import json
            info_file = output_path / f"retrieval_info_{i}.json"
            with open(info_file, 'w') as f:
                json.dump(retrieval_info, f, indent=2)
    
    print(f"Retrieval debugging complete. Results in: {output_path}")


# Example usage in your main inference script:
def main_with_visualization():
    """
    Example of how to integrate visualization into your main inference loop.
    """
    # Load your model (adapt this to your actual model loading code)
    # model = load_your_worldmem_model()
    
    # Enable memory visualization if using VGGT retrieval
    # if hasattr(model, 'vggt_retriever'):
    #     print("Memory visualization enabled!")
    
    # Your existing inference code here...
    # results = run_inference(model, data)
    
    # Add visualization exports
    # enhanced_validation_step_with_visualization(model, batch, batch_idx)
    # interactive_session_with_visualization(model)
    # analyze_memory_evolution(model, sequence_data)
    
    print("Inference complete with memory visualization!")


if __name__ == "__main__":
    main_with_visualization()
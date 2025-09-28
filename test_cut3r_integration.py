#!/usr/bin/env python3
"""
Test script to verify VGGT-based surfel memory retrieval integration.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_vggt_memory_retriever():
    """Test the VGGT-based surfel memory retriever."""
    try:
        from algorithms.worldmem.vggt_memory_retriever import VGGTSurfelMemoryRetriever
        print("✓ Successfully imported VGGTSurfelMemoryRetriever")
        
        # Test initialization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Note: This will try to download the model if not available
        retriever = VGGTSurfelMemoryRetriever(device=device)
        print("✓ Successfully initialized VGGTSurfelMemoryRetriever")
        
        # Create a dummy image and camera pose for testing
        dummy_image = torch.rand(3, 256, 256)  # RGB image
        dummy_pose = torch.eye(4)  # Identity camera pose
        
        print("Testing add_view_to_memory...")
        retriever.add_view_to_memory(dummy_image, dummy_pose, view_index=0)
        print("✓ Successfully added view to memory")
        
        # Test retrieval
        print("Testing retrieve_relevant_views...")
        target_pose = torch.eye(4)
        target_pose[0, 3] = 0.1  # Slight translation
        
        retrieved_indices = retriever.retrieve_relevant_views(target_pose, k=1)
        print(f"✓ Retrieved indices: {retrieved_indices}")
        
        # Get memory stats
        stats = retriever.get_memory_stats()
        print(f"✓ Memory stats: {stats}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure the vggt directory is accessible and VGGT dependencies are installed")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_worldmem_integration():
    """Test the integration with WorldMemMinecraft."""
    try:
        from algorithms.worldmem.df_video import WorldMemMinecraft
        from omegaconf import DictConfig
        
        print("✓ Successfully imported WorldMemMinecraft")
        
        # Create a minimal config for testing
        cfg = DictConfig({
            'n_frames': 16,
            'frame_stack': 1,
            'memory_condition_length': 4,
            'pose_cond_dim': 5,
            'use_plucker': True,
            'relative_embedding': True,
            'state_embed_only_on_qk': True,
            'use_memory_attention': True,
            'add_timestamp_embedding': True,
            'ref_mode': 'sequential',
            'log_curve': False,
            'focal_length': 0.35,
            'log_video': False,
            'self_consistency_eval': False,
            'next_frame_length': 1,
            'require_pose_prediction': False,
            'condition_index_method': 'vggt_surfel',  # Use our new method
            'dinov3_model_id': 'facebook/dinov2-large',
            'diffusion': DictConfig({
                'architecture': DictConfig({
                    'network_size': 64,
                    'attn_heads': 4,
                    'attn_dim_head': 64,
                    'dim_mults': [1, 2, 4, 8],
                    'resolution': 64,
                    'attn_resolutions': [16, 32, 64, 128],
                    'use_init_temporal_attn': True,
                    'use_linear_attn': True,
                    'time_emb_type': 'rotary'
                }),
                'beta_schedule': 'sigmoid',
                'objective': 'pred_v',
                'use_fused_snr': True,
                'cum_snr_decay': 0.96,
                'clip_noise': 20.0,
                'sampling_timesteps': 20,
                'ddim_sampling_eta': 0.0,
                'stabilization_level': 15
            })
        })
        
        print("Testing WorldMemMinecraft initialization with vggt_surfel...")
        # Note: This will try to initialize the full model, which requires more dependencies
        model = WorldMemMinecraft(cfg)
        print("✓ Successfully initialized WorldMemMinecraft with VGGT surfel memory")
        
        # Check if the VGGT retriever was properly initialized
        if hasattr(model, 'vggt_surfel_retriever'):
            print("✓ VGGT surfel retriever properly initialized")
            stats = model.vggt_surfel_retriever.get_memory_stats()
            print(f"✓ Initial memory stats: {stats}")
        else:
            print("✗ VGGT surfel retriever not found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Testing VGGT-based surfel memory integration")
    print("="*60)
    
    print("\n1. Testing VGGTSurfelMemoryRetriever...")
    test1_passed = test_vggt_memory_retriever()
    
    print("\n2. Testing WorldMemMinecraft integration...")
    test2_passed = test_worldmem_integration()
    
    print("\n" + "="*60)
    print("Test Results:")
    print(f"VGGTSurfelMemoryRetriever: {'PASS' if test1_passed else 'FAIL'}")
    print(f"WorldMemMinecraft integration: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! VGGT integration is working.")
        print("\nNext steps:")
        print("1. Make sure VGGT model weights are downloaded")
        print("2. Test with real video data")
        print("3. Compare performance with other memory retrieval methods")
        print("4. VGGT should be significantly faster than CUT3R!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure vggt directory is accessible")
        print("2. Install VGGT dependencies")
        print("3. Check CUDA availability if using GPU")
    
    print("="*60)

if __name__ == "__main__":
    main()

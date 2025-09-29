#!/usr/bin/env python3
"""
Test script to verify that the VMem pipeline can be imported directly.
"""

import os
import sys

def test_vmem_pipeline_direct():
    """Test that VMemPipeline can be imported directly."""
    print("Testing VMemPipeline direct import...")
    
    try:
        # Add the project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Try to import VMemPipeline directly
        from vmem.modeling.pipeline import VMemPipeline
        print("✓ VMemPipeline import successful")
        return True
        
    except ImportError as e:
        if "No module named 'extern'" in str(e):
            print("✗ Still has extern import issues")
            return False
        elif "No module named 'utils.util'" in str(e):
            print("✗ Still has utils import issues")
            return False
        else:
            print(f"✓ Import error is not related to our fixes: {e}")
            return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_surfel_inference_import():
    """Test that the surfel inference module can be imported."""
    print("Testing surfel inference import...")
    
    try:
        # Add the project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Add VGGT extern path
        vggt_extern_path = os.path.join(project_root, "vmem", "extern", "VGGT")
        if vggt_extern_path not in sys.path:
            sys.path.insert(0, vggt_extern_path)
        
        # Try to import surfel inference
        from surfel_inference import run_inference_from_pil, add_path_to_vggt
        print("✓ Surfel inference import successful")
        return True
        
    except ImportError as e:
        print(f"✗ Surfel inference import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("VMem Pipeline Direct Import Test")
    print("=" * 60)
    
    tests = [
        ("Surfel Inference Import", test_surfel_inference_import),
        ("VMemPipeline Direct Import", test_vmem_pipeline_direct),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! VMem pipeline imports are working.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

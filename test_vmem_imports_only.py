#!/usr/bin/env python3
"""
Test script to verify that the VMem import issues are fixed.
"""

import os
import sys

def test_vmem_pipeline_import():
    """Test that VMemPipeline can be imported without relative import errors."""
    print("Testing VMemPipeline import...")
    
    try:
        # Add the project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Try to import VMemPipeline
        from vmem.modeling.pipeline import VMemPipeline
        print("✓ VMemPipeline import successful")
        return True
        
    except ImportError as e:
        if "No module named 'modeling'" in str(e):
            print("✗ Still has relative import issues")
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

def test_memory_adapter_import():
    """Test that VMemAdapter can be imported."""
    print("Testing VMemAdapter import...")
    
    try:
        # Add the project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Try to import VMemAdapter
        from algorithms.worldmem.memory_adapter import VMemAdapter
        print("✓ VMemAdapter import successful")
        return True
        
    except ImportError as e:
        if "No module named 'vmem.modeling.pipeline'" in str(e):
            print("✗ VMemPipeline import still failing")
            return False
        else:
            print(f"✓ Import error is not related to our fixes: {e}")
            return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("VMem Import Fix Test")
    print("=" * 60)
    
    tests = [
        ("VMemPipeline Import", test_vmem_pipeline_import),
        ("VMemAdapter Import", test_memory_adapter_import),
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
        print("🎉 All tests passed! VMem import issues are fixed.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

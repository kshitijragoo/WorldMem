#!/usr/bin/env python3
"""
Test script to verify that the import syntax fixes work.
"""

import os
import sys

def test_import_syntax_fixes():
    """Test that the import syntax is correct."""
    print("Testing import syntax fixes...")
    
    # Test VMem pipeline imports
    vmem_pipeline_path = "/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/vmem/modeling/pipeline.py"
    
    if os.path.exists(vmem_pipeline_path):
        with open(vmem_pipeline_path, 'r') as f:
            content = f.read()
            
        if "from utils.util import" in content:
            print("✓ VMem pipeline uses correct import syntax: utils.util")
        else:
            print("✗ VMem pipeline import syntax incorrect")
            return False
    else:
        print("✗ VMem pipeline file not found")
        return False
    
    # Test memory adapter imports
    adapter_path = "/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/worldmem/algorithms/worldmem/memory_adapter.py"
    
    if os.path.exists(adapter_path):
        with open(adapter_path, 'r') as f:
            content = f.read()
            
        if "from utils.util import" in content:
            print("✓ Memory adapter uses correct import syntax: utils.util")
        else:
            print("✗ Memory adapter import syntax incorrect")
            return False
    else:
        print("✗ Memory adapter file not found")
        return False
    
    return True

def test_import_path_setup():
    """Test that the import path setup is correct."""
    print("Testing import path setup...")
    
    # Check VMem pipeline path setup
    vmem_pipeline_path = "/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/vmem/modeling/pipeline.py"
    
    if os.path.exists(vmem_pipeline_path):
        with open(vmem_pipeline_path, 'r') as f:
            content = f.read()
            
        if "vmem_utils_path = os.path.join(os.path.dirname(__file__), \"..\", \"utils\")" in content:
            print("✓ VMem pipeline has correct path setup")
        else:
            print("✗ VMem pipeline path setup incorrect")
            return False
    
    # Check memory adapter path setup
    adapter_path = "/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/worldmem/algorithms/worldmem/memory_adapter.py"
    
    if os.path.exists(adapter_path):
        with open(adapter_path, 'r') as f:
            content = f.read()
            
        if "vmem_path = os.path.abspath(os.path.join(os.path.dirname(__file__), \"../../../vmem\"))" in content:
            print("✓ Memory adapter has correct path setup")
        else:
            print("✗ Memory adapter path setup incorrect")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Import Syntax Fix Test")
    print("=" * 60)
    
    tests = [
        ("Import Syntax Fixes", test_import_syntax_fixes),
        ("Import Path Setup", test_import_path_setup),
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
        print("🎉 All tests passed! Import syntax is fixed.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

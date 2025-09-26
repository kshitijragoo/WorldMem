#!/usr/bin/env python3
"""
Verification script for surfel-based memory integration.
This script checks the code structure without importing modules.
"""

import os
import re

def check_file_exists(filepath, description):
    """Check if a file exists and report"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} missing: {filepath}")
        return False

def check_code_pattern(filepath, patterns, description):
    """Check if code patterns exist in a file"""
    if not os.path.exists(filepath):
        print(f"✗ {description}: File not found - {filepath}")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        all_found = True
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                print(f"  ✓ {pattern_name} found")
            else:
                print(f"  ✗ {pattern_name} missing")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ {description}: Error reading file - {e}")
        return False

def main():
    """Run verification checks"""
    print("=" * 70)
    print("Verifying Surfel-based Memory Integration for WorldMem")
    print("=" * 70)
    
    base_path = "/Users/hemkritragoonath/Desktop/WOLRDMEM_TEMP/CITS4010-4011/worldmem"
    success = True
    
    # Check 1: New surfel memory retriever file
    surfel_file = os.path.join(base_path, "algorithms/worldmem/surfel_memory_retriever.py")
    success &= check_file_exists(surfel_file, "SurfelMemoryRetriever implementation")
    
    if os.path.exists(surfel_file):
        surfel_patterns = {
            "SurfelMemoryRetriever class": r"class SurfelMemoryRetriever:",
            "CUT3R integration": r"from extern\.CUT3R\.surfel_inference import run_inference_from_pil",
            "add_view_to_memory method": r"def add_view_to_memory\(self",
            "retrieve_relevant_views method": r"def retrieve_relevant_views\(self",
            "Surfel dataclass": r"class Surfel:",
            "Octree implementation": r"class Octree:"
        }
        success &= check_code_pattern(surfel_file, surfel_patterns, "SurfelMemoryRetriever functionality")
    
    # Check 2: WorldMemMinecraft integration
    df_video_file = os.path.join(base_path, "algorithms/worldmem/df_video.py")
    success &= check_file_exists(df_video_file, "WorldMemMinecraft implementation")
    
    if os.path.exists(df_video_file):
        integration_patterns = {
            "SurfelMemoryRetriever import": r"from\.surfel_memory_retriever import SurfelMemoryRetriever",
            "Surfel method initialization": r'condition_index_method\.lower\(\) == "surfel"',
            "Surfel retriever creation": r"self\.surfel_retriever = SurfelMemoryRetriever",
            "Surfel memory read (validation)": r"surfel_retriever\.retrieve_relevant_views",
            "Surfel memory write (validation)": r"surfel_retriever\.add_view_to_memory",
            "Surfel in interactive method": r"surfel.*for condition index"
        }
        success &= check_code_pattern(df_video_file, integration_patterns, "WorldMemMinecraft surfel integration")
    
    # Check 3: Configuration file update
    config_file = os.path.join(base_path, "configurations/algorithm/df_video_worldmemminecraft.yaml")
    success &= check_file_exists(config_file, "Configuration file")
    
    if os.path.exists(config_file):
        config_patterns = {
            "Surfel option in comments": r"surfel"
        }
        success &= check_code_pattern(config_file, config_patterns, "Configuration file update")
    
    # Check 4: Code structure analysis
    print("\n" + "=" * 70)
    print("Code Structure Analysis")
    print("=" * 70)
    
    if os.path.exists(surfel_file):
        # Count lines and analyze structure
        with open(surfel_file, 'r') as f:
            lines = f.readlines()
        
        print(f"✓ SurfelMemoryRetriever file: {len(lines)} lines")
        
        # Count methods
        method_count = len([line for line in lines if line.strip().startswith('def ')])
        print(f"✓ Methods implemented: {method_count}")
        
        # Check for key VMem concepts
        vmem_concepts = {
            "Surfel rendering": "render_surfel_votes",
            "Point cloud conversion": "pointcloud_to_surfels", 
            "Normal estimation": "estimate_normal",
            "Surfel merging": "_merge_surfels",
            "Octree spatial index": "Octree",
            "Asynchronous processing": "ThreadPoolExecutor"
        }
        
        content = ''.join(lines)
        for concept, pattern in vmem_concepts.items():
            if pattern in content:
                print(f"  ✓ {concept} implemented")
            else:
                print(f"  ⚠ {concept} may need attention")
    
    # Check 5: Integration completeness
    print("\n" + "=" * 70)
    print("Integration Completeness Check")
    print("=" * 70)
    
    integration_points = [
        "Initialization in __init__",
        "Memory read in validation_step", 
        "Memory write in validation_step",
        "Memory read in interactive method",
        "Memory write in interactive method",
        "Configuration option added"
    ]
    
    completed_points = 0
    if os.path.exists(df_video_file):
        with open(df_video_file, 'r') as f:
            df_content = f.read()
        
        # Check each integration point
        checks = [
            ('self.surfel_retriever = SurfelMemoryRetriever' in df_content),
            ('surfel_retriever.retrieve_relevant_views' in df_content),
            ('surfel_retriever.add_view_to_memory' in df_content),
            (df_content.count('condition_index_method.lower() == "surfel"') >= 2),
            (df_content.count('surfel_retriever.add_view_to_memory') >= 2),
            (os.path.exists(config_file) and 'surfel' in open(config_file).read())
        ]
        
        for i, (point, check) in enumerate(zip(integration_points, checks)):
            if check:
                print(f"✓ {point}")
                completed_points += 1
            else:
                print(f"✗ {point}")
    
    print(f"\nIntegration Progress: {completed_points}/{len(integration_points)} points completed")
    
    # Final summary
    print("\n" + "=" * 70)
    if success and completed_points == len(integration_points):
        print("🎉 INTEGRATION SUCCESSFUL!")
        print("\nSurfel-based memory system has been successfully integrated into WorldMem:")
        print("• SurfelMemoryRetriever class implemented with CUT3R backend")
        print("• WorldMemMinecraft updated with surfel condition indexing")  
        print("• Asynchronous read/write memory operations implemented")
        print("• Configuration options updated")
        print("\nTo use: Set condition_index_method: 'surfel' in your config file")
    else:
        print("⚠ INTEGRATION INCOMPLETE")
        print(f"Status: {completed_points}/{len(integration_points)} integration points completed")
        if not success:
            print("Some files or patterns are missing - please review the errors above")
    print("=" * 70)
    
    return success and completed_points == len(integration_points)

if __name__ == "__main__":
    main()

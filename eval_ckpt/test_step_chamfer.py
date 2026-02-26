#!/usr/bin/env python3
"""
Test script for step_chamfer_reward.py

This script tests the chamfer distance calculation functionality using sample STEP files.
"""

import os
import sys
from step_chamfer_reward import process_step_files

# Sample STEP file content (minimal valid STEP file)
SAMPLE_STEP_1 = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('sample1.step', '2024-01-01T00:00:00', (''), (''), '', '', '');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (0.0, 0.0, 0.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = AXIS2_PLACEMENT_3D('', #1, #2, $);
#4 = CYLINDRICAL_SURFACE('', #3, 1.0);
ENDSEC;
END-ISO-10303-21;
"""

SAMPLE_STEP_2 = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('sample2.step', '2024-01-01T00:00:00', (''), (''), '', '', '');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('', (1.0, 1.0, 1.0));
#2 = DIRECTION('', (0.0, 0.0, 1.0));
#3 = AXIS2_PLACEMENT_3D('', #1, #2, $);
#4 = CYLINDRICAL_SURFACE('', #3, 1.5);
ENDSEC;
END-ISO-10303-21;
"""

def test_basic_functionality():
    """Test basic functionality with sample STEP contents."""
    print("Testing basic functionality...")
    
    # Test parameters
    lower_bound = 0.5
    upper_bound = 2.0
    
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print()
    
    # Test the processing function
    chamfer_distance, reward = process_step_files(
        SAMPLE_STEP_1, SAMPLE_STEP_2, 
        lower_bound, upper_bound,
        scale_normalize=True,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"Chamfer Distance: {chamfer_distance}")
    print(f"Reward: {reward}")
    
    # Test reward calculation logic
    print(f"\nTesting reward calculation logic:")
    
    # Test case 1: Distance below lower bound -> reward = 1.0
    test_distance = 0.3
    expected_reward = 1.0
    actual_reward = 1.0 if test_distance <= lower_bound else 1.0 - (test_distance - lower_bound) / (upper_bound - lower_bound)
    print(f"Distance {test_distance} -> Expected reward: {expected_reward}, Actual: {actual_reward}")
    
    # Test case 2: Distance between bounds -> linear interpolation
    test_distance = 1.25
    expected_reward = 1.0 - (test_distance - lower_bound) / (upper_bound - lower_bound)
    actual_reward = 1.0 if test_distance <= lower_bound else max(0.0, min(1.0, 1.0 - (test_distance - lower_bound) / (upper_bound - lower_bound)))
    print(f"Distance {test_distance} -> Expected reward: {expected_reward:.3f}, Actual: {actual_reward:.3f}")
    
    # Test case 3: Distance above upper bound -> reward = 0.0
    test_distance = 2.5
    expected_reward = 0.0
    actual_reward = 0.0 if test_distance > upper_bound else (1.0 if test_distance <= lower_bound else 1.0 - (test_distance - lower_bound) / (upper_bound - lower_bound))
    print(f"Distance {test_distance} -> Expected reward: {expected_reward}, Actual: {actual_reward}")


def test_with_real_files():
    """Test with real STEP files if available."""
    print("\n" + "="*50)
    print("Testing with real STEP files...")
    
    # Look for STEP files in the expected directory structure
    # UPDATE these paths to directories containing generated STEP files to evaluate:
    test_dirs = [
        "./data/STEP_generated/eval_output",
    ]
    
    step_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"Searching in: {test_dir}")
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.endswith('.step'):
                        step_files.append(os.path.join(root, file))
                        if len(step_files) >= 2:  # We only need 2 files for testing
                            break
                if len(step_files) >= 2:
                    break
            if len(step_files) >= 2:
                break
    
    if len(step_files) >= 2:
        print(f"Found STEP files: {step_files[:2]}")
        
        try:
            with open(step_files[0], 'r') as f:
                content1 = f.read()
            with open(step_files[1], 'r') as f:
                content2 = f.read()
            
            print("Testing with real STEP files...")
            chamfer_distance, reward = process_step_files(
                content1, content2,
                0.5, 2.0,
                scale_normalize=True,
                verbose=True
            )
            
            print(f"\nReal file results:")
            print(f"Chamfer Distance: {chamfer_distance}")
            print(f"Reward: {reward}")
            
        except Exception as e:
            print(f"Error reading real STEP files: {e}")
    else:
        print("No real STEP files found for testing")


if __name__ == "__main__":
    print("Step Chamfer Reward - Test Script")
    print("="*50)
    
    try:
        test_basic_functionality()
        test_with_real_files()
        
        print("\n" + "="*50)
        print("Test completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)










import numpy as np
import os
from typing import List, Set

# Hardcoded list of zero shear stress cases (from find_zero_cases.py output)
ZERO_STRESS_CASES = {
    44, 60, 102, 106, 129, 177, 224, 229, 237, 260, 286, 293, 304, 313, 320, 369, 
    421, 436, 492, 503, 505, 511, 576, 578, 581, 588, 610, 612, 706, 715, 716, 
    718, 744, 771, 774, 775, 779, 884, 914, 915, 985, 998, 1025, 1033, 1062, 
    1073, 1099, 1106, 1135, 1172, 1178, 1183, 1199, 1250, 1253, 1288, 1289, 
    1342, 1370, 1371, 1443, 1469, 1508, 1547, 1548, 1558, 1580, 1600, 1629, 
    1670, 1676, 1703, 1734, 1761, 1765, 1773, 1803, 1873, 1877, 1922, 1952, 
    2032, 2035, 2042, 2051, 2146, 2186, 2273, 2360, 2363, 2388, 2408, 2409, 
    2415, 2447, 2482, 2486, 2541, 2549, 2575, 2579, 2607, 2654, 2661, 2671, 
    2744, 2748, 2778, 2828, 2834, 2849, 2866, 2929
}

def extract_case_index_from_filename(filename: str) -> int:
    """
    Extract case index from filename like 'processed_geometry_data_44.npz'
    Returns -1 if cannot extract index
    """
    try:
        # Handle different possible filename patterns
        basename = os.path.basename(filename)
        if 'processed_geometry_data_' in basename:
            # Extract number between 'processed_geometry_data_' and '.npz'
            start = basename.find('processed_geometry_data_') + len('processed_geometry_data_')
            end = basename.find('.npz')
            if end == -1:
                end = len(basename)
            return int(basename[start:end])
        elif 'transformed_data_' in basename:
            start = basename.find('transformed_data_') + len('transformed_data_')
            end = basename.find('.npz')
            if end == -1:
                end = len(basename)
            return int(basename[start:end])
        else:
            # Try to extract any number from filename
            import re
            numbers = re.findall(r'\d+', basename)
            if numbers:
                return int(numbers[-1])  # Take the last number found
            return -1
    except (ValueError, AttributeError):
        return -1

def filter_zero_stress_files(npz_files: List[str]) -> List[str]:
    """
    Filter out files that correspond to zero shear stress cases
    
    Args:
        npz_files: List of NPZ file paths
        
    Returns:
        List of filtered NPZ file paths (excluding zero stress cases)
    """
    filtered_files = []
    excluded_count = 0
    
    for file_path in npz_files:
        case_index = extract_case_index_from_filename(file_path)
        
        if case_index in ZERO_STRESS_CASES:
            excluded_count += 1
            print(f"Excluding zero stress case: {case_index} ({os.path.basename(file_path)})")
        else:
            filtered_files.append(file_path)
    
    print(f"\nZero stress case filtering summary:")
    print(f"  Original files: {len(npz_files)}")
    print(f"  Excluded files: {excluded_count}")
    print(f"  Remaining files: {len(filtered_files)}")
    print(f"  Exclusion rate: {excluded_count/len(npz_files)*100:.1f}%")
    
    return filtered_files

def verify_zero_case_exclusion(npz_files: List[str]) -> bool:
    """
    Verify that no zero stress cases remain in the file list
    
    Args:
        npz_files: List of NPZ file paths to verify
        
    Returns:
        True if all zero cases are excluded, False otherwise
    """
    remaining_zero_cases = []
    
    for file_path in npz_files:
        case_index = extract_case_index_from_filename(file_path)
        if case_index in ZERO_STRESS_CASES:
            remaining_zero_cases.append(case_index)
    
    if remaining_zero_cases:
        print(f"WARNING: Found {len(remaining_zero_cases)} zero stress cases still in dataset:")
        print(f"Cases: {remaining_zero_cases}")
        return False
    else:
        print("✓ Verification passed: No zero stress cases found in dataset")
        return True

def get_zero_stress_case_count() -> int:
    """Return the total number of known zero stress cases"""
    return len(ZERO_STRESS_CASES)

def is_zero_stress_case(case_index: int) -> bool:
    """Check if a given case index is a zero stress case"""
    return case_index in ZERO_STRESS_CASES
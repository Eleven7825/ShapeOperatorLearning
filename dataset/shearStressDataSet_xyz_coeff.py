#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
import re
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy

class ShearStressDataset(Dataset):
    """Dataset that loads data with spherical harmonic coefficients instead of A matrices"""
    def __init__(self, npz_files, coefficients_file, **kwargs):
        """
        Initialize the dataset

        Args:
            npz_files: List of NPZ files to load
            coefficients_file: Path to the NPZ file containing spherical harmonic coefficients
            **kwargs:
                ref_xyz_vtk_path: Path to the VTK file containing reference xyz coordinates
                value_key: NPZ key to load as target (default: 'transformed_values')
                out_dim: Number of output components to keep (default: 3)
        """
        super().__init__()
        values_list = []
        coeff_list = []
        
        if 'ref_xyz_vtk_path' not in kwargs:
            raise ValueError("ref_xyz_vtk_path must be provided via config['data']['ref_xyz_vtk_path']")
        ref_xyz_vtk_path = kwargs['ref_xyz_vtk_path']
        value_key = kwargs.get('value_key', 'transformed_values')
        out_dim   = kwargs.get('out_dim', 3)
        print(f"Using reference VTK file: {ref_xyz_vtk_path}")
        
        # Load reference xyz coordinates from VTK file
        try:
            ref_xyz = self._load_vtk_points(ref_xyz_vtk_path)
            print(f"Loaded reference xyz coordinates with shape: {ref_xyz.shape}")
        except Exception as e:
            print(f"Error loading VTK file: {e}")
            raise
        
        # Load coefficients data
        print(f"Loading coefficients from {coefficients_file}...")
        coeff_data = np.load(coefficients_file)
        coefficients = coeff_data['coefficients']
        case_numbers = coeff_data['case_numbers']
        
        # Create a mapping from case number to coefficients
        coeff_mapping = {int(case): coeffs for case, coeffs in zip(case_numbers, coefficients)}
        
        print(f"Loaded coefficients for {len(coeff_mapping)} cases")
        print(f"Each coefficient vector has length: {coefficients.shape[1]}")
        
        # Regular expression to extract case number from file path - INCLUDING THE NEW PATTERN
        case_pattern = re.compile(r'(processed_spheroid_data_|transformed_data_|processed_geometry_data_|processed_TAA_data_)(\d+)\.npz')
        
        print(f"Loading {len(npz_files)} data files...")
        loaded_files = 0
        skipped_files = 0
        missing_coeffs = 0
        
        for i, file in enumerate(npz_files):
            try:
                # Print progress every 100 files
                if i % 100 == 0:
                    print(f"Processing file {i}/{len(npz_files)}...")
                
                # Extract case number from filename
                case_match = case_pattern.search(file)
                if not case_match:
                    print(f"Could not extract case number from {file}, skipping...")
                    skipped_files += 1
                    continue
                
                # The case number is in the second capture group because the first is the prefix pattern
                case_number = int(case_match.group(2))
                
                # Check if we have coefficients for this case
                if case_number not in coeff_mapping:
                    print(f"No coefficients found for case {case_number}, skipping...")
                    missing_coeffs += 1
                    skipped_files += 1
                    continue
                
                # Load data from NPZ file
                data = np.load(file)
                
                # Try to get transformed_values, or interpolated_values as fallback
                try:
                    if value_key in data:
                        values = data[value_key]
                        if i == 0:
                            print(f"Using '{value_key}' from files")
                    elif value_key == 'transformed_values' and 'interpolated_values' in data:
                        values = data['interpolated_values']
                        if i == 0:
                            print(f"Falling back to 'interpolated_values'")
                    else:
                        print(f"Key '{value_key}' not found in {file}, skipping...")
                        skipped_files += 1
                        continue
                except Exception as e:
                    print(f"Error loading values from {file}: {e}")
                    skipped_files += 1
                    continue

                # Ensure 2D: (N,) → (N, 1)
                if values.ndim == 1:
                    values = values[:, None]

                if np.all(values == 0):
                    print(f"Skipping {file} due to all its values are zero!")
                    skipped_files += 1
                    continue

                values = values[:, :out_dim]
                
                # Get coefficients for this case
                case_coeffs = coeff_mapping[case_number]
                
                # Repeat coefficients for each point in the data
                n_points = len(ref_xyz)
                
                # Make sure the values have the right shape
                if len(values) != n_points:
                    print(f"Warning: Values length ({len(values)}) doesn't match ref_xyz length ({n_points}) for file {file}")
                    # Use the minimum length to be safe
                    min_length = min(len(values), n_points)
                    values = values[:min_length]
                    ref_xyz_used = ref_xyz[:min_length]
                    n_points = min_length
                else:
                    ref_xyz_used = ref_xyz
                
                repeated_coeffs = np.tile(case_coeffs, (n_points, 1))
                
                # Store data
                values_list.append(values)
                coeff_list.append(repeated_coeffs)
                
                loaded_files += 1
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                skipped_files += 1
                continue
        
        print(f"Dataset loading complete.")
        print(f"Loaded {loaded_files} files, skipped {skipped_files} files.")
        print(f"Files with missing coefficients: {missing_coeffs}")
        
        if loaded_files == 0:
            raise ValueError("No files were successfully loaded. Check your file paths and patterns.")
        
        # Concatenate all data
        self.values = np.concatenate(values_list, axis=0)
        self.coefficients = np.concatenate(coeff_list, axis=0)
        
        # Create repeated xyz array for all data points
        if loaded_files > 0:
            # Get the number of points from the first loaded file
            points_per_file = len(values_list[0])
            self.xyz = np.zeros((len(self.values), 3), dtype=np.float32)
            
            # Fill in the xyz values
            current_idx = 0
            for i in range(loaded_files):
                current_file_points = len(values_list[i])
                self.xyz[current_idx:current_idx+current_file_points] = ref_xyz[:current_file_points]
                current_idx += current_file_points
        else:
            self.xyz = np.array([])
        
        # Convert to PyTorch tensors
        self.xyz = torch.FloatTensor(self.xyz)
        self.values = torch.FloatTensor(self.values)
        self.coefficients = torch.FloatTensor(self.coefficients)

        # Store the number of successfully loaded files for train/val split calculation
        self.loaded_files = loaded_files

        print(f"Dataset created with {len(self.xyz)} samples.")
        print(f"xyz shape: {self.xyz.shape}")
        print(f"values shape: {self.values.shape}")
        print(f"coefficients shape: {self.coefficients.shape}")
    
    def _load_vtk_points(self, vtk_file_path):
        """
        Load points from a VTK file with improved error handling.
        
        Args:
            vtk_file_path: Path to the VTK file
            
        Returns:
            numpy array of points/vertices
        """
        # Expand user path
        vtk_file_path = os.path.expanduser(vtk_file_path)
        
        # Check if file exists
        if not os.path.exists(vtk_file_path):
            raise FileNotFoundError(f"VTK file not found: {vtk_file_path}")
        
        try:
            # Create VTK reader
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(vtk_file_path)
            
            # Read the file (skip CanReadFile check for VTK compatibility)
            reader.Update()
            
            # Get polydata
            polydata = reader.GetOutput()
            if polydata is None:
                raise ValueError("Failed to get polydata from VTK file")
            
            # Get points
            points = polydata.GetPoints()
            if points is None:
                raise ValueError("No points found in VTK file")
            
            n_points = points.GetNumberOfPoints()
            if n_points == 0:
                raise ValueError("VTK file contains 0 points")
            
            print(f"Successfully reading VTK file with {n_points} points")
            
            # Method 1: Try using vtk_to_numpy (more efficient)
            try:
                points_data = points.GetData()
                points_array = vtk_to_numpy(points_data)
                print("Used efficient vtk_to_numpy method")
                
                # Ensure it's the right shape
                if points_array.shape[1] != 3:
                    raise ValueError(f"Expected 3D points, got shape {points_array.shape}")
                
                return points_array.astype(np.float32)
                
            except Exception as vtk_numpy_error:
                print(f"vtk_to_numpy failed: {vtk_numpy_error}")
                print("Falling back to manual extraction...")
                
                # Method 2: Manual extraction (fallback)
                points_array = np.zeros((n_points, 3), dtype=np.float32)
                for i in range(n_points):
                    point = points.GetPoint(i)
                    if len(point) != 3:
                        raise ValueError(f"Point {i} has wrong dimension: {len(point)}")
                    points_array[i] = point
                
                print("Used manual extraction method")
                return points_array
                
        except Exception as e:
            print(f"Error loading VTK points: {e}")
            print(f"File path: {vtk_file_path}")
            
            # Additional debugging info
            if os.path.exists(vtk_file_path):
                print(f"File size: {os.path.getsize(vtk_file_path)} bytes")
                
                # Try to read first few lines to check format
                try:
                    with open(vtk_file_path, 'r') as f:
                        first_lines = [f.readline().strip() for _ in range(10)]
                        print("First few lines of VTK file:")
                        for i, line in enumerate(first_lines):
                            print(f"  {i+1}: {line}")
                except:
                    print("Could not read file as text")
            
            raise
        
    def __len__(self):
        return len(self.xyz)
    
    def __getitem__(self, idx):
        return (
            self.coefficients[idx],
            self.xyz[idx],
            self.values[idx]
        )

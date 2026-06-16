#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
import re
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.decomposition import TruncatedSVD

class ShearStressDatasetSVD(Dataset):
    """Dataset that loads data with SVD-based geometry representation instead of direct coefficients"""
    def __init__(self, npz_files, n_components=10, **kwargs):
        """
        Initialize the dataset
        
        Args:
            npz_files: List of NPZ files to load
            n_components: Number of SVD components to use (default: 10)
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        values_list = []
        geometry_list = []
        
        self.n_components = n_components
        print(f"Using {n_components} SVD components for geometry representation")
        
        # Regular expression to extract case number from file path
        case_pattern = re.compile(r'(processed_spheroid_data_|transformed_data_|processed_geometry_data_)(\d+)\.npz')
        
        print(f"Loading {len(npz_files)} data files...")
        loaded_files = 0
        skipped_files = 0
        
        # Store all data for processing
        all_initial_momentum = []
        case_numbers = []
        case_to_values = {}  # Map case number to values
        
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
                
                case_number = int(case_match.group(2))
                
                # Load data from NPZ file first
                data = np.load(file)
                
                # Try to get transformed_values, or interpolated_values as fallback
                try:
                    if 'transformed_values' in data:
                        values = data['transformed_values']
                        if loaded_files == 0:  # Only print for the first successful file
                            print(f"Using transformed_values from files")
                    elif 'interpolated_values' in data:
                        values = data['interpolated_values']
                        if loaded_files == 0:  # Only print for the first successful file
                            print(f"Using interpolated_values from files")
                    else:
                        print(f"No values found in {file}, skipping...")
                        skipped_files += 1
                        continue
                except Exception as e:
                    print(f"Error loading values from {file}: {e}")
                    skipped_files += 1
                    continue
                
                if np.all(values == 0):
                    print(f"Skipping {file} due to all its values are zero!")
                    skipped_files += 1
                    continue
                
                # Only use the first 3 components of values (if there are more)
                values = values[:, 0:3]
                
                # Load initial momentum from VTK file
                momentum_file = os.path.expanduser(f"~/10tb/shiyi/vecPrj/Apr1/matchings/matching_{case_number}/initial_momentum.vtk")
                
                if not os.path.exists(momentum_file):
                    print(f"Initial momentum file not found for case {case_number}, skipping...")
                    skipped_files += 1
                    continue
                
                try:
                    initial_momentum = self._load_vtk_points(momentum_file)
                    all_initial_momentum.append(initial_momentum.flatten())
                    case_numbers.append(case_number)
                    case_to_values[case_number] = values
                except Exception as e:
                    print(f"Error loading momentum file for case {case_number}: {e}")
                    skipped_files += 1
                    continue
                
                loaded_files += 1
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                skipped_files += 1
                continue
        
        print(f"Dataset loading complete.")
        print(f"Loaded {loaded_files} files, skipped {skipped_files} files.")
        
        if loaded_files == 0:
            raise ValueError("No files were successfully loaded. Check your file paths and patterns.")
        
        # Fit SVD on all initial momentum vectors
        print(f"Fitting SVD on {len(all_initial_momentum)} initial momentum vectors...")
        all_momentum_matrix = np.array(all_initial_momentum)
        print(f"Initial momentum matrix shape: {all_momentum_matrix.shape}")
        
        # Fit SVD
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(all_momentum_matrix)
        
        print(f"SVD explained variance ratio: {self.svd.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(self.svd.explained_variance_ratio_):.4f}")
        
        # Transform each initial momentum vector to low-dimensional representation
        svd_representations = []
        values_list = []
        xyz_coords_list = []
        case_to_coords = {}  # Map case number to coordinates
        
        # Load coordinates from NPZ files
        for i, file in enumerate(npz_files):
            case_match = case_pattern.search(file)
            if case_match:
                case_number = int(case_match.group(2))
                if case_number in case_numbers:
                    try:
                        data = np.load(file)
                        if 'target_points' in data:
                            coords = data['target_points']
                            case_to_coords[case_number] = coords
                    except:
                        continue
        
        for i, case_num in enumerate(case_numbers):
            momentum_vector = all_initial_momentum[i].reshape(1, -1)
            svd_repr = self.svd.transform(momentum_vector).flatten()
            
            # Get the corresponding values and coordinates for this case
            values = case_to_values[case_num]
            if case_num not in case_to_coords:
                print(f"No coordinates found for case {case_num}, skipping...")
                continue
            coords = case_to_coords[case_num]
            
            values_list.append(values)
            xyz_coords_list.append(coords)
            
            # Get the number of points from the values
            n_points = len(values)
            
            # Repeat SVD representation for each point
            repeated_svd = np.tile(svd_repr, (n_points, 1))
            svd_representations.append(repeated_svd)
        
        # Concatenate all data
        self.values = np.concatenate(values_list, axis=0)
        self.geometry_svd = np.concatenate(svd_representations, axis=0)
        self.xyz_coords = np.concatenate(xyz_coords_list, axis=0)
        
        # Convert to PyTorch tensors
        self.values = torch.FloatTensor(self.values)
        self.geometry_svd = torch.FloatTensor(self.geometry_svd)
        self.xyz_coords = torch.FloatTensor(self.xyz_coords)
        
        print(f"Dataset created with {len(self.values)} samples.")
        print(f"values shape: {self.values.shape}")
        print(f"geometry_svd shape: {self.geometry_svd.shape}")
        print(f"xyz_coords shape: {self.xyz_coords.shape}")
    
    def _load_vtk_points(self, vtk_file_path):
        """
        Load points from a VTK file with improved error handling.
        Supports both POLYDATA and STRUCTURED_GRID formats.
        
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
            # First, determine the dataset type by reading the file header
            dataset_type = None
            with open(vtk_file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line_num > 10:  # Don't read too far
                        break
                    if line.strip().startswith('DATASET'):
                        dataset_type = line.strip().split()[-1].upper()
                        break
            
            print(f"Detected dataset type: {dataset_type}")
            
            # Choose appropriate reader based on dataset type
            if dataset_type == 'STRUCTURED_GRID':
                reader = vtk.vtkStructuredGridReader()
            elif dataset_type == 'POLYDATA':
                reader = vtk.vtkPolyDataReader()
            else:
                # Try generic reader as fallback
                reader = vtk.vtkDataSetReader()
            
            reader.SetFileName(vtk_file_path)
            reader.Update()
            
            # Get the output dataset
            dataset = reader.GetOutput()
            if dataset is None:
                raise ValueError("Failed to get dataset from VTK file")
            
            # Get points from the dataset
            points = dataset.GetPoints()
            if points is None:
                raise ValueError("No points found in VTK file")
            
            n_points = points.GetNumberOfPoints()
            if n_points == 0:
                raise ValueError("VTK file contains 0 points")
            
            print(f"Successfully loaded {n_points} points from {dataset_type} dataset")
            
            # Try using vtk_to_numpy (more efficient)
            try:
                points_data = points.GetData()
                points_array = vtk_to_numpy(points_data)
                
                # Ensure it's the right shape
                if points_array.shape[1] != 3:
                    raise ValueError(f"Expected 3D points, got shape {points_array.shape}")
                
                return points_array.astype(np.float32)
                
            except Exception as vtk_numpy_error:
                print(f"vtk_to_numpy failed: {vtk_numpy_error}")
                print("Falling back to manual extraction...")
                
                # Manual extraction (fallback)
                points_array = np.zeros((n_points, 3), dtype=np.float32)
                for i in range(n_points):
                    point = points.GetPoint(i)
                    if len(point) != 3:
                        raise ValueError(f"Point {i} has wrong dimension: {len(point)}")
                    points_array[i] = point
                
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
        
    def get_svd_components(self):
        """Return the SVD components for analysis"""
        return self.svd.components_
    
    def get_explained_variance_ratio(self):
        """Return the explained variance ratio"""
        return self.svd.explained_variance_ratio_
    
    def reconstruct_geometry(self, svd_repr):
        """Reconstruct the original geometry from SVD representation"""
        return self.svd.inverse_transform(svd_repr)
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        return (
            self.geometry_svd[idx],
            self.xyz_coords[idx],
            self.values[idx]
        )
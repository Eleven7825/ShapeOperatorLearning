#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.calculateA import rotate_matrix
from torch.utils.data import Dataset
from utils.project_vector_on_spheroid import project_tangent, compute_orthogonality_error

class ShearStressDataset(Dataset):
    """Dataset that loads and processes data on CPU"""
    def __init__(self, npz_files):
        super().__init__()
        A_list = []
        values_list = []
        xyz_list = []
        
        print(f"Loading {len(npz_files)} data files...")
        loaded_files = 0
        skipped_files = 0
        
        for i, file in enumerate(npz_files):
            try:
                # Print progress every 100 files
                if i % 100 == 0:
                    print(f"Processing file {i}/{len(npz_files)}...")
                
                if compute_orthogonality_error(file) > 0.05:
                    print(f"Skipping {file} due to large orthogonality error!")
                    skipped_files += 1
                    continue
                
                
                data = np.load(file)
                A = data['A_matrices']
                A_orig = A
                A, R_x, phi = rotate_matrix(A)
                ref_xyz = data['ref_xyz']
                M = np.linalg.cholesky(A_orig)
                xyz = ref_xyz @ np.linalg.inv(M)
                values = data['interpolated_values']
                if np.all(values==0):
                    print(f"Skipping {file} due to all its values are zero!")
                    skipped_files += 1
                    continue
                values[:, 0:3] = project_tangent(A_orig, xyz, values[:,0:3])
                values[:, 0:3] = values[:, 0:3] @ R_x.T
                values[:, 0:3] = values[:, 0:3] @ np.linalg.cholesky(A)
                
                A_upper = np.array([
                    A[0, 0], A[0, 1], A[0, 2],
                    A[1, 1], A[1, 2], A[2, 2]
                ])
                
                xyz_rotated = xyz @ R_x.T
                xyz_sphere = xyz_rotated @ np.linalg.cholesky(A)
                
                n_points = len(data['ref_xyz'])
                A_repeated = np.tile(A_upper, (n_points, 1))
                
                A_list.append(A_repeated)
                xyz_list.append(xyz_sphere)
                values_list.append(values[:, 0:3])
                loaded_files += 1
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                skipped_files += 1
                continue
        
        print(f"Dataset loading complete. Loaded {loaded_files} files, skipped {skipped_files} files.")
        
        self.A_matrices = np.concatenate(A_list, axis=0)
        self.xyz = np.concatenate(xyz_list, axis=0)
        self.values = np.concatenate(values_list, axis=0)
        
        # Convert to PyTorch tensors on CPU
        self.A_matrices = torch.FloatTensor(self.A_matrices)
        self.xyz = torch.FloatTensor(self.xyz)
        self.values = torch.FloatTensor(self.values)
        
        print(f"Dataset created with {len(self.A_matrices)} samples.")
        
    def __len__(self):
        return len(self.A_matrices)
    
    def __getitem__(self, idx):
        return (
            self.A_matrices[idx],
            self.xyz[idx],
            self.values[idx]
        )
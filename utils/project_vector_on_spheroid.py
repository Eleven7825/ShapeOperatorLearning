#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:14:06 2025

@author: shiyichen
"""

import numpy as np
from utils.calculateA import rotate_matrix
# Use non-interactive backend to avoid Qt platform plugin errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def project_tangent(A, x, v):
    """Vectorized tangent space projection (handles batches)"""
    x = np.atleast_2d(x)
    v = np.atleast_2d(v)
    n = x @ A
    v_dot_n = np.einsum('ij,ij->i', v, n)
    n_norm_sq = np.einsum('ij,ij->i', n, n)
    projection_scale = (v_dot_n / n_norm_sq)[:, np.newaxis]
    v_tangent = v - projection_scale * n
    return np.squeeze(v_tangent) if v.ndim == 1 else v_tangent

# def main():
#     # Load data
#     data = np.load("processed_data/processed_spheroid_data_1.npz")
#     A = data['A_matrices']
#     values = data['interpolated_values']
#     values = values[:,0:3]
#     ref_xyz = data['ref_xyz']
    
#     # Cholesky decomposition and coordinate transformation
#     M = np.linalg.cholesky(A)
#     #xyz = ref_xyz @ np.linalg.inv(M).T  # Corrected matrix inversion
#     xyz = ref_xyz @ np.linalg.inv(M)
    
#     # Project values onto tangent space
#     projected_values = project_tangent(A, xyz, values)
    
#     # Calculate normals and verify orthogonality
#     normals = xyz @ A
#     dot_products = np.einsum('ij,ij->i', projected_values, normals)
#     average_dot = np.mean(dot_products)
    
#     print(f"Verification results:")
#     print(f"Average projection-normal inner product: {average_dot:.3e}")
#     print("(Should be very close to zero for correct implementation)")

def compute_orthogonality_error(file_path):
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        return np.nan  # Return NaN if file not found
    
    # Extract required data
    try:
        A_orig = data['A_matrices']
        values_orig = data['interpolated_values'][:, 0:3]
        ref_xyz = data['ref_xyz']
    except KeyError as e:
        print(f"Missing key {e} in {file_path}")
        return np.nan  # Return NaN if keys are missing
    
    # Compute Cholesky decomposition
    try:
        M = np.linalg.cholesky(A_orig)
    except np.linalg.LinAlgError:
        print(f"Cholesky decomposition failed for {file_path}")
        return np.nan  # Return NaN if Cholesky decomposition fails
    
    try:
        inv_M = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print(f"Matrix inversion failed for {file_path}")
        return np.nan
    
    xyz = ref_xyz @ inv_M
    # Compute xyz @ A_orig
    xyz_A = xyz @ A_orig
    
    # Calculate orthoError
    orthoError = np.einsum('ij,ij->i', values_orig, xyz_A)
    
    # Compute norms of values_orig
    norms = np.sqrt(np.einsum('ij,ij->i', values_orig, values_orig))
    # Avoid division by zero by replacing zero norms with a small value
    norms[norms == 0] = 1e-10
    
    orthoganalityError = orthoError / norms
    return np.mean(orthoganalityError)

# def main():
#     # Load data
#     data = np.load("processed_data/processed_spheroid_data_214.npz")
#     A = data['A_matrices']
#     A_orig = A
#     A, R_x, phi = rotate_matrix(A)  # Assuming rotate_matrix is defined elsewhere
#     values = data['interpolated_values']
#     values= values[:, 0:3]
#     values_orig = values
#     # Rotate the values
#     values = values @ R_x.T
    
#     # Compute reference coordinates and transform them
#     ref_xyz = data['ref_xyz']
#     M = np.linalg.cholesky(A_orig)
#     xyz = ref_xyz @ np.linalg.inv(M)
#     xyz_rotated = xyz @ R_x.T
#     xyz_sphere = xyz_rotated @ np.linalg.cholesky(A)
#     # plt.scatter(xyz_rotated[:,0], xyz_rotated[:,1], xyz_rotated[:,2])
#     # plt.axis("equal")
#     # plt.show()
    
#     # Project values onto the tangent space
#     projected_values = project_tangent(A_orig, xyz, values)
    
#     # Transform the projected values
#     projected_values_rotated = projected_values @ R_x.T
#     projected_values_rotated_transformed = projected_values_rotated @ np.linalg.cholesky(A)
    
    
#     # Verify orthogonality between rotated projected values and xyz_sphere
#     norms = np.sqrt(np.einsum('ij,ij->i', values_orig, values_orig))
#     orthoError = np.einsum('ij,ij->i', values_orig, xyz @ A)
#     orthoganalityError = orthoError/norms
#     orthoganalityError_normalized = np.mean(orthoganalityError)
    
#     average_dot = np.mean(np.einsum('ij,ij->i', projected_values_rotated_transformed, xyz_sphere))
    
#     print("Verification results:")
#     print(f"Average original orthogonality error: {orthoganalityError_normalized:.3e}")
#     print(f"Average dot product (rotated projected values vs xyz_sphere): {average_dot:.3e}")
#     print("(Should be very close to zero for orthogonality)")

def main():
    # Collect all orthogonality errors
    errors = []
    high_error_cases = []  # To store case indices with errors > 0.05
    for i in range(1, 2301):
        file_path = f"processed_data/processed_spheroid_data_{i}.npz"
        if not os.path.exists(file_path):
            continue  # Skip non-existent files
        error = compute_orthogonality_error(file_path)
        if np.abs(error) > 0.05:
            high_error_cases.append(i)
        if not np.isnan(error):
            errors.append(error)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(errors), bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Normalized Orthogonality Errors')
    plt.xlabel('Normalized Orthogonality Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Printout high error cases:
    print("Case indices with normalized orthogonality error > 0.05:")
    print(high_error_cases)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate SVD coefficients from VTK geometry files.
Loads template geometry and computes dx as the difference between template and target geometries.
Applies SVD reduction to obtain coefficients for each case.
"""

import os
import re
import numpy as np
from tqdm import tqdm
import argparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from svd_utils import SVD_reduce


def load_vtk_points(vtk_file_path):
    """
    Load points from a VTK file.

    Args:
        vtk_file_path: Path to the VTK file

    Returns:
        numpy array of points/vertices with shape (n_points, 3)
    """
    vtk_file_path = os.path.expanduser(vtk_file_path)

    if not os.path.exists(vtk_file_path):
        raise FileNotFoundError(f"VTK file not found: {vtk_file_path}")

    try:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_file_path)
        reader.Update()

        polydata = reader.GetOutput()
        if polydata is None:
            raise ValueError("Failed to get polydata from VTK file")

        points = polydata.GetPoints()
        if points is None:
            raise ValueError("No points found in VTK file")

        points_array = vtk_to_numpy(points.GetData())

        if points_array.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got shape {points_array.shape}")

        return points_array

    except Exception as e:
        raise RuntimeError(f"Error loading VTK file {vtk_file_path}: {e}")


def calculate_dx_matrix(template_points, target_points):
    """
    Calculate displacement vector as difference between template and target points.

    Args:
        template_points: numpy array of template points (n_points, 3)
        target_points: numpy array of target points (n_points, 3)

    Returns:
        dx: displacement matrix (n_points, 3) representing the difference
    """
    if template_points.shape != target_points.shape:
        raise ValueError(f"Template and target shapes don't match: {template_points.shape} vs {target_points.shape}")

    dx = target_points - template_points
    return dx


def get_case_number(matching_dir):
    """Extract case number from matching directory name"""
    match = re.search(r'matching_(\d+)', matching_dir)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description='Calculate SVD coefficients from VTK geometry files')
    parser.add_argument('--template_vtk', type=str,
                        default='../Nov6/matchings/matching_1/1-shoot-1.vtk',
                        help='Path to template VTK file (reference geometry)')
    parser.add_argument('--matchings_dir', type=str,
                        default='../Nov6/matchings',
                        help='Directory containing matching subdirectories')
    parser.add_argument('--target_vtk_name', type=str,
                        default='1-shoot-16.vtk',
                        help='Name of the target VTK file in each matching directory')
    parser.add_argument('--output_file', type=str, default='coefficient_data.npz',
                        help='Output NPZ file to save coefficient array')
    parser.add_argument('--mode', type=int, default=3,
                        help='Number of SVD modes to keep (default: 3 to match original file)')
    parser.add_argument('--case_range', type=int, nargs=2, default=[1, 3000],
                        help='Range of case numbers to process [start end]')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')

    args = parser.parse_args()

    # Load template geometry once
    print(f"Loading template geometry from: {args.template_vtk}")
    try:
        template_points = load_vtk_points(args.template_vtk)
        print(f"Template points shape: {template_points.shape}")
    except Exception as e:
        print(f"Error loading template VTK: {e}")
        raise

    # Prepare to collect coefficient data
    coefficient_data = []
    case_numbers = []

    # Generate list of case numbers to process
    case_start, case_end = args.case_range
    all_cases = list(range(case_start, case_end + 1))

    if args.max_files:
        all_cases = all_cases[:args.max_files]

    print(f"Processing {len(all_cases)} cases (SVD mode={args.mode})...")

    # Collect dx data for all cases
    # dx shape: (num_cases, num_points, 3)
    dx_list = []
    valid_cases = []

    for case_num in tqdm(all_cases, desc="Loading target geometries and computing dx"):
        matching_dir = os.path.join(args.matchings_dir, f"matching_{case_num}")
        target_vtk = os.path.join(matching_dir, args.target_vtk_name)

        if not os.path.exists(target_vtk):
            continue

        try:
            target_points = load_vtk_points(target_vtk)

            # Calculate displacement: (num_points, 3)
            dx = calculate_dx_matrix(template_points, target_points)
            dx_list.append(dx)
            valid_cases.append(case_num)

        except Exception as e:
            print(f"Error processing case {case_num}: {e}")
            continue

    if len(valid_cases) == 0:
        print("ERROR: No valid cases found to process!")
        return

    print(f"Successfully loaded {len(valid_cases)} cases")

    # Assemble dx matrix: shape (num_cases, num_points, 3)
    dx = np.array(dx_list)
    print(f"Displacement matrix shape: dx={dx.shape}")

    # Extract components following main.py pattern:
    # dx[:, :, 0] gives (num_cases, num_points) for x-component
    # Transpose to (num_points, num_cases) for SVD
    dx1_train = dx[:, :, 0].transpose()  # x-component: (num_points, num_cases)
    dx2_train = dx[:, :, 1].transpose()  # y-component: (num_points, num_cases)
    dx3_train = dx[:, :, 2].transpose()  # z-component: (num_points, num_cases)

    print(f"Component matrices shape: dx1={dx1_train.shape}, dx2={dx2_train.shape}, dx3={dx3_train.shape}")

    # Apply SVD reduction (following main.py pattern)
    print(f"Applying SVD reduction with mode={args.mode}...")
    Ux, coeff_x, Uy, coeff_y, Uz, coeff_z = SVD_reduce(dx1_train, dx2_train, dx3_train, args.mode)

    print(f"SVD basis shapes: Ux={Ux.shape}, Uy={Uy.shape}, Uz={Uz.shape}")
    print(f"Coefficient shapes: coeff_x={coeff_x.shape}, coeff_y={coeff_y.shape}, coeff_z={coeff_z.shape}")

    # Combine coefficients into a single array (following main.py pattern)
    # f_train = np.concatenate((coeff_x, coeff_y, coeff_z), axis=0).transpose()
    combined_coeffs = np.concatenate((coeff_x, coeff_y, coeff_z), axis=0).transpose()

    print(f"Combined coefficient matrix shape: {combined_coeffs.shape}")
    print(f"Total features per case: {combined_coeffs.shape[1]}")

    # Save the array to an NPZ file (matching original file format)
    # Original format: coefficients (n_cases, 3*mode), case_numbers, l_max (as scalar)
    np.savez(
        args.output_file,
        coefficients=combined_coeffs,
        case_numbers=np.array(valid_cases),
        l_max=args.mode  # Store mode as l_max for compatibility with original format
    )

    print(f"\nSuccessfully saved coefficient data to {args.output_file}")
    print(f"Array shape: {combined_coeffs.shape}")
    print(f"Cases processed: {len(valid_cases)}")


if __name__ == "__main__":
    main()

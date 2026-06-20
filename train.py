import torch
import yaml
import numpy as np
import warnings

# Suppress the Axes3D import warning from Matplotlib
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

# Use non-interactive backend to avoid Qt platform plugin errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv
import importlib
import argparse
import sys
from torch.utils.data import DataLoader, Subset
import time

# Add utils directory to path for zero case filtering
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from zero_case_filter import filter_zero_stress_files, verify_zero_case_exclusion

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a ShearStress neural network model with spherical harmonic coefficients')
    parser.add_argument('--config', type=str, default='./inputs/config.yaml', 
                        help='Path to the configuration YAML file (default: ./inputs/config.yaml)')
    parser.add_argument('--coefficients', type=str, required=True,
                        help='Path to coefficient_data.npz produced by the data generation pipeline')
    parser.add_argument('--skip_zero_filter', action='store_true',
                        help='Skip the hardcoded zero-stress case filter (use for non-Nov6 datasets)')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_path = args.config
    coefficients_path = args.coefficients
    print(f"Loading configuration from: {config_path}")
    print(f"Using coefficients from: {coefficients_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Dynamically import modules based on configuration
    dataset_module = importlib.import_module(config['modules']['dataset_module'])
    model_module = importlib.import_module(config['modules']['model_module'])
    training_module = importlib.import_module(config['modules']['training_function_module'])
    
    # Get classes and functions from the imported modules
    ShearStressDataset = getattr(dataset_module, config['modules']['dataset_class'])
    ShearStressNN = getattr(model_module, config['modules']['model_class'])
    train_model = getattr(training_module, config['modules']['training_function'])
    
    # Create results directory if it doesn't exist
    results_dir = config['output']['result_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy config file to results directory
    config_filename = os.path.basename(config_path)
    with open(os.path.join(results_dir, config_filename), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    # Set random seeds
    seed = config['training']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Generate file paths
    npz_files = [
        config['data']['npz_files_pattern'].format(i) 
        for i in range(config['data']['npz_range'][0], config['data']['npz_range'][1] + 1)
    ]
    
    # Filter out zero shear stress cases (skip for datasets validated at prep time)
    if args.skip_zero_filter:
        print("Skipping zero-stress case filter (--skip_zero_filter set)")
    else:
        print(f"\n{'='*60}")
        print("FILTERING OUT ZERO SHEAR STRESS CASES")
        print(f"{'='*60}")
        npz_files = filter_zero_stress_files(npz_files)
        verify_zero_case_exclusion(npz_files)
        print(f"{'='*60}\n")
    
    # Create dataset with coefficients
    print(f"Creating dataset with spherical harmonic coefficients...")
    start_time = time.time()
    ref_xyz_vtk_path = config['data']['ref_xyz_vtk_path']
    value_key = config['data'].get('value_key', 'transformed_values')
    out_dim   = config['model'].get('out_dim', 3)
    full_dataset = ShearStressDataset(
        npz_files, coefficients_path,
        ref_xyz_vtk_path=ref_xyz_vtk_path,
        value_key=value_key,
        out_dim=out_dim,
    )
    print(f"Dataset creation time: {(time.time() - start_time) / 60:.2f} minutes")

    # Get the actual number of loaded shapes from the dataset
    # The dataset stores how many files were actually loaded successfully
    if hasattr(full_dataset, 'loaded_files'):
        actual_shapes_loaded = full_dataset.loaded_files
    else:
        # Fallback: assume all files in npz_files were loaded
        # This will be incorrect if some files failed to load
        print("WARNING: Dataset doesn't report loaded_files count. Split may be inaccurate if files failed to load.")
        actual_shapes_loaded = len(npz_files)

    # Create train-val split based on actual loaded shapes
    points_per_shape = len(full_dataset) // actual_shapes_loaded
    n_train_shapes = int(config['data']['train_split'] * actual_shapes_loaded)
    train_size = n_train_shapes * points_per_shape
    val_size = len(full_dataset) - train_size

    print(f"Files in list: {len(npz_files)}")
    print(f"Actually loaded shapes: {actual_shapes_loaded}")
    print(f"Points per shape: {points_per_shape}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train shapes: {n_train_shapes}, samples: {train_size}")
    print(f"Val shapes: {actual_shapes_loaded - n_train_shapes}, samples: {val_size}")
    
    # Split indices
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Determine batch size
    batch_size = points_per_shape if config['training']['batch_size'] == 'auto' else config['training']['batch_size']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers']
    )
    
    # Determine input dimensions based on coefficient size
    coefficient_size = full_dataset.coefficients.shape[1] if hasattr(full_dataset, 'coefficients') else 25

    # Update branch_dims to use coefficient size as input dimension
    branch_dims = config['model']['branch_dims'].copy()
    branch_dims[0] = coefficient_size

    # Initialize model with appropriate parameters based on model type
    print(f"Initializing model with branch_dims={branch_dims} (using coefficient size {coefficient_size})")

    # Build model initialization kwargs
    model_kwargs = {
        'branch_dims': branch_dims,
        'trunk_dims': config['model']['trunk_dims']
    }

    # Add optional parameters if they exist in config
    if 'final_dim' in config['model']:
        model_kwargs['final_dim'] = config['model']['final_dim']
    if 'out_dim' in config['model']:
        model_kwargs['out_dim'] = config['model']['out_dim']
    if 'combined_dims' in config['model']:
        model_kwargs['combined_dims'] = config['model']['combined_dims']
    if 'dropout' in config['model']:
        model_kwargs['dropout'] = config['model']['dropout']

    model = ShearStressNN(**model_kwargs)
    
    # Train model
    train_losses, val_losses, epoch_val, epoch_train = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=config['training']['n_epochs'],
        lr=float(config['training']['learning_rate']),
        results_dir=results_dir
    )
    
    # Plot training curves if specified
    if config['output']['plot_curves']:
        plt.figure(figsize=(12, 6))
        plt.plot(epoch_train, train_losses, label='Train Loss')
        plt.plot(epoch_val, val_losses, label='Val Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves (Using Spherical Harmonic Coefficients)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'training_curves.png'))
        plt.show()

if __name__ == "__main__":
    main()

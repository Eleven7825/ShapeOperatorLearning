import torch
import numpy as np
# Use non-interactive backend to avoid Qt platform plugin errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import yaml
import argparse
from scipy.interpolate import griddata
import importlib
import sys
import re
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def project_vector_to_sphere(v, n):
    """Project vector v onto sphere surface using normal n"""
    return v - np.dot(v, n) * n

def calculate_local_basis(XX, normals):
    """Calculate local theta and phi basis vectors at each point"""
    n = len(XX)
    e_theta = np.zeros((n, 3))
    e_phi = np.zeros((n, 3))
    
    x_axis = np.array([-1, 0, 0])
    
    for i in range(n):
        n_vec = normals[i]
        
        # e_phi is perpendicular to both x_axis and the normal vector
        e_phi[i] = np.cross(x_axis, n_vec)
        e_phi[i] = e_phi[i] / np.linalg.norm(e_phi[i])
        
        # e_theta is perpendicular to both the normal vector and e_phi
        e_theta[i] = np.cross(n_vec, e_phi[i])
        e_theta[i] = e_theta[i] / np.linalg.norm(e_theta[i])
    
    return e_theta, e_phi

def decompose_vector(v_projected, e_theta, e_phi):
    """Decompose projected vector into theta and phi components"""
    v_theta = np.dot(v_projected, e_theta)
    v_phi = np.dot(v_projected, e_phi)
    return v_theta, v_phi

def create_streamline_plot(theta, phi, v_theta, v_phi, title=""):
    """Create 2D streamline plot in theta-phi space"""
    # Create a regular grid for interpolation
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(np.min(theta), np.max(theta), 100),
        np.linspace(np.min(phi), np.max(phi), 100)
    )
    
    # Interpolate vector components onto regular grid
    v_theta_grid = griddata((theta, phi), v_theta, (theta_grid, phi_grid), method='cubic')
    v_phi_grid = griddata((theta, phi), v_phi, (theta_grid, phi_grid), method='cubic')
    
    # Calculate speed for color but don't create a colorbar
    speed = np.sqrt(v_theta_grid**2 + v_phi_grid**2)
    stream = plt.streamplot(theta_grid, phi_grid, v_theta_grid, v_phi_grid, 
                          density=1.5,
                          color='blue')  # Set fixed color instead of using speed
    plt.xlabel('Theta (from -x axis)')
    plt.ylabel('Phi (y-z plane)')
    plt.title(title)

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to Cartesian coordinates"""
    r = 1  # unit sphere
    x = -r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(theta) * np.sin(phi)
    return x, y, z

def process_vectors_simple(values, XX):
    """Process vectors: project and decompose (simplified for coefficient-based approach)"""
    # Calculate normals on sphere (normalized position vectors)
    normals = XX / np.linalg.norm(XX, axis=1)[:, np.newaxis]
    
    # Project vectors onto sphere surface
    projected_values = np.array([project_vector_to_sphere(v, n) 
                               for v, n in zip(values, normals)])
    
    # Calculate local basis vectors
    e_theta, e_phi = calculate_local_basis(XX, normals)
    
    # Decompose vectors into theta and phi components
    v_theta = np.array([decompose_vector(v, et, ep)[0] 
                       for v, et, ep in zip(projected_values, e_theta, e_phi)])
    v_phi = np.array([decompose_vector(v, et, ep)[1] 
                     for v, et, ep in zip(projected_values, e_theta, e_phi)])
    
    # Calculate theta and phi coordinates
    theta = np.arccos(-XX[:, 0] / np.sqrt(np.sum(XX**2, axis=1)))
    phi = np.arctan2(XX[:, 2], XX[:, 1])
    
    return theta, phi, v_theta, v_phi

def load_vtk_points(vtk_file_path):
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
        
        # Read the file
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
        
        # Try using vtk_to_numpy (more efficient)
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
            
            # Manual extraction (fallback)
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
        raise

def extract_case_number_from_file(file_path):
    """Extract case number from NPZ file path"""
    case_pattern = re.compile(r'(processed_spheroid_data_|transformed_data_|processed_geometry_data_)(\d+)\.npz')
    case_match = case_pattern.search(file_path)
    if case_match:
        return int(case_match.group(2))
    else:
        raise ValueError(f"Could not extract case number from {file_path}")

def load_and_predict(config_path='inputs/config.yaml', data_index=2300, result_dir=None, coefficients_file='coefficient_data.npz'):
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get result directory from config if not specified
    if result_dir is None:
        result_dir = config['output']['result_dir']
    
    # Add the current directory to the Python path to ensure local modules can be found
    sys.path.insert(0, os.getcwd())
    
    # Dynamically import model module
    try:
        print(f"Importing model module from {config['modules']['model_module']}")
        model_module = importlib.import_module(config['modules']['model_module'])
        ShearStressNN = getattr(model_module, config['modules']['model_class'])
    except ImportError as e:
        print(f"Error importing model module: {e}")
        print("Falling back to shearStressNN_coeff import...")
        from shearStressNN_coeff import ShearStressNN
    
    # Load coefficients data
    print(f"Loading coefficients from {coefficients_file}...")
    coeff_data = np.load(coefficients_file)
    coefficients = coeff_data['coefficients']
    case_numbers = coeff_data['case_numbers']
    
    # Create a mapping from case number to coefficients
    coeff_mapping = {int(case): coeffs for case, coeffs in zip(case_numbers, coefficients)}
    
    # Check if we have coefficients for the requested data_index
    if data_index not in coeff_mapping:
        print(f"No coefficients found for case {data_index}")
        print(f"Available cases: {sorted(coeff_mapping.keys())}")
        return
    
    # Get coefficients for this case
    case_coeffs = coeff_mapping[data_index]
    coefficient_size = len(case_coeffs)
    
    # Update branch_dims to use coefficient size as input dimension
    branch_dims = config['model']['branch_dims'].copy()
    branch_dims[0] = coefficient_size
    
    # Create model with architecture from config
    model = ShearStressNN(
        branch_dims=branch_dims,
        trunk_dims=config['model']['trunk_dims'],
        final_dim=config['model']['final_dim']
    )
    
    print(f"Using model architecture: branch_dims={branch_dims}, trunk_dims={config['model']['trunk_dims']}, final_dim={config['model']['final_dim']}")
    
    # Check if model checkpoint exists
    model_path = os.path.join(result_dir, 'shear_stress_model.pt')
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        # Try alternative paths
        alternative_paths = [
            'shear_stress_model.pt',
            os.path.join('slurms', result_dir, 'shear_stress_model.pt')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            print(f"Model checkpoint not found in any of the expected locations.")
            return
    
    print(f"Loading model from {model_path}")
    
    # Load checkpoint and handle DataParallel state dict
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix from state dict keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Determine data file path from config pattern or use default
    if 'npz_files_pattern' in config['data'] and 'npz_range' in config['data']:
        data_file = config['data']['npz_files_pattern'].format(data_index)
        if data_file is None:
            print(f"Could not find data file for index {data_index}")
            return
    
    print(f"Loading data from {data_file}")
    
    try:
        data = np.load(data_file)
    except FileNotFoundError:
        print(f"Data file not found at {data_file}")
        return
    
    # Load reference xyz coordinates from VTK file
    ref_xyz_vtk_path = config['data'].get('ref_xyz_vtk_path', 
                                         os.path.expanduser("../Nov6/matchings/matching_1/1-shoot-1.vtk"))
    
    try:
        ref_xyz = load_vtk_points(ref_xyz_vtk_path)
        print(f"Loaded reference xyz coordinates with shape: {ref_xyz.shape}")
    except Exception as e:
        print(f"Error loading VTK file: {e}")
        return
    
    # Get true values from data file
    try:
        if 'transformed_values' in data:
            true_values = data['transformed_values']
            print("Using transformed_values from data file")
        elif 'interpolated_values' in data:
            true_values = data['interpolated_values']
            print("Using interpolated_values from data file")
        else:
            print("No suitable values found in data file")
            print(f"Available keys: {list(data.keys())}")
            return
    except Exception as e:
        print(f"Error loading values from data file: {e}")
        return
    
    # Only use the first 3 components of values (if there are more)
    true_values = true_values[:, 0:3]
    
    # Make sure we have the right number of points
    n_points = len(ref_xyz)
    if len(true_values) != n_points:
        print(f"Warning: Values length ({len(true_values)}) doesn't match ref_xyz length ({n_points})")
        min_length = min(len(true_values), n_points)
        true_values = true_values[:min_length]
        ref_xyz = ref_xyz[:min_length]
        n_points = min_length
    
    # Prepare coefficients (repeat for each point)
    repeated_coeffs = np.tile(case_coeffs, (n_points, 1))
    
    # Convert to tensors
    coeffs_tensor = torch.FloatTensor(repeated_coeffs)
    xyz_tensor = torch.FloatTensor(ref_xyz)
    true_values_tensor = torch.FloatTensor(true_values)
    
    print(f"Coefficient tensor shape: {coeffs_tensor.shape}")
    print(f"XYZ tensor shape: {xyz_tensor.shape}")
    print(f"True values tensor shape: {true_values_tensor.shape}")
    
    # Make predictions
    with torch.no_grad():
        predictions = []
        batch_size = 1000
        for i in range(0, len(ref_xyz), batch_size):
            batch_coeffs = coeffs_tensor[i:i+batch_size]
            batch_xyz = xyz_tensor[i:i+batch_size]
            batch_pred = model(batch_coeffs, batch_xyz)
            predictions.append(batch_pred)
        
        # Calculate loss
        loss = model.loss(coeffs_tensor, xyz_tensor, true_values_tensor)
        print(f'Loss: {loss:.6f}')
        
        predicted_values = torch.cat(predictions, dim=0).numpy()
    
    # Calculate errors
    diff = predicted_values - true_values
    true_norm = np.linalg.norm(true_values, axis=1)
    error = np.mean(np.linalg.norm(diff, axis=1))
    error_rel = np.mean(np.linalg.norm(diff, axis=1) / (true_norm + 1e-8))  # Add small epsilon to avoid division by zero
    print(f'Absolute error: {error:.6f}')
    print(f'Relative error: {error_rel:.6f}')
    
    # Convert points to unit sphere coordinates for visualization
    # Normalize the reference xyz to unit sphere
    ref_xyz_norm = ref_xyz / np.linalg.norm(ref_xyz, axis=1)[:, np.newaxis]
    
    # Process vectors for streamline plots
    theta_true, phi_true, v_theta_true, v_phi_true = process_vectors_simple(true_values, ref_xyz_norm)
    theta_pred, phi_pred, v_theta_pred, v_phi_pred = process_vectors_simple(predicted_values, ref_xyz_norm)
    
    # Create output directory for plots
    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create 3D color plot
    components = ['x', 'y', 'z']
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.1])
    
    # Store all scatter objects to get global min/max
    scatters = []
    
    # Create subplots for each component
    for i in range(3):
        # Predicted values (upper row)
        ax1 = fig.add_subplot(gs[0, i], projection='3d')
        scatter1 = ax1.scatter(ref_xyz_norm[:, 0], ref_xyz_norm[:, 1], ref_xyz_norm[:, 2], 
                             c=predicted_values[:, i], cmap='viridis')
        ax1.set_title(f'Predicted Shear Stress Component {components[i]}')
        scatters.append(scatter1)
        
        # True values (lower row)
        ax2 = fig.add_subplot(gs[1, i], projection='3d')
        scatter2 = ax2.scatter(ref_xyz_norm[:, 0], ref_xyz_norm[:, 1], ref_xyz_norm[:, 2], 
                             c=true_values[:, i], cmap='viridis')
        ax2.set_title(f'True Shear Stress Component {components[i]}')
        scatters.append(scatter2)
        
        # Set same color scale for predicted and true plots of the same component
        vmin = min(predicted_values[:, i].min(), true_values[:, i].min())
        vmax = max(predicted_values[:, i].max(), true_values[:, i].max())
        scatter1.set_clim(vmin, vmax)
        scatter2.set_clim(vmin, vmax)
        
        # Set axis labels
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    # Add single colorbar on the right
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(scatters[0], cax=cax, label='Shear Stress Value')
    
    plt.suptitle(f'Shear Stress Predictions vs Truth (Case {data_index})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'shear_stress_components_case_{data_index}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create streamline visualization
    plt.figure(figsize=(15, 5))
    
    # True values streamlines
    plt.subplot(121)
    create_streamline_plot(theta_true, phi_true, v_theta_true, v_phi_true,
                         title=f'True Values Streamlines (Case {data_index})')
    
    # Predicted values streamlines
    plt.subplot(122)
    create_streamline_plot(theta_pred, phi_pred, v_theta_pred, v_phi_pred,
                         title=f'Predicted Values Streamlines (Case {data_index})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'streamlines_case_{data_index}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nVisualization complete for case {data_index}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Model used coefficients of size: {coefficient_size}")
    print(f"Number of data points: {n_points}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize shear stress predictions using coefficient-based model')
    parser.add_argument('--config', type=str, default='./inputs/config.yaml', 
                        help='Path to the configuration YAML file (default: ./inputs/config.yaml)')
    parser.add_argument('--data-index', type=int, default=2300,
                        help='Index of the data file to visualize (default: 2300)')
    parser.add_argument('--result-dir', type=str, default=None,
                        help='Directory containing model checkpoint and for saving plots (default: from config)')
    parser.add_argument('--coefficients', type=str, default='coefficient_data.npz',
                        help='Path to the NPZ file containing spherical harmonic coefficients')
    
    args = parser.parse_args()
    
    load_and_predict(
        config_path=args.config,
        data_index=args.data_index,
        result_dir=args.result_dir,
        coefficients_file=args.coefficients
    )

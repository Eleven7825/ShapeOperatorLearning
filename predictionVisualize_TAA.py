#!/usr/bin/env python3
"""
Prediction visualization for TAA cylinder cases in cylindrical coordinates.

Cylindrical frame at each surface point:
  n̂    = (x/r, y/r, 0)          outward radial normal
  ê_θ  = (-sin θ, cos θ, 0)     azimuthal tangent
  ê_z  = (0, 0, 1)              axial tangent

Plots produced (saved to result_dir/plots/):
  1. Unwrapped cylinder (θ vs z) — WSS magnitude + 3 components, pred vs truth
  2. Streamlines on the unrolled (θ, z) surface
  3. 3D cylinder surface coloured by WSS magnitude
"""

import os
import sys
import re
import argparse
import importlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import yaml
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def cart_to_cyl(xyz):
    """Return (r, theta, z) arrays from (N,3) Cartesian array."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)          # [-π, π]
    return r, theta, z


def cyl_basis(theta):
    """
    Local cylindrical tangent vectors at each point.

    Returns
    -------
    e_th : (N,3)  azimuthal unit vector
    e_z  : (N,3)  axial unit vector (constant)
    """
    n = len(theta)
    e_th = np.stack([-np.sin(theta), np.cos(theta), np.zeros(n)], axis=1)
    e_z  = np.tile([0., 0., 1.], (n, 1))
    return e_th, e_z


def project_to_surface(wss, xyz):
    """
    Project WSS vectors onto the cylinder surface (remove radial component).

    Returns projected WSS and its (W_theta, W_z) tangential components.
    """
    r, theta, _ = cart_to_cyl(xyz)
    # Outward radial normal
    normal = np.stack([xyz[:, 0] / r, xyz[:, 1] / r, np.zeros(len(r))], axis=1)
    # Remove normal component
    wss_tang = wss - (np.sum(wss * normal, axis=1, keepdims=True)) * normal

    e_th, e_z = cyl_basis(theta)
    W_th = np.sum(wss_tang * e_th, axis=1)
    W_z  = np.sum(wss_tang * e_z,  axis=1)
    return wss_tang, W_th, W_z, theta


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_vtk_points(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VTK not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    pts = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def unwrapped_plot(ax, theta, z, values, title, cmap='coolwarm', vmin=None, vmax=None):
    """Scatter plot on the unrolled (θ, z) surface."""
    sc = ax.scatter(np.degrees(theta), z, c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=8)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('z (cm)')
    ax.set_title(title, fontsize=9)
    return sc


def streamline_plot(ax, theta, z, W_th, W_z, title):
    """2D streamlines on the unrolled cylinder (θ-z) surface."""
    th_grid, z_grid = np.meshgrid(
        np.linspace(theta.min(), theta.max(), 80),
        np.linspace(z.min(),     z.max(),     80),
    )
    Wth_g = griddata((theta, z), W_th, (th_grid, z_grid), method='cubic')
    Wz_g  = griddata((theta, z), W_z,  (th_grid, z_grid), method='cubic')
    speed = np.sqrt(Wth_g**2 + Wz_g**2)
    ax.streamplot(np.degrees(th_grid), z_grid,
                  Wth_g, Wz_g,
                  color=speed, cmap='plasma', density=1.2, linewidth=0.8)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('z (cm)')
    ax.set_title(title, fontsize=9)


def cylinder_3d_plot(ax, xyz, values, title, cmap='coolwarm', vmin=None, vmax=None):
    """3D scatter on cylinder surface coloured by scalar values."""
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=6)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title, fontsize=9)
    return sc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_and_predict(config_path, data_index, result_dir, coefficients_file):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if result_dir is None:
        result_dir = config['output']['result_dir']

    sys.path.insert(0, os.getcwd())
    model_module  = importlib.import_module(config['modules']['model_module'])
    ShearStressNN = getattr(model_module, config['modules']['model_class'])

    # Load coefficients
    coeff_data   = np.load(coefficients_file)
    coeff_map    = {int(c): v for c, v in zip(coeff_data['case_numbers'],
                                               coeff_data['coefficients'])}
    if data_index not in coeff_map:
        raise ValueError(f"Case {data_index} not in coefficient file. "
                         f"Available: {sorted(coeff_map.keys())[:10]} ...")
    case_coeffs      = coeff_map[data_index]
    coefficient_size = len(case_coeffs)

    # Load model
    branch_dims        = config['model']['branch_dims'].copy()
    branch_dims[0]     = coefficient_size
    model = ShearStressNN(branch_dims=branch_dims,
                          trunk_dims=config['model']['trunk_dims'],
                          final_dim=config['model']['final_dim'])

    model_path = os.path.join(result_dir, 'shear_stress_model.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state)
    model.eval()

    # Load data
    data_file = config['data']['npz_files_pattern'].format(data_index)
    data      = np.load(data_file)
    key       = 'transformed_values' if 'transformed_values' in data else 'interpolated_values'
    true_wss  = data[key][:, :3].astype(np.float32)

    # Reference geometry (LDDMM mesh points)
    ref_xyz = load_vtk_points(config['data']['ref_xyz_vtk_path'])
    n_pts   = len(ref_xyz)
    if len(true_wss) != n_pts:
        m = min(len(true_wss), n_pts)
        true_wss = true_wss[:m]; ref_xyz = ref_xyz[:m]; n_pts = m

    # Predict
    coeffs_t = torch.FloatTensor(np.tile(case_coeffs, (n_pts, 1)))
    xyz_t    = torch.FloatTensor(ref_xyz)
    preds = []
    with torch.no_grad():
        for i in range(0, n_pts, 1000):
            preds.append(model(coeffs_t[i:i+1000], xyz_t[i:i+1000]))
        pred_wss = torch.cat(preds).numpy()
        loss     = model.loss(coeffs_t, xyz_t, torch.FloatTensor(true_wss)).item()

    # Metrics
    diff     = pred_wss - true_wss
    mag_true = np.linalg.norm(true_wss, axis=1)
    abs_err  = np.mean(np.linalg.norm(diff, axis=1))
    rel_err  = np.mean(np.linalg.norm(diff, axis=1) / (mag_true + 1e-8))
    print(f"Loss: {loss:.6f}  |  Abs error: {abs_err:.6f}  |  Rel error: {rel_err:.4%}")

    # Cylindrical decomposition
    _, W_th_true, W_z_true, theta = project_to_surface(true_wss,  ref_xyz)
    _, W_th_pred, W_z_pred, _     = project_to_surface(pred_wss,  ref_xyz)
    _, z = cart_to_cyl(ref_xyz)[1], cart_to_cyl(ref_xyz)[2]
    r, theta, z = cart_to_cyl(ref_xyz)
    mag_true_s = np.linalg.norm(true_wss, axis=1)
    mag_pred_s = np.linalg.norm(pred_wss, axis=1)

    plots_dir = os.path.join(result_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1: Unwrapped cylinder — 3 WSS components + magnitude
    # ------------------------------------------------------------------
    components = ['x', 'y', 'z']
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f'WSS on Unrolled Cylinder Surface — Case {data_index}\n'
                 f'Abs err: {abs_err:.4e}  Rel err: {rel_err:.2%}', fontsize=12)

    for col, comp in enumerate(components):
        vmin = min(true_wss[:, col].min(), pred_wss[:, col].min())
        vmax = max(true_wss[:, col].max(), pred_wss[:, col].max())
        sc = unwrapped_plot(axes[0, col], theta, z, pred_wss[:, col],
                            f'Pred W_{comp}', vmin=vmin, vmax=vmax)
        unwrapped_plot(axes[1, col], theta, z, true_wss[:, col],
                       f'True W_{comp}', vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=[axes[0, col], axes[1, col]], shrink=0.6,
                     label=f'W_{comp} (Pa)')

    vmag = max(mag_true_s.max(), mag_pred_s.max())
    sc_m = unwrapped_plot(axes[0, 3], theta, z, mag_pred_s,
                          'Pred |WSS|', cmap='hot', vmin=0, vmax=vmag)
    unwrapped_plot(axes[1, 3], theta, z, mag_true_s,
                   'True |WSS|', cmap='hot', vmin=0, vmax=vmag)
    plt.colorbar(sc_m, ax=[axes[0, 3], axes[1, 3]], shrink=0.6, label='|WSS| (Pa)')

    axes[0, 0].set_ylabel('Predicted\nz (cm)', fontsize=10)
    axes[1, 0].set_ylabel('True\nz (cm)', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'unrolled_wss_case_{data_index}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Streamlines on unrolled cylinder (azimuthal vs axial WSS)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'WSS Streamlines on Unrolled Cylinder — Case {data_index}', fontsize=12)
    streamline_plot(axes[0], theta, z, W_th_true, W_z_true, 'True WSS streamlines')
    streamline_plot(axes[1], theta, z, W_th_pred, W_z_pred, 'Predicted WSS streamlines')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'streamlines_case_{data_index}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: 3D cylinder surface coloured by WSS magnitude
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f'3D Cylinder Surface — |WSS| — Case {data_index}', fontsize=12)
    vmag = max(mag_true_s.max(), mag_pred_s.max())

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = cylinder_3d_plot(ax1, ref_xyz, mag_pred_s, 'Predicted |WSS|',
                            vmin=0, vmax=vmag)
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = cylinder_3d_plot(ax2, ref_xyz, mag_true_s, 'True |WSS|',
                            vmin=0, vmax=vmag)
    plt.colorbar(sc1, ax=[ax1, ax2], shrink=0.5, label='|WSS| (Pa)')
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'3d_wss_case_{data_index}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to: {plots_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize TAA shear stress predictions in cylindrical coordinates')
    parser.add_argument('--config', type=str,
                        default='./inputs/config_TAA.yaml')
    parser.add_argument('--data-index', type=int, default=350,
                        help='Sample index to visualize (must be in validation set)')
    parser.add_argument('--result-dir', type=str, default=None)
    parser.add_argument('--coefficients', type=str,
                        default='../TAA_CFD_pipeline/coefficient_data.npz')
    args = parser.parse_args()

    load_and_predict(
        config_path=args.config,
        data_index=args.data_index,
        result_dir=args.result_dir,
        coefficients_file=args.coefficients,
    )

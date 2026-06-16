"""
SVD utility functions for dimensionality reduction
Author: Minglang Yin, minglang_yin@brown.edu
Modified to support 3D data
"""
import numpy as np
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def SVD_reduce(Ax, Ay, Az, mode):
    """
    Singular-value decomposition reduction for 3D data.

    Args:
        Ax (np.ndarray): Data matrix for x-component
        Ay (np.ndarray): Data matrix for y-component
        Az (np.ndarray): Data matrix for z-component
        mode (int): Number of modes to keep

    Returns:
        tuple: (Ux, coeff_x, Uy, coeff_y, Uz, coeff_z)
    """
    # Singular-value decomposition
    Ux, sx, VxT = svd(Ax)
    Uy, sy, VyT = svd(Ay)
    Uz, sz, VzT = svd(Az)

    # Truncate sigma - handle both tall (m > n) and wide (m < n) matrices
    # scipy.svd returns sigma with length = min(m, n)
    m_x, n_x = Ax.shape
    Sigmax = np.zeros((m_x, n_x))
    Sigmax[:len(sx), :len(sx)] = np.diag(sx)

    m_y, n_y = Ay.shape
    Sigmay = np.zeros((m_y, n_y))
    Sigmay[:len(sy), :len(sy)] = np.diag(sy)

    m_z, n_z = Az.shape
    Sigmaz = np.zeros((m_z, n_z))
    Sigmaz[:len(sz), :len(sz)] = np.diag(sz)

    # Truncate to mode
    Ux = Ux[:, :mode]
    Sigmax = Sigmax[:mode, :]

    Uy = Uy[:, :mode]
    Sigmay = Sigmay[:mode, :]

    Uz = Uz[:, :mode]
    Sigmaz = Sigmaz[:mode, :]

    # Plot energy modes
    fig = plt.figure(constrained_layout=False, figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)

    # Energy in x
    ax = fig.add_subplot(gs[0])
    EnergySqr = sx**2
    num_modes_x = len(sx)
    x_modes = np.linspace(1, num_modes_x, num_modes_x)
    ax.scatter(x_modes, EnergySqr/sum(EnergySqr), color='red', label='Mode Energy %')
    ax.plot(x_modes, EnergySqr/sum(EnergySqr), color='red')
    ax.axvline(x=mode, color='blue', linestyle='--', label=f'cut off mode = {mode}')
    ax.set_title(r"Energy fraction ($\sigma^{2}_{i}/\Sigma_{j}\sigma^{2}_{j}$) of $\Delta x$")
    ax.set_xlabel("POD Mode")
    ax.set_ylabel("Energy Fraction")
    ax.legend()
    ax.set_yscale("log")

    # Energy in y
    ax = fig.add_subplot(gs[1])
    EnergySqr = sy**2
    num_modes_y = len(sy)
    y_modes = np.linspace(1, num_modes_y, num_modes_y)
    ax.scatter(y_modes, EnergySqr/sum(EnergySqr), color='green', label='Mode Energy %')
    ax.plot(y_modes, EnergySqr/sum(EnergySqr), color='green')
    ax.axvline(x=mode, color='blue', linestyle='--', label=f'cut off mode = {mode}')
    ax.set_title(r"Energy fraction ($\sigma^{2}_{i}/\Sigma_{j}\sigma^{2}_{j}$) of $\Delta y$")
    ax.set_xlabel("POD Mode")
    ax.set_ylabel("Energy Fraction")
    ax.legend()
    ax.set_yscale("log")

    # Energy in z
    ax = fig.add_subplot(gs[2])
    EnergySqr = sz**2
    num_modes_z = len(sz)
    z_modes = np.linspace(1, num_modes_z, num_modes_z)
    ax.scatter(z_modes, EnergySqr/sum(EnergySqr), color='purple', label='Mode Energy %')
    ax.plot(z_modes, EnergySqr/sum(EnergySqr), color='purple')
    ax.axvline(x=mode, color='blue', linestyle='--', label=f'cut off mode = {mode}')
    ax.set_title(r"Energy fraction ($\sigma^{2}_{i}/\Sigma_{j}\sigma^{2}_{j}$) of $\Delta z$")
    ax.set_xlabel("POD Mode")
    ax.set_ylabel("Energy Fraction")
    ax.legend()
    ax.set_yscale("log")

    fig.savefig('POD_mode_frac.png')
    plt.close()

    # Calculate coefficients
    coeff_x = Sigmax.dot(VxT)
    coeff_y = Sigmay.dot(VyT)
    coeff_z = Sigmaz.dot(VzT)

    return Ux, coeff_x, Uy, coeff_y, Uz, coeff_z

import numpy as np

def rotate_matrix(A):
    # 1. Get eigenvalues and eigenvectors, sorted in descending order
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Get the eigenvector corresponding to largest eigenvalue
    v = eigenvectors[:, 0]  # v = (vx, vy, vz)
    
    # 1. Calculate phi using atan2(-vz, vy)
    phi = np.arctan(-v[2] / v[1])
    
    # Create rotation matrix R_x(phi)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])
    
    # 2. Calculate A_prime = R_x @ A @ R_x.T
    A_prime = R_x @ A @ R_x.T
    
    return A_prime, R_x, phi

if __name__ == "__main__":
    # Create matrix A
    A = np.array([[3, 1, 2],
                  [1, 4, -1],
                  [2, -1, 5]])
    
    # Create matrix B by applying a rotation around x-axis to A
    theta = np.pi/3  # Some arbitrary angle
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    B = R_x @ A @ R_x.T
    
    # Apply rotate_matrix to both matrices
    A_prime_A, R_x_A, phi_A = rotate_matrix(A)
    A_prime_B, R_x_B, phi_B = rotate_matrix(B)
    
    print("Original matrix A:")
    print(A)
    print("\nOriginal matrix B:")
    print(B)
    print("\nRotated A_prime from A:")
    print(A_prime_A)
    print("\nRotated A_prime from B:")
    print(A_prime_B)
    print("\nDifference between rotated matrices:")
    print(np.abs(A_prime_A - A_prime_B))
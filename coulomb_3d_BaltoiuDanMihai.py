import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
import time
from mpl_toolkits.mplot3d import Axes3D 

# Set parameters for the grid
N = 30            # Number of grid points in each dimension (Lower if your computer has lower specs)
x_min, x_max = -10.0, 10.0  # Spatial domain in x
y_min, y_max = -10.0, 10.0  # Spatial domain in y
z_min, z_max = -10.0, 10.0  # Spatial domain in z
h = (x_max - x_min) / (N - 1)  # Grid spacing

# Generate grid points
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
z = np.linspace(z_min, z_max, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-10  # Avoid division by zero

# Define Coulomb potential V(r) = -1/r (in atomic units)
V = -1.0 / R

# Flatten the potential to create a 1D array
V_flat = V.flatten()

# Construct the kinetic energy matrix using finite differences (Laplacian)
# 3D Laplacian using Kronecker products
e = np.ones(N)
# 1D Laplacian
T_1D = diags([e, -2*e, e], [-1, 0, 1], shape=(N, N))
T_1D = T_1D / h**2

# 3D Laplacian using Kronecker products
I = identity(N)
Laplacian = kron(kron(T_1D, I), I) + kron(kron(I, T_1D), I) + kron(kron(I, I), T_1D)

# Hamiltonian matrix H = - (1/2) * Laplacian + V
# In atomic units, ħ = 1 and mass m = 1/2, so kinetic term is - (1/2) * Laplacian
H = (-0.5) * Laplacian + diags(V_flat, 0, format='csc')

def power_iteration(H, num_iterations=1000, tol=1e-10):
    """
    Power Iteration method to find the dominant eigenvalue and eigenvector.
    Suitable for finding the highest energy state.
    """
    b_k = np.random.rand(H.shape[1])
    b_k /= np.linalg.norm(b_k)
    
    for _ in range(num_iterations):
        # Matrix-by-vector product
        b_k1 = H.dot(b_k)
        
        # Compute the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm
        
        # Check for convergence
        if np.linalg.norm(H.dot(b_k) - b_k1_norm * b_k) < tol:
            break
    
    eigenvalue = b_k1_norm
    eigenvector = b_k
    return eigenvalue, eigenvector

def inverse_power_iteration(H, num_iterations=1000, tol=1e-10):
    """
    Inverse Power Iteration method to find the smallest eigenvalue and corresponding eigenvector.
    """
    b_k = np.random.rand(H.shape[1])
    b_k /= np.linalg.norm(b_k)
    
    for _ in range(num_iterations):
        # Solve H * y = b_k
        y = spsolve(H, b_k)
        
        # Normalize the vector
        b_k1 = y / np.linalg.norm(y)
        
        # Check for convergence
        if np.linalg.norm(H.dot(b_k1) - y / np.linalg.norm(y)) < tol:
            break
        
        b_k = b_k1
    
    eigenvalue = b_k1.dot(H.dot(b_k1))
    eigenvector = b_k1
    return eigenvalue, eigenvector

def shifted_inverse_power_iteration(H, shift, num_iterations=1000, tol=1e-10):
    """
    Shifted Inverse Power Iteration to find the eigenvalue closest to the shift.
    """
    # Shift the Hamiltonian
    H_shifted = H - shift * diags(np.ones(H.shape[0]), 0, format='csc')
    
    b_k = np.random.rand(H.shape[1])
    b_k /= np.linalg.norm(b_k)
    
    for _ in range(num_iterations):
        # Solve (H - shift I) y = b_k
        try:
            y = spsolve(H_shifted, b_k)
        except:
            print("Shifted matrix is singular or ill-conditioned.")
            return None, None
        
        # Normalize the vector
        b_k1 = y / np.linalg.norm(y)
        
        # Check for convergence
        if np.linalg.norm(H_shifted.dot(b_k1) - y / np.linalg.norm(y)) < tol:
            break
        
        b_k = b_k1
    
    # Compute the eigenvalue using Rayleigh quotient
    eigenvalue = b_k1.dot(H.dot(b_k1))
    eigenvector = b_k1
    return eigenvalue, eigenvector

def find_eigenstates(H, num_states=5):
    """
    Find the first num_states eigenvalues and eigenvectors using Power Iteration, Inverse Power Iteration, and Shifted Inverse Power Iteration.
    """
    eigenvalues = []
    eigenvectors = []
    H_deflated = H.copy()
    
    for i in range(num_states):
        if i == 0:
            # The first state using Inverse Power Iteration (the fundamental state)
            print(f"Finding state {i+1} using Inverse Power Iteration...")
            eigenvalue, eigenvector = inverse_power_iteration(H_deflated)
        elif i == 1:
            # The second state using Shifted Inverse Power Iteration
            shift = -0.1  # Adjust the shift according to the expectations for the second state.
            print(f"Finding state {i+1} using Shifted Inverse Power Iteration with shift={shift}...")
            eigenvalue, eigenvector = shifted_inverse_power_iteration(H_deflated, shift)
            if eigenvalue is None:
                print("Shifted Inverse Power Iteration has failed. Using Inverse Power Iteration.")
                eigenvalue, eigenvector = inverse_power_iteration(H_deflated)
        else:
            # The next states using Power Iteration to find the larger eigenvalues.
            print(f"Finding state {i+1} using Power Iteration...")
            eigenvalue, eigenvector = power_iteration(H_deflated)
        
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        
        print(f"Eigenvalue found for state {i+1}: {eigenvalue:.6f}")
        
        # Deflation: H = H - λ |v><v|
        H_deflated = H_deflated - diags(eigenvalue * eigenvector**2, 0)
    
    return eigenvalues, eigenvectors

def plot_wavefunctions(eigenvectors, eigenvalues, N, x, y, z, num_states=5):
    """
    Plot 2D slices of the wave functions corresponding to the eigenvectors.
    """
    for i in range(num_states):
        eigenvector = eigenvectors[i].reshape((N, N, N))
        
        # Choose a central slice in z-direction
        slice_z = N // 2
        wave_slice = eigenvector[:, :, slice_z]
        
        plt.figure(figsize=(6,5))
        plt.contourf(x, y, wave_slice, levels=50, cmap='viridis')
        plt.title(f"Wave Function Slice for state {i+1}\nEnergy = {eigenvalues[i]:.4f}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

def plot_wavefunctions_3d(eigenvectors, eigenvalues, N, x, y, z, num_states=5, threshold=0.1):
    """
    Optional: Plot 3D isosurfaces of the wave functions.
    Note: This can be computationally intensive and may require significant memory.
    """
    from skimage import measure  # For extracting isosurfaces
    
    for i in range(num_states):
        eigenvector = eigenvectors[i].reshape((N, N, N))
        
        # Normalize the wave function for visualization
        eigenvector_normalized = eigenvector / np.max(np.abs(eigenvector))
        
        # Extract the isosurface at a given threshold
        verts, faces, normals, values = measure.marching_cubes(eigenvector_normalized, level=threshold)
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scale vertices to the grid
        verts[:, 0] = x[0] + verts[:, 0] * (x[-1] - x[0]) / N
        verts[:, 1] = y[0] + verts[:, 1] * (y[-1] - y[0]) / N
        verts[:, 2] = z[0] + verts[:, 2] * (z[-1] - z[0]) / N
        
        # Create the mesh
        mesh = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                               color='cyan', lw=1, alpha=0.7)
        
        ax.set_title(f"3D Isosurface of the Wave Function for State {i+1}\nEnergy = {eigenvalues[i]:.4f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.tight_layout()
        plt.show()

def main():
    print("Starting the eigenvalue calculation for the Coulomb potential in 3D...")
    start_time = time.time()
    
    # Find the first 5 eigenstates
    num_states = 5
    eigenvalues, eigenvectors = find_eigenstates(H, num_states=num_states)
    
    # Sort the eigenvalues and their corresponding eigenvectors.
    idx = np.argsort(eigenvalues)
    eigenvalues = np.array(eigenvalues)[idx]
    eigenvectors = np.array(eigenvectors)[idx]
    
    end_time = time.time()
    print(f"\nThe eigenvalue calculation has been completed in {end_time - start_time:.2f} seconds.\n")
    
    print("The first 5 energy levels (in atomic units):")
    for i, val in enumerate(eigenvalues):
        print(f"State {i+1}: Energy = {val:.6f}")
    
    # Plotting the wave functions (or 2D slices)
    plot_wavefunctions(eigenvectors, eigenvalues, N, x, y, z, num_states=num_states)
    
    # Optional: Plotting 3D isosurfaces (requires skimage)
    try:
        import skimage
        plot_3d = input("Do you want to plot the 3D isosurfaces of the wave functions? (y/n): ")
        if plot_3d.lower() == 'y':
            plot_wavefunctions_3d(eigenvectors, eigenvalues, N, x, y, z, num_states=num_states, threshold=0.1)
    except ImportError:
        print("The skimage library is not installed. 3D plotting is bypassed.")
    
if __name__ == "__main__":
    main()
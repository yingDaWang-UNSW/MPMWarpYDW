"""
Create a 50m x 50m x 200m rock domain with spatially varying properties
using Gaussian random fields (blurred noise) for realistic heterogeneity.

Properties vary in realistic ranges for weak-to-medium strength rock:
- Young's modulus: 5-20 GPa
- Yield stress: 10-100 MPa
- Cohesion: 1-10 MPa
- Friction angle: 25-45 degrees
- Density: 2400-2800 kg/m³
- Strain criteria: 0.001-0.05 (failure strain)

Uses Gaussian blur to create spatially correlated property fields.
"""

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

# Domain parameters
Lx, Ly, Lz = 50.0, 50.0, 200.0  # meters
spacing = 1.0  # meters
nx = int(Lx / spacing) + 1  # 51
ny = int(Ly / spacing) + 1  # 51
nz = int(Lz / spacing) + 1  # 201

print(f"Creating domain: {Lx}m × {Ly}m × {Lz}m")
print(f"Spacing: {spacing}m")
print(f"Grid: {nx} × {ny} × {nz} = {nx*ny*nz} particles")

# Generate particle positions
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
n_particles = len(positions)
print(f"Total particles: {n_particles}")


def create_gaussian_field(shape, mean, std, correlation_length):
    """
    Create a spatially correlated random field using Gaussian blur.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the 3D grid (nx, ny, nz)
    mean : float
        Mean value of the field
    std : float
        Standard deviation of the field
    correlation_length : float
        Spatial correlation length in grid cells (blur sigma)
        Larger values = smoother, more correlated fields
    
    Returns:
    --------
    field : ndarray
        Flattened array of spatially correlated values
    """
    # Generate white noise
    noise = np.random.randn(*shape)
    
    # Apply Gaussian blur for spatial correlation
    # sigma in grid cells (correlation_length / spacing)
    sigma = correlation_length / spacing
    blurred = gaussian_filter(noise, sigma=sigma, mode='wrap')
    
    # Normalize and scale to desired mean and std
    blurred = (blurred - blurred.mean()) / blurred.std()
    field = mean + std * blurred
    
    return field.flatten()


# Set random seed for reproducibility
np.random.seed(42)

# Correlation lengths (in meters) - controls how "patchy" the properties are
# Larger = more uniform regions, Smaller = more variable
correlation_length_short = 5.0   # For highly variable properties
correlation_length_medium = 10.0  # For moderately variable properties
correlation_length_long = 20.0    # For slowly varying properties

print("\nGenerating spatially correlated property fields...")

# Young's modulus: 5-20 GPa (mean 12.5 GPa, std ~3.75 GPa for 95% in range)
E_mean = 12.5e9
E_std = 3.75e9
E = create_gaussian_field((nx, ny, nz), E_mean, E_std, correlation_length_medium)
E = np.clip(E, 5e9, 20e9)  # Enforce bounds
print(f"Young's modulus: {E.min()/1e9:.1f} - {E.max()/1e9:.1f} GPa")

# Yield stress: 10-100 MPa (mean 55 MPa)
ys_mean = 55e6
ys_std = 22.5e6
ys = create_gaussian_field((nx, ny, nz), ys_mean, ys_std, correlation_length_short)
ys = np.clip(ys, 10e6, 100e6)
print(f"Yield stress: {ys.min()/1e6:.1f} - {ys.max()/1e6:.1f} MPa")

# Cohesion: 1-10 MPa (mean 5.5 MPa) - correlates with yield stress
cohesion_mean = 5.5e6
cohesion_std = 2.25e6
cohesion = create_gaussian_field((nx, ny, nz), cohesion_mean, cohesion_std, correlation_length_short)
cohesion = np.clip(cohesion, 1e6, 10e6)
print(f"Cohesion: {cohesion.min()/1e6:.1f} - {cohesion.max()/1e6:.1f} MPa")

# Friction angle: 25-45 degrees (mean 35 degrees)
friction_mean = 35.0
friction_std = 5.0
friction_angle = create_gaussian_field((nx, ny, nz), friction_mean, friction_std, correlation_length_medium)
friction_angle = np.clip(friction_angle, 25.0, 45.0)
print(f"Friction angle: {friction_angle.min():.1f} - {friction_angle.max():.1f} degrees")

# Convert to alpha parameter for Drucker-Prager
# alpha ≈ 2*sin(phi) / (sqrt(3)*(3-sin(phi))) for plane strain
friction_rad = np.deg2rad(friction_angle)
sin_phi = np.sin(friction_rad)
alpha = (2.0 * sin_phi) / (np.sqrt(3.0) * (3.0 - sin_phi))
print(f"Alpha (DP): {alpha.min():.3f} - {alpha.max():.3f}")

# Dilation angle: 0-15 degrees (typically 0.2-0.5 of friction angle)
dilation_angle = 0.3 * friction_angle
print(f"Dilation angle: {dilation_angle.min():.1f} - {dilation_angle.max():.1f} degrees")

# Density: 2400-2800 kg/m³ (mean 2600 kg/m³)
density_mean = 2600.0
density_std = 100.0
density = create_gaussian_field((nx, ny, nz), density_mean, density_std, correlation_length_long)
density = np.clip(density, 2400.0, 2800.0)
print(f"Density: {density.min():.0f} - {density.max():.0f} kg/m³")

# Strain criteria: 0.001-0.05 (failure strain, mean 0.0255)
strain_mean = 0.0255
strain_std = 0.012
strainCriteria = create_gaussian_field((nx, ny, nz), strain_mean, strain_std, correlation_length_short)
strainCriteria = np.clip(strainCriteria, 0.001, 0.05)
print(f"Strain criteria: {strainCriteria.min():.4f} - {strainCriteria.max():.4f}")

# Poisson's ratio: 0.2-0.3 (less variable)
nu_mean = 0.25
nu_std = 0.025
nu = create_gaussian_field((nx, ny, nz), nu_mean, nu_std, correlation_length_long)
nu = np.clip(nu, 0.2, 0.3)
print(f"Poisson's ratio: {nu.min():.3f} - {nu.max():.3f}")

# Hardening/softening parameters
# Use slight hardening initially, then softening
hardening = np.full(n_particles, 0.1)  # Slight hardening
softening = np.full(n_particles, 0.15)  # Moderate softening (net softening)

# Viscous damping (very light, 1e5-1e6 Pa·s)
eta_shear_mean = 5e5
eta_shear_std = 2.5e5
eta_shear = create_gaussian_field((nx, ny, nz), eta_shear_mean, eta_shear_std, correlation_length_medium)
eta_shear = np.clip(eta_shear, 1e5, 1e6)
print(f"Eta shear: {eta_shear.min():.1e} - {eta_shear.max():.1e} Pa·s")

eta_bulk = 3.0 * eta_shear  # Bulk viscosity typically 3x shear

# Initialize state variables
particle_vol = np.full(n_particles, spacing**3)  # Volume per particle
particle_mass = density * particle_vol
materialLabel = np.ones(n_particles, dtype=np.int32)  # All MPM initially
activeLabel = np.ones(n_particles, dtype=np.int32)  # All active
accumulated_strain = np.zeros(n_particles)
damage = np.zeros(n_particles)

# Save to HDF5
output_file = "random_rock_domain_50x50x200.h5"
print(f"\nSaving to {output_file}...")

with h5py.File(output_file, 'w') as f:
    # Positions and basic properties (MUST match runMPMYDW.py expectations)
    f.create_dataset('x', data=positions.T)  # Transposed: shape (3, n_particles)
    f.create_dataset('particle_volume', data=particle_vol)
    f.create_dataset('mass', data=particle_mass)
    f.create_dataset('density', data=density)
    
    # Elastic properties
    f.create_dataset('E', data=E)
    f.create_dataset('nu', data=nu)
    
    # Plasticity properties (Drucker-Prager)
    f.create_dataset('ys', data=ys)
    f.create_dataset('alpha', data=alpha)  # Pressure sensitivity
    f.create_dataset('hardening', data=hardening)
    f.create_dataset('softening', data=softening)
    f.create_dataset('strainCriteria', data=strainCriteria)
    
    # Alternative formulation (for compatibility)
    f.create_dataset('cohesion', data=cohesion)
    f.create_dataset('friction_angle', data=friction_angle)
    f.create_dataset('dilation_angle', data=dilation_angle)
    
    # Viscous damping
    f.create_dataset('eta_shear', data=eta_shear)
    f.create_dataset('eta_bulk', data=eta_bulk)
    
    # State variables
    f.create_dataset('materialLabel', data=materialLabel)
    f.create_dataset('activeLabel', data=activeLabel)
    f.create_dataset('accumulated_strain', data=accumulated_strain)
    f.create_dataset('damage', data=damage)
    
    # Domain metadata
    f.attrs['Lx'] = Lx
    f.attrs['Ly'] = Ly
    f.attrs['Lz'] = Lz
    f.attrs['spacing'] = spacing
    f.attrs['n_particles'] = n_particles
    f.attrs['description'] = "Heterogeneous rock domain with Gaussian random field properties"

print("Done!")
print(f"\nDomain statistics:")
print(f"  Total mass: {particle_mass.sum()/1e9:.2f} × 10⁹ kg")
print(f"  Volume: {Lx*Ly*Lz:.0f} m³")
print(f"  Average density: {particle_mass.sum()/(Lx*Ly*Lz):.0f} kg/m³")
print(f"\nUse this in config JSON:")
print(f'  "domainFile": "benchmarks/cavingBenchmark/{output_file}"')

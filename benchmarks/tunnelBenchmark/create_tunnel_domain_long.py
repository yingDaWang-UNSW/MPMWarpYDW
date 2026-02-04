"""
Create a LONG tunnel domain for proper plane strain Kirsch validation.

The thin slab (Ly=4m) violates plane strain because σ_yy ≈ σ_h instead of σ_yy = ν(σ_xx + σ_zz).
This domain uses a longer tunnel (Ly=40m) so the central region achieves true plane strain.

Analysis should use only the central Y-slice where end effects are negligible.
"""

import numpy as np
import h5py
import os

# ============================================================
# DOMAIN PARAMETERS
# ============================================================

# Tunnel geometry
tunnel_radius = 10.0  # meters (a in Kirsch equations)
tunnel_center_xz = np.array([100.0, 100.0])  # X, Z center of domain

# Domain size
Lx = 200.0  # meters (20× tunnel radius total width)
Ly = 40.0   # meters - LONG TUNNEL for proper plane strain
Lz = 200.0  # meters (20× tunnel radius total height)

# Tunnel center is in the middle of Y
tunnel_center = np.array([tunnel_center_xz[0], Ly/2, tunnel_center_xz[1]])

# Particle spacing - using 0.5m (high-res) for reasonable accuracy with long tunnel
spacing = 0.5  # meters

# Material properties (elastic validation - very high yield stress)
E = 10e9        # 10 GPa - typical rock
nu = 0.25       # Poisson's ratio
density = 2600  # kg/m³
ys = 1e20       # Very high yield stress for elastic-only simulation

# Geostatic parameters
g = 9.81        # gravity m/s²
K0 = 0.5        # Lateral earth pressure coefficient
z_surface = Lz  # Surface at top of domain

# ============================================================
# GENERATE PARTICLE POSITIONS
# ============================================================

nx = int(Lx / spacing) + 1
ny = int(Ly / spacing) + 1
nz = int(Lz / spacing) + 1

print(f"Long Tunnel Benchmark Domain Generator")
print(f"=" * 50)
print(f"Tunnel radius: {tunnel_radius}m")
print(f"Domain: {Lx}m × {Ly}m × {Lz}m")
print(f"Tunnel length: {Ly}m (vs. previous 4m)")
print(f"Spacing: {spacing}m")
print(f"Initial grid: {nx} × {ny} × {nz} = {nx*ny*nz:,} particles")

# Generate regular grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

# ============================================================
# REMOVE TUNNEL (CIRCULAR HOLE) - WITH REFINED BOUNDARY
# ============================================================

# Distance from tunnel axis (Y-axis through tunnel center)
dx_from_center = positions[:, 0] - tunnel_center[0]
dz_from_center = positions[:, 2] - tunnel_center[2]

# Radial distance from tunnel axis
r_from_tunnel = np.sqrt(dx_from_center**2 + dz_from_center**2)

# Use refined sub-grid test for particles near tunnel boundary
refine_factor = 4
fine_spacing = spacing / refine_factor

tunnel_influence_radius = tunnel_radius + spacing * 2
near_tunnel = r_from_tunnel < tunnel_influence_radius

keep_mask = np.ones(len(positions), dtype=bool)
near_indices = np.where(near_tunnel)[0]

# Create sub-grid offsets (only in XZ plane, tunnel is uniform in Y)
sub_offsets_1d = np.linspace(-spacing/2 + fine_spacing/2, 
                               spacing/2 - fine_spacing/2, 
                               refine_factor)
sub_dx, sub_dz = np.meshgrid(sub_offsets_1d, sub_offsets_1d, indexing='ij')
sub_dx = sub_dx.flatten()
sub_dz = sub_dz.flatten()

print(f"\nRefined tunnel exclusion (sub-grid factor: {refine_factor}x)...")

for idx in near_indices:
    px, py, pz = positions[idx]
    sub_x = px + sub_dx
    sub_z = pz + sub_dz
    sub_r = np.sqrt((sub_x - tunnel_center[0])**2 + (sub_z - tunnel_center[2])**2)
    if np.any(sub_r < tunnel_radius):
        keep_mask[idx] = False

positions = positions[keep_mask]
n_particles = len(positions)
print(f"After tunnel removal: {n_particles:,} particles")

# ============================================================
# COMPUTE PARTICLE VOLUMES
# ============================================================

# Simple uniform volume
particle_volume = np.full(n_particles, spacing**3)

# ============================================================
# ASSIGN MATERIAL PROPERTIES (SPATIALLY VARYING)
# ============================================================

# Uniform properties for elastic validation
density_array = np.full(n_particles, density)
E_array = np.full(n_particles, E)
nu_array = np.full(n_particles, nu)
ys_array = np.full(n_particles, ys)

# ============================================================
# SAVE TO HDF5
# ============================================================

output_file = "tunnel_domain_elastic_long.h5"
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)

print(f"\nSaving to: {output_path}")

with h5py.File(output_path, 'w') as f:
    # Positions: shape (3, N) as expected by the loader
    f.create_dataset('x', data=positions.T)
    
    # Particle volumes
    f.create_dataset('particle_volume', data=particle_volume)
    
    # Spatial properties
    f.create_dataset('density', data=density_array)
    f.create_dataset('E', data=E_array)
    f.create_dataset('nu', data=nu_array)
    f.create_dataset('ys', data=ys_array)
    
    # Metadata attributes
    f.attrs['tunnel_radius'] = tunnel_radius
    f.attrs['tunnel_center_x'] = tunnel_center[0]
    f.attrs['tunnel_center_y'] = tunnel_center[1]
    f.attrs['tunnel_center_z'] = tunnel_center[2]
    f.attrs['domain_Lx'] = Lx
    f.attrs['domain_Ly'] = Ly
    f.attrs['domain_Lz'] = Lz
    f.attrs['spacing'] = spacing
    f.attrs['density'] = density
    f.attrs['E'] = E
    f.attrs['nu'] = nu
    f.attrs['K0'] = K0
    f.attrs['gravity'] = g

print(f"\nDatasets saved:")
print(f"  - x: shape {positions.T.shape}")
print(f"  - particle_volume: shape {particle_volume.shape}")
print(f"  - density, E, nu, ys: shape {density_array.shape}")

# Summary statistics
print(f"\n{'='*50}")
print(f"DOMAIN SUMMARY")
print(f"{'='*50}")
print(f"Particles: {n_particles:,}")
print(f"Domain: {Lx} × {Ly} × {Lz} m")
print(f"Tunnel radius: {tunnel_radius} m")
print(f"Tunnel center: ({tunnel_center[0]}, {tunnel_center[1]}, {tunnel_center[2]})")
print(f"Spacing: {spacing} m")
print(f"\nRecommended analysis:")
print(f"  - Use central Y-slice only: y = {Ly/2} ± {spacing/2}")
print(f"  - This avoids end effects from Y boundaries")
print(f"\nExpected plane strain in central region:")
print(f"  σ_yy should equal ν(σ_xx + σ_zz) ≈ {nu * (K0 + 1) * density * g * 100 / 1e6:.3f} MPa")

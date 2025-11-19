"""
Create HDF5 input file with spatially-varying material properties for MPM-XPBD simulation.

This script generates:
- Particle positions (x, y, z)
- Particle volumes
- Spatially-varying mechanical properties:
  - density
  - E (Young's modulus)
  - nu (Poisson's ratio)
  - ys (yield stress)
  - alpha (Drucker-Prager pressure sensitivity)
  - hardening
  - softening
  - eta_shear (shear viscosity)
  - eta_bulk (bulk viscosity)
  - strainCriteria (critical plastic strain for phase change)
"""

import numpy as np
import scipy.ndimage as ndi
import h5py
import pyvista as pv

# --- Settings ---
img_size = 100
center = (50, 50)
outer_radius = 500
inner_radius = 15
layer_count = 10
spacing = 1  # Real-world spacing between pixels/layers

# --- Generate annulus mask using EDT ---
mask = np.zeros((img_size, img_size), dtype=np.uint8)
mask[center] = 1
edt = ndi.distance_transform_edt(1 - mask)  # Distance from center

# Binary annulus
annulus = (edt <= outer_radius) & (edt >= inner_radius)

# Keep arch
arch_mask = annulus

# Get 2D coordinates of centroids (cell centers)
arch_indices = np.argwhere(arch_mask)
arch_coords_2d = (arch_indices + 0.5) * spacing  # cell-centered

# Extrude into 3D by repeating in Z
z_layers = np.arange(layer_count) * spacing
arch_coords_3d = np.vstack([
    np.hstack([arch_coords_2d, np.full((arch_coords_2d.shape[0], 1), z)]) for z in z_layers
])

# Add origin offset
arch_coords_3d = arch_coords_3d + 25
arch_coords_3d = arch_coords_3d[:, [2, 1, 0]]

nPoints = arch_coords_3d.shape[0]
print(f"Generated {nPoints} particles")

# --- Particle volumes (uniform for now) ---
particle_volume = np.full((nPoints,), spacing**3, dtype=np.float32)

# --- Spatially-varying material properties ---
# Extract coordinates for property assignment
x_coords = arch_coords_3d[:, 0]
y_coords = arch_coords_3d[:, 1]
z_coords = arch_coords_3d[:, 2]

# Compute normalized positions (0 to 1)
x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min() + 1e-10)
y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-10)
z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-10)

# Distance from center (radial)
center_x = x_coords.mean()
center_y = y_coords.mean()
radius = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
radius_norm = radius / (radius.max() + 1e-10)

# --- Define spatial variations ---
# Base values (you can adjust these)
base_density = 3000.0  # kg/m³
base_E = 1e9  # Pa
base_nu = 0.2
base_ys = 5e7  # Pa
base_alpha = 0.3
base_hardening = 0.3
base_softening = 0.0
base_eta_shear = 1e6  # Pa·s
base_eta_bulk = 1e6  # Pa·s
base_strainCriteria = 0.05

# Example 1: Weaker material at the top (lower yield stress)
density = np.full(nPoints, base_density, dtype=np.float32)
E = base_E * (0.5 + 0.5 * z_norm)  # Stiffer at bottom
nu = np.full(nPoints, base_nu, dtype=np.float32)
ys = base_ys * (0.3 + 0.7 * z_norm)**2  # Much weaker at top
alpha = np.full(nPoints, base_alpha, dtype=np.float32)

# Example 2: Higher hardening at the edges (radial variation)
hardening = base_hardening * (1.0 + 0.5 * radius_norm)  # More hardening at edges
softening = np.full(nPoints, base_softening, dtype=np.float32)

# Example 3: Higher viscosity at the bottom
eta_shear = base_eta_shear * (0.5 + 1.5 * z_norm)  # More viscous at bottom
eta_bulk = base_eta_bulk * (0.5 + 1.5 * z_norm)

# Example 4: Easier to damage at the top
strainCriteria = base_strainCriteria * (0.5 + 0.5 * z_norm)  # Lower threshold (easier damage) at top

# Convert to float32 for consistency
E = E.astype(np.float32)
ys = ys.astype(np.float32)
hardening = hardening.astype(np.float32)
eta_shear = eta_shear.astype(np.float32)
eta_bulk = eta_bulk.astype(np.float32)
strainCriteria = strainCriteria.astype(np.float32)

# --- Print statistics ---
print("\n=== Spatial Property Statistics ===")
print(f"Density:        {density.min():.2e} to {density.max():.2e} kg/m³")
print(f"Young's modulus: {E.min():.2e} to {E.max():.2e} Pa")
print(f"Poisson's ratio: {nu.min():.3f} to {nu.max():.3f}")
print(f"Yield stress:    {ys.min():.2e} to {ys.max():.2e} Pa")
print(f"Alpha:          {alpha.min():.3f} to {alpha.max():.3f}")
print(f"Hardening:      {hardening.min():.3f} to {hardening.max():.3f}")
print(f"Softening:      {softening.min():.3f} to {softening.max():.3f}")
print(f"Shear visc:     {eta_shear.min():.2e} to {eta_shear.max():.2e} Pa·s")
print(f"Bulk visc:      {eta_bulk.min():.2e} to {eta_bulk.max():.2e} Pa·s")
print(f"Strain criteria: {strainCriteria.min():.3f} to {strainCriteria.max():.3f}")
print("===================================\n")

# --- Save to HDF5 ---
h5_filename = "./arch_spatial_properties.h5"
with h5py.File(h5_filename, "w") as h5file:
    # Geometry
    h5file.create_dataset("x", data=arch_coords_3d.T)
    h5file.create_dataset("particle_volume", data=particle_volume)
    
    # Mechanical properties
    h5file.create_dataset("density", data=density)
    h5file.create_dataset("E", data=E)
    h5file.create_dataset("nu", data=nu)
    h5file.create_dataset("ys", data=ys)
    h5file.create_dataset("alpha", data=alpha)
    h5file.create_dataset("hardening", data=hardening)
    h5file.create_dataset("softening", data=softening)
    h5file.create_dataset("eta_shear", data=eta_shear)
    h5file.create_dataset("eta_bulk", data=eta_bulk)
    h5file.create_dataset("strainCriteria", data=strainCriteria)

print(f"Saved HDF5: {h5_filename} with {nPoints} particles and spatial properties")

# --- Save to VTP for ParaView visualization ---
cloud = pv.PolyData(arch_coords_3d)
cloud["volume"] = particle_volume
cloud["density"] = density
cloud["E"] = E
cloud["nu"] = nu
cloud["ys"] = ys
cloud["alpha"] = alpha
cloud["hardening"] = hardening
cloud["softening"] = softening
cloud["eta_shear"] = eta_shear
cloud["eta_bulk"] = eta_bulk
cloud["strainCriteria"] = strainCriteria

vtp_filename = "./arch_spatial_properties.vtp"
cloud.save(vtp_filename)
print(f"Saved VTP: {vtp_filename}")
print("You can open the VTP file in ParaView to visualize the spatial property distributions")

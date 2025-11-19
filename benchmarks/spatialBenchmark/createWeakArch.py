"""
Create HDF5 input file with WEAK spatially-varying material properties that will fail easily.

This script generates a weak arch that will bend and fail quickly for testing.
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

# --- WEAK material properties for easy failure ---
base_density = 2000.0  # kg/m³ (lighter)
base_E = 2e8  # Pa (MUCH weaker - 5x weaker than original)
base_nu = 0.2
base_ys = 5e6  # Pa (10x weaker yield stress!)
base_alpha = 0.3
base_hardening = 0.0  # No hardening (material doesn't strengthen)
base_softening = 0.2  # Moderate softening (material weakens after yield)
base_eta_shear = 1e5  # Pa·s (lower viscosity)
base_eta_bulk = 1e5  # Pa·s
base_strainCriteria = 0.02  # Lower threshold for easier MPM->XPBD conversion

# Very weak at the top of the arch (where it will bend/fail)
density = np.full(nPoints, base_density, dtype=np.float32)
E = base_E * (0.3 + 0.7 * z_norm)  # Top is only 30% as stiff
nu = np.full(nPoints, base_nu, dtype=np.float32)
ys = base_ys * (0.1 + 0.9 * z_norm)**3  # Top is VERY weak (10% of base)
alpha = np.full(nPoints, base_alpha, dtype=np.float32)

# Net effect: (hardening - softening) determines behavior
# At top (z_norm=1): hardening=0.0, softening=0.3 → net softening = -0.3
# At bottom (z_norm=0): hardening=0.0, softening=0.1 → net softening = -0.1
# Top softens more aggressively, bottom is more stable
hardening = base_hardening * np.ones(nPoints)  # No hardening anywhere
softening = base_softening * (0.5 + z_norm)  # More softening at top (0.1 to 0.3)

# Lower viscosity for faster deformation
eta_shear = base_eta_shear * np.ones(nPoints)
eta_bulk = base_eta_bulk * np.ones(nPoints)

# Much easier to trigger failure at top
strainCriteria = base_strainCriteria * (0.3 + 0.7 * z_norm)  # Top has very low threshold

# Convert to float32 for consistency
E = E.astype(np.float32)
ys = ys.astype(np.float32)
hardening = hardening.astype(np.float32)
softening = softening.astype(np.float32)
eta_shear = eta_shear.astype(np.float32)
eta_bulk = eta_bulk.astype(np.float32)
strainCriteria = strainCriteria.astype(np.float32)

# --- Print statistics ---
print("\n=== WEAK Material - Spatial Property Statistics ===")
print(f"Density:        {density.min():.2e} to {density.max():.2e} kg/m³")
print(f"Young's modulus: {E.min():.2e} to {E.max():.2e} Pa (WEAK!)")
print(f"Poisson's ratio: {nu.min():.3f} to {nu.max():.3f}")
print(f"Yield stress:    {ys.min():.2e} to {ys.max():.2e} Pa (VERY WEAK!)")
print(f"Alpha:          {alpha.min():.3f} to {alpha.max():.3f}")
print(f"Hardening:      {hardening.min():.3f} to {hardening.max():.3f}")
print(f"Softening:      {softening.min():.3f} to {softening.max():.3f}")
print(f"Net effect (H-S): {(hardening - softening).min():.3f} to {(hardening - softening).max():.3f} (negative = softening)")
print(f"Shear visc:     {eta_shear.min():.2e} to {eta_shear.max():.2e} Pa·s")
print(f"Bulk visc:      {eta_bulk.min():.2e} to {eta_bulk.max():.2e} Pa·s")
print(f"Strain criteria: {strainCriteria.min():.3f} to {strainCriteria.max():.3f} (LOW!)")
print("===================================================\n")

# --- Save to HDF5 ---
h5_filename = "./arch_weak_properties.h5"
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

print(f"Saved HDF5: {h5_filename} with {nPoints} particles and WEAK spatial properties")

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

vtp_filename = "./arch_weak_properties.vtp"
cloud.save(vtp_filename)
print(f"Saved VTP: {vtp_filename}")
print("This weak arch should bend and fail easily under gravity!")

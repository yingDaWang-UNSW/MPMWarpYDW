"""
Create HDF5 input file for MPM-XPBD coupling test.

Setup:
- Bottom layer: Rigid slab (very high yield stress, unyielding)
- Top layer: Weak block (zero yield stress, instantly converts to XPBD)

This tests whether XPBD particles properly load the MPM slab below.
"""

import numpy as np
import h5py
import pyvista as pv

# --- Domain dimensions ---
slab_width = 200.0   # meters
slab_depth = 200.0   # meters
slab_height = 20.0   # meters (rigid foundation)

block_width = 100.0  # meters (centered on slab)
block_depth = 100.0  # meters
block_height = 50.0  # meters (will fail and fall onto slab)

particle_spacing = 2.5  # meters (spacing between particle centers) - FINER RESOLUTION

# --- Generate rigid slab particles ---
print("Generating rigid slab...")
nx_slab = int(slab_width / particle_spacing)
ny_slab = int(slab_depth / particle_spacing)
nz_slab = int(slab_height / particle_spacing)

x_slab = np.linspace(0, slab_width, nx_slab)
y_slab = np.linspace(0, slab_depth, ny_slab)
z_slab = np.linspace(0, slab_height, nz_slab)

X_slab, Y_slab, Z_slab = np.meshgrid(x_slab, y_slab, z_slab, indexing='ij')
slab_coords = np.vstack([X_slab.ravel(), Y_slab.ravel(), Z_slab.ravel()]).T
n_slab = len(slab_coords)
print(f"  Slab particles: {n_slab}")

# --- Generate weak block particles (on top of slab) ---
print("Generating weak block...")
nx_block = int(block_width / particle_spacing)
ny_block = int(block_depth / particle_spacing)
nz_block = int(block_height / particle_spacing)

# Center the block on the slab
x_offset = (slab_width - block_width) / 2.0
y_offset = (slab_depth - block_depth) / 2.0
z_offset = slab_height  # Start at top of slab

x_block = np.linspace(x_offset, x_offset + block_width, nx_block)
y_block = np.linspace(y_offset, y_offset + block_depth, ny_block)
z_block = np.linspace(z_offset, z_offset + block_height, nz_block)

X_block, Y_block, Z_block = np.meshgrid(x_block, y_block, z_block, indexing='ij')
block_coords = np.vstack([X_block.ravel(), Y_block.ravel(), Z_block.ravel()]).T
n_block = len(block_coords)
print(f"  Block particles: {n_block}")

# --- Combine into single domain ---
all_coords = np.vstack([slab_coords, block_coords])
nPoints = len(all_coords)
print(f"\nTotal particles: {nPoints}")

# --- Particle volumes (uniform cubic) ---
particle_volume = np.full(nPoints, particle_spacing**3, dtype=np.float32)

# --- Material properties ---
# Initialize arrays
density = np.zeros(nPoints, dtype=np.float32)
E = np.zeros(nPoints, dtype=np.float32)
nu = np.full(nPoints, 0.2, dtype=np.float32)  # Same Poisson's ratio everywhere
ys = np.zeros(nPoints, dtype=np.float32)
alpha = np.full(nPoints, 0.3, dtype=np.float32)  # Same pressure sensitivity
hardening = np.zeros(nPoints, dtype=np.float32)
softening = np.zeros(nPoints, dtype=np.float32)
eta_shear = np.zeros(nPoints, dtype=np.float32)
eta_bulk = np.zeros(nPoints, dtype=np.float32)
strainCriteria = np.full(nPoints, 1.0, dtype=np.float32)  # High by default

# Slab properties (UNYIELDING - very high yield stress)
density[:n_slab] = 3000.0  # kg/m³ (concrete-like)
E[:n_slab] = 5e9  # Pa (5 GPa - like strong rock)
ys[:n_slab] = 1e9  # Pa (1 GPa - truly unyielding, won't yield under geostatic stress)
hardening[:n_slab] = 0.0
softening[:n_slab] = 0.0
eta_shear[:n_slab] = 1e7  # Pa·s
eta_bulk[:n_slab] = 1e7
strainCriteria[:n_slab] = 1.0  # High threshold (won't convert to XPBD)

# Block properties (WEAK - will fail easily)
density[n_slab:] = 2000.0  # kg/m³ (lighter)
E[n_slab:] = 1e7  # Pa (10 MPa - weak)
ys[n_slab:] = 1e4  # Pa (10 kPa - weak, will yield under its own weight)
alpha[n_slab:] = 0.0  # DISABLE pressure hardening for weak block (allow it to fail)
hardening[n_slab:] = 0.0
softening[n_slab:] = 0.5  # Softens rapidly after yield
eta_shear[n_slab:] = 1e5  # Pa·s (low viscosity)
eta_bulk[n_slab:] = 1e5
strainCriteria[n_slab:] = 0.000001  # Zero - convert to XPBD immediately upon yielding

# --- Print statistics ---
print("\n=== Material Properties ===")
print("\nRIGID SLAB (bottom layer):")
print(f"  Particles: {n_slab}")
print(f"  Density: {density[:n_slab].mean():.0f} kg/m³")
print(f"  Young's modulus: {E[:n_slab].mean():.2e} Pa")
print(f"  Yield stress: {ys[:n_slab].mean():.2e} Pa (UNYIELDING)")
print(f"  Mass: {(density[:n_slab] * particle_volume[:n_slab]).sum():.1f} kg")

print("\nWEAK BLOCK (top layer):")
print(f"  Particles: {n_block}")
print(f"  Density: {density[n_slab:].mean():.0f} kg/m³")
print(f"  Young's modulus: {E[n_slab:].mean():.2e} Pa")
print(f"  Yield stress: {ys[n_slab:].mean():.2e} Pa (ZERO - instant failure!)")
print(f"  Strain criteria: {strainCriteria[n_slab:].mean():.4f} (very low)")
print(f"  Mass: {(density[n_slab:] * particle_volume[n_slab:]).sum():.1f} kg")
print(f"  Weight: {(density[n_slab:] * particle_volume[n_slab:]).sum() * 9.81:.1f} N")

print("\nExpected behavior:")
print("  1. Block particles instantly yield (ys=0)")
print("  2. Block converts to XPBD granular material")
print("  3. XPBD particles fall onto rigid slab")
print("  4. Check if slab experiences realistic load from XPBD particles")
print("=====================================\n")

# --- Save to HDF5 ---
h5_filename = "./coupling_test_domain.h5"
with h5py.File(h5_filename, "w") as h5file:
    # Geometry (transpose for expected format)
    h5file.create_dataset("x", data=all_coords.T)
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

print(f"Saved HDF5: {h5_filename}")

# --- Save to VTP for ParaView visualization ---
cloud = pv.PolyData(all_coords)
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

# Add layer labels for visualization
layer_label = np.zeros(nPoints, dtype=np.float32)
layer_label[:n_slab] = 0  # Slab
layer_label[n_slab:] = 1  # Block
cloud["layer"] = layer_label

vtp_filename = "./coupling_test_domain.vtp"
cloud.save(vtp_filename)
print(f"Saved VTP: {vtp_filename}")
print("Open in ParaView and color by 'ys' or 'layer' to see the two regions")

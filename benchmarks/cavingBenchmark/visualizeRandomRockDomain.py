"""
Visualize the spatially varying properties of the random rock domain.
Creates slice plots showing property distributions at different heights.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the domain
filename = "random_rock_domain_50x50x200.h5"
print(f"Loading {filename}...")

with h5py.File(filename, 'r') as f:
    positions = f['x'][:].T  # Transpose back to (n_particles, 3)
    E = f['E'][:]
    ys = f['ys'][:]
    alpha = f['alpha'][:]
    cohesion = f['cohesion'][:]
    friction_angle = f['friction_angle'][:]
    density = f['density'][:]
    strainCriteria = f['strainCriteria'][:]
    eta_shear = f['eta_shear'][:]
    
    Lx = f.attrs['Lx']
    Ly = f.attrs['Ly']
    Lz = f.attrs['Lz']
    spacing = f.attrs['spacing']
    n_particles = f.attrs['n_particles']

print(f"Loaded {n_particles} particles")
print(f"Domain: {Lx}m × {Ly}m × {Lz}m")

# Reshape data to 3D grid
nx = int(Lx / spacing) + 1
ny = int(Ly / spacing) + 1
nz = int(Lz / spacing) + 1

E_grid = E.reshape((nx, ny, nz))
ys_grid = ys.reshape((nx, ny, nz))
alpha_grid = alpha.reshape((nx, ny, nz))
cohesion_grid = cohesion.reshape((nx, ny, nz))
friction_grid = friction_angle.reshape((nx, ny, nz))
density_grid = density.reshape((nx, ny, nz))
strain_grid = strainCriteria.reshape((nx, ny, nz))
eta_grid = eta_shear.reshape((nx, ny, nz))

# Create visualization
fig = plt.figure(figsize=(16, 12))

# Select three slice heights: bottom (z=50m), middle (z=100m), top (z=150m)
z_slices = [50, 100, 150]
z_indices = [int(z/spacing) for z in z_slices]

properties = [
    (E_grid / 1e9, "Young's Modulus (GPa)", 5, 20, 'viridis'),
    (ys_grid / 1e6, "Yield Stress (MPa)", 10, 100, 'plasma'),
    (cohesion_grid / 1e6, "Cohesion (MPa)", 1, 10, 'plasma'),
    (friction_grid, "Friction Angle (°)", 25, 45, 'RdYlGn'),
    (alpha_grid, "Alpha (DP)", 0.19, 0.36, 'RdYlGn'),
    (density_grid, "Density (kg/m³)", 2400, 2800, 'coolwarm'),
    (strain_grid, "Strain Criteria", 0.001, 0.05, 'hot_r'),
    (eta_grid / 1e5, "Eta Shear (×10⁵ Pa·s)", 1, 10, 'cividis'),
]

for i, (prop, title, vmin, vmax, cmap) in enumerate(properties):
    for j, (z_idx, z_val) in enumerate(zip(z_indices, z_slices)):
        ax = plt.subplot(len(properties), len(z_slices), i * len(z_slices) + j + 1)
        
        # Extract slice (X-Y plane at constant Z)
        slice_data = prop[:, :, z_idx].T  # Transpose for correct orientation
        
        im = ax.imshow(slice_data, origin='lower', cmap=cmap, 
                      vmin=vmin, vmax=vmax, aspect='equal',
                      extent=[0, Lx, 0, Ly])
        
        if j == 0:
            ax.set_ylabel(title, fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_title(f'Z = {z_val}m', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        ax.set_xlabel('X (m)', fontsize=8)
        if j == 0:
            ax.set_ylabel('Y (m)', fontsize=8)
        ax.tick_params(labelsize=8)

plt.suptitle(f'Spatially Varying Rock Properties - {filename}', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

output_png = "random_rock_properties_visualization.png"
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to {output_png}")

# Create histograms
fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

hist_data = [
    (E / 1e9, "Young's Modulus (GPa)", axes[0]),
    (ys / 1e6, "Yield Stress (MPa)", axes[1]),
    (cohesion / 1e6, "Cohesion (MPa)", axes[2]),
    (friction_angle, "Friction Angle (°)", axes[3]),
    (alpha, "Alpha (DP)", axes[4]),
    (density, "Density (kg/m³)", axes[5]),
    (strainCriteria, "Strain Criteria", axes[6]),
    (eta_shear / 1e5, "Eta Shear (×10⁵ Pa·s)", axes[7]),
]

for data, title, ax in hist_data:
    ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±σ: {std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
    ax.legend(fontsize=8)

# Remove the extra subplot
axes[-1].remove()

plt.suptitle('Property Distributions - Histograms', fontsize=14, fontweight='bold')
plt.tight_layout()

output_hist = "random_rock_properties_histograms.png"
plt.savefig(output_hist, dpi=150, bbox_inches='tight')
print(f"Saved histograms to {output_hist}")

print("\nProperty statistics:")
print(f"  E:        {E.min()/1e9:.1f} - {E.max()/1e9:.1f} GPa (mean: {E.mean()/1e9:.1f})")
print(f"  ys:       {ys.min()/1e6:.1f} - {ys.max()/1e6:.1f} MPa (mean: {ys.mean()/1e6:.1f})")
print(f"  cohesion: {cohesion.min()/1e6:.1f} - {cohesion.max()/1e6:.1f} MPa (mean: {cohesion.mean()/1e6:.1f})")
print(f"  friction: {friction_angle.min():.1f} - {friction_angle.max():.1f}° (mean: {friction_angle.mean():.1f})")
print(f"  alpha:    {alpha.min():.3f} - {alpha.max():.3f} (mean: {alpha.mean():.3f})")
print(f"  density:  {density.min():.0f} - {density.max():.0f} kg/m³ (mean: {density.mean():.0f})")
print(f"  strain:   {strainCriteria.min():.4f} - {strainCriteria.max():.4f} (mean: {strainCriteria.mean():.4f})")
print(f"  eta:      {eta_shear.min():.1e} - {eta_shear.max():.1e} Pa·s (mean: {eta_shear.mean():.1e})")

plt.show()

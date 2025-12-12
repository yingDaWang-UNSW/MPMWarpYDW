"""
Create HDF5 input file for dynamic MPM-XPBD benchmark.

Setup:
- Large weak base: 200m × 200m × 50m (z=0 to z=50)
- Small strong block: 50m × 50m × 50m centered above at z=100m (z=100 to z=150)
- Air gap between base and block for dynamic impact

Physics:
- Strong block falls onto weak base
- Weak base should yield and transition to XPBD on impact
- Strong block should remain MPM (high yield stress)
- Tests dynamic coupling and impact response
"""

import numpy as np
import h5py
import pyvista as pv

# --- Domain dimensions ---
# Weak base
base_width = 200.0   # meters (x direction)
base_depth = 200.0   # meters (y direction)
base_height = 50.0   # meters (z direction)

# Strong block (centered above base)
block_width = 50.0   # meters
block_depth = 50.0   # meters
block_height = 50.0  # meters
block_z_bottom = 100.0  # meters (bottom of block)

# Particle spacing
particle_spacing = 2.5  # meters

def create_dynamic_domain():
    """Create the dynamic benchmark domain with strong block above weak base."""
    
    print("\n" + "="*70)
    print("CREATING DYNAMIC BENCHMARK DOMAIN")
    print("="*70)
    
    # --- Generate base particles ---
    print("\nGenerating weak base particles...")
    nx_base = int(base_width / particle_spacing)
    ny_base = int(base_depth / particle_spacing)
    nz_base = int(base_height / particle_spacing)
    
    x_base = np.linspace(0, base_width, nx_base)
    y_base = np.linspace(0, base_depth, ny_base)
    z_base = np.linspace(0, base_height, nz_base)
    
    X_base, Y_base, Z_base = np.meshgrid(x_base, y_base, z_base, indexing='ij')
    base_coords = np.vstack([X_base.ravel(), Y_base.ravel(), Z_base.ravel()]).T
    n_base = len(base_coords)
    print(f"  Base particles: {n_base}")
    
    # --- Generate block particles (centered above base) ---
    print("\nGenerating strong block particles...")
    
    # Center block in x-y plane
    block_x_start = (base_width - block_width) / 2
    block_y_start = (base_depth - block_depth) / 2
    
    nx_block = int(block_width / particle_spacing)
    ny_block = int(block_depth / particle_spacing)
    nz_block = int(block_height / particle_spacing)
    
    x_block = np.linspace(block_x_start, block_x_start + block_width, nx_block)
    y_block = np.linspace(block_y_start, block_y_start + block_depth, ny_block)
    z_block = np.linspace(block_z_bottom, block_z_bottom + block_height, nz_block)
    
    X_block, Y_block, Z_block = np.meshgrid(x_block, y_block, z_block, indexing='ij')
    block_coords = np.vstack([X_block.ravel(), Y_block.ravel(), Z_block.ravel()]).T
    n_block = len(block_coords)
    print(f"  Block particles: {n_block}")
    
    # --- Combine coordinates ---
    coords = np.vstack([base_coords, block_coords])
    nPoints = len(coords)
    print(f"\nTotal particles: {nPoints}")
    
    # Create masks for base and block
    base_mask = np.zeros(nPoints, dtype=bool)
    block_mask = np.zeros(nPoints, dtype=bool)
    base_mask[:n_base] = True
    block_mask[n_base:] = True
    
    # --- Particle volumes (uniform cubic) ---
    particle_volume = np.full(nPoints, particle_spacing**3, dtype=np.float32)
    
    # --- Material properties ---
    # Common properties
    density = np.zeros(nPoints, dtype=np.float32)
    E = np.zeros(nPoints, dtype=np.float32)
    nu = np.full(nPoints, 0.2, dtype=np.float32)
    ys = np.zeros(nPoints, dtype=np.float32)
    alpha = np.full(nPoints, 0.3, dtype=np.float32)
    hardening = np.zeros(nPoints, dtype=np.float32)
    softening = np.zeros(nPoints, dtype=np.float32)
    eta_shear = np.zeros(nPoints, dtype=np.float32)
    eta_bulk = np.zeros(nPoints, dtype=np.float32)
    strainCriteria = np.zeros(nPoints, dtype=np.float32)
    
    # --- WEAK BASE properties ---
    # Low yield stress - will transition to XPBD on impact
    base_density = 2500.0  # kg/m³ (soil-like)
    base_E = 1e8  # Pa (100 MPa - moderately stiff)
    base_ys = 1e7  # Pa (100 kPa - weak, will yield on impact)
    
    density[base_mask] = base_density
    E[base_mask] = base_E
    ys[base_mask] = base_ys
    alpha[base_mask] = 0.3  # Pressure sensitivity
    hardening[base_mask] = 0.0
    softening[base_mask] = 0.2  # Some softening to trigger damage
    eta_shear[base_mask] = 1e5
    eta_bulk[base_mask] = 1e5
    strainCriteria[base_mask] = 0.01  # Low strain criteria for quick transition
    
    # --- STRONG BLOCK properties ---
    # High yield stress - stays MPM
    block_density = 3000.0  # kg/m³ (concrete-like)
    block_E = 5e9  # Pa (5 GPa - stiff)
    block_ys = 1e9  # Pa (1 GPa - unyielding)
    
    density[block_mask] = block_density
    E[block_mask] = block_E
    ys[block_mask] = block_ys
    alpha[block_mask] = 0.3
    hardening[block_mask] = 0.0
    softening[block_mask] = 0.0
    eta_shear[block_mask] = 1e7
    eta_bulk[block_mask] = 1e7
    strainCriteria[block_mask] = 1.0  # High threshold (won't convert)
    
    # --- Statistics ---
    print("\n" + "="*70)
    print("MATERIAL PROPERTIES")
    print("="*70)
    
    print("\nWEAK BASE (0-50m):")
    print(f"  Dimensions: {base_width}m × {base_depth}m × {base_height}m")
    print(f"  Particles: {n_base}")
    print(f"  Density: {base_density:.0f} kg/m³")
    print(f"  Young's modulus: {base_E:.2e} Pa")
    print(f"  Yield stress: {base_ys:.2e} Pa (WILL YIELD ON IMPACT)")
    print(f"  Strain criteria: {strainCriteria[base_mask].mean():.4f}")
    base_mass = (density[base_mask] * particle_volume[base_mask]).sum()
    base_weight = base_mass * 9.81
    print(f"  Total mass: {base_mass:.1f} kg")
    print(f"  Total weight: {base_weight/1e6:.2f} MN")
    
    print(f"\nSTRONG BLOCK (z={block_z_bottom}-{block_z_bottom + block_height}m, centered):")
    print(f"  Dimensions: {block_width}m × {block_depth}m × {block_height}m")
    print(f"  Center: ({base_width/2}m, {base_depth/2}m)")
    print(f"  Particles: {n_block}")
    print(f"  Density: {block_density:.0f} kg/m³")
    print(f"  Young's modulus: {block_E:.2e} Pa")
    print(f"  Yield stress: {block_ys:.2e} Pa (UNYIELDING)")
    print(f"  Strain criteria: {strainCriteria[block_mask].mean():.4f}")
    block_mass = (density[block_mask] * particle_volume[block_mask]).sum()
    block_weight = block_mass * 9.81
    print(f"  Total mass: {block_mass:.1f} kg")
    print(f"  Total weight: {block_weight/1e6:.2f} MN")
    
    # Impact analysis
    print("\n" + "="*70)
    print("IMPACT ANALYSIS")
    print("="*70)
    gap = block_z_bottom - base_height
    print(f"Air gap: {gap:.1f} m")
    impact_velocity = np.sqrt(2 * 9.81 * gap)
    print(f"Impact velocity (free fall): {impact_velocity:.2f} m/s")
    
    impact_area = block_width * block_depth
    impact_stress = block_weight / impact_area
    print(f"Impact area: {impact_area:.0f} m²")
    print(f"Static stress from block: {impact_stress/1e3:.2f} kPa")
    print(f"Base yield stress: {base_ys/1e3:.2f} kPa")
    print(f"Stress/Yield ratio: {impact_stress/base_ys:.4f}")
    
    # Dynamic impact stress (rough estimate)
    # Impact duration ~ 0.1-0.5s for typical impacts
    # Dynamic amplification factor ~ 2-5x for impulse loading
    print(f"\nExpected dynamic amplification: 2-5×")
    print(f"Expected dynamic stress: {impact_stress*2/1e3:.2f} - {impact_stress*5/1e3:.2f} kPa")
    if impact_stress * 2 > base_ys:
        print(">>> BASE WILL YIELD ON IMPACT <<<")
    else:
        print("Note: May need stronger block or weaker base for yielding")
    
    # --- Save HDF5 ---
    output_h5 = "dynamic_domain.h5"
    output_vtp = "dynamic_domain.vtp"
    
    print(f"\n{'='*70}")
    print(f"SAVING OUTPUT FILES")
    print(f"{'='*70}")
    
    with h5py.File(output_h5, "w") as f:
        # Coordinates (note: shape is (3, N) for this format)
        f.create_dataset("x", data=coords.T, dtype=np.float32)
        f.create_dataset("particle_volume", data=particle_volume)
        
        # Material properties
        f.create_dataset("density", data=density)
        f.create_dataset("E", data=E)
        f.create_dataset("nu", data=nu)
        f.create_dataset("ys", data=ys)
        f.create_dataset("alpha", data=alpha)
        f.create_dataset("hardening", data=hardening)
        f.create_dataset("softening", data=softening)
        f.create_dataset("eta_shear", data=eta_shear)
        f.create_dataset("eta_bulk", data=eta_bulk)
        f.create_dataset("strainCriteria", data=strainCriteria)
    
    print(f"  Saved: {output_h5}")
    
    # --- Save VTP for visualization ---
    points = coords
    mesh = pv.PolyData(points)
    
    # Add all properties to mesh for visualization
    mesh["density"] = density
    mesh["E"] = E
    mesh["ys"] = ys
    mesh["alpha"] = alpha
    mesh["strainCriteria"] = strainCriteria
    mesh["softening"] = softening
    mesh["region"] = np.where(base_mask, 0, 1).astype(np.int32)  # 0=base, 1=block
    
    mesh.save(output_vtp)
    print(f"  Saved: {output_vtp}")
    
    print("\n" + "="*70)
    print("DOMAIN CREATION COMPLETE")
    print("="*70)
    
    return output_h5, output_vtp


if __name__ == "__main__":
    create_dynamic_domain()

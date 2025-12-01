"""
Create HDF5 input files for MPM-XPBD coupling test - Column variant.

Setup:
- Tall column: 50m × 50m × 200m
- Bottom half (0-100m): Rigid foundation (high yield stress, stays MPM)
- Top half (100-200m): Variable properties (3 cases)

Cases:
1. SAME: Top same as bottom (control - no transition expected)
2. WEAK_SAME_DENSITY: Top weak (converts to XPBD), same density as bottom
3. WEAK_LIGHT: Top weak (converts to XPBD), 0.1× density

This tests whether XPBD particles properly load the MPM column below.
"""

import numpy as np
import h5py
import pyvista as pv

# --- Domain dimensions ---
column_width = 50.0   # meters (square cross-section)
column_depth = 50.0   # meters
column_height = 200.0 # meters (tall column)
split_height = 100.0  # meters (divide at midpoint)

particle_spacing = 2.5  # meters (spacing between particle centers)

def create_column_coupling_test(case_name, top_density, top_ys, top_strain_criteria):
    """
    Create coupling test for specified case.
    
    Args:
        case_name: Name identifier for output files
        top_density: Density for top half (kg/m³)
        top_ys: Yield stress for top half (Pa)
        top_strain_criteria: Damage threshold for top half
    """
    
    print(f"\n{'='*70}")
    print(f"CREATING CASE: {case_name}")
    print(f"{'='*70}")
    
    # --- Generate column particles ---
    print("Generating column particles...")
    nx = int(column_width / particle_spacing)
    ny = int(column_depth / particle_spacing)
    nz = int(column_height / particle_spacing)
    
    x = np.linspace(0, column_width, nx)
    y = np.linspace(0, column_depth, ny)
    z = np.linspace(0, column_height, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    nPoints = len(coords)
    
    # Identify bottom and top halves
    z_coords = coords[:, 2]
    bottom_mask = z_coords < split_height
    top_mask = z_coords >= split_height
    n_bottom = np.sum(bottom_mask)
    n_top = np.sum(top_mask)
    
    print(f"  Total particles: {nPoints}")
    print(f"  Bottom half (0-{split_height}m): {n_bottom} particles")
    print(f"  Top half ({split_height}-{column_height}m): {n_top} particles")
    
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
    strainCriteria = np.full(nPoints, 1.0, dtype=np.float32)
    
    # BOTTOM HALF properties (RIGID - same for all cases)
    bottom_density = 3000.0  # kg/m³ (concrete-like)
    bottom_ys = 1e9  # Pa (1 GPa - truly unyielding)
    
    density[bottom_mask] = bottom_density
    E[bottom_mask] = 5e9  # Pa (5 GPa - like strong rock)
    ys[bottom_mask] = bottom_ys
    hardening[bottom_mask] = 0.0
    softening[bottom_mask] = 0.0
    eta_shear[bottom_mask] = 1e7  # Pa·s
    eta_bulk[bottom_mask] = 1e7
    strainCriteria[bottom_mask] = 1.0  # High threshold (won't convert to XPBD)
    
    # TOP HALF properties (VARIABLE - depends on case)
    density[top_mask] = top_density
    E[top_mask] = 1e7  # Pa (10 MPa - weak for cases that fail)
    ys[top_mask] = top_ys
    if top_ys < 1e6:  # If weak, disable pressure hardening to ensure failure
        alpha[top_mask] = 0.0
    hardening[top_mask] = 0.0
    if top_ys < 1e6:  # If weak, add softening
        softening[top_mask] = 0.5
    eta_shear[top_mask] = 1e5 if top_ys < 1e6 else 1e7  # Lower viscosity if weak
    eta_bulk[top_mask] = 1e5 if top_ys < 1e6 else 1e7
    strainCriteria[top_mask] = top_strain_criteria
    
    # --- Print statistics ---
    print("\n=== Material Properties ===")
    print("\nBOTTOM HALF (0-100m) - RIGID FOUNDATION:")
    print(f"  Particles: {n_bottom}")
    print(f"  Density: {density[bottom_mask].mean():.0f} kg/m³")
    print(f"  Young's modulus: {E[bottom_mask].mean():.2e} Pa")
    print(f"  Yield stress: {ys[bottom_mask].mean():.2e} Pa (UNYIELDING)")
    print(f"  Mass: {(density[bottom_mask] * particle_volume[bottom_mask]).sum():.1f} kg")
    print(f"  Weight: {(density[bottom_mask] * particle_volume[bottom_mask]).sum() * 9.81:.1f} N")
    
    print(f"\nTOP HALF (100-200m) - {case_name}:")
    print(f"  Particles: {n_top}")
    print(f"  Density: {density[top_mask].mean():.0f} kg/m³")
    print(f"  Young's modulus: {E[top_mask].mean():.2e} Pa")
    print(f"  Yield stress: {ys[top_mask].mean():.2e} Pa")
    print(f"  Strain criteria: {strainCriteria[top_mask].mean():.4f}")
    print(f"  Mass: {(density[top_mask] * particle_volume[top_mask]).sum():.1f} kg")
    print(f"  Weight: {(density[top_mask] * particle_volume[top_mask]).sum() * 9.81:.1f} N")
    
    # Calculate expected stress at interface (z=100m)
    contact_area = column_width * column_depth
    top_weight = (density[top_mask] * particle_volume[top_mask]).sum() * 9.81
    expected_stress_from_top = top_weight / contact_area
    
    print(f"\nExpected stress at interface (z={split_height}m):")
    print(f"  Contact area: {contact_area:.1f} m²")
    print(f"  Top weight: {top_weight:.1f} N")
    print(f"  Expected stress from top: {expected_stress_from_top/1e3:.2f} kPa")
    
    if "SAME" in case_name:
        print("\nExpected behavior:")
        print("  1. Top remains MPM (no transition)")
        print("  2. Column behaves as single rigid body")
        print("  3. Stress profile follows ρgh geostatic gradient")
    else:
        print("\nExpected behavior:")
        print("  1. Top particles yield and convert to XPBD")
        print("  2. XPBD particles load bottom MPM foundation")
        print("  3. Check stress at interface matches top weight")
    print("="*70)
    
    # --- Save to HDF5 ---
    h5_filename = f"coupling_column_{case_name.lower()}.h5"
    with h5py.File(h5_filename, "w") as h5file:
        # Geometry (transpose for expected format)
        h5file.create_dataset("x", data=coords.T)
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
    
    print(f"\nSaved HDF5: {h5_filename}")
    
    # --- Save to VTP for ParaView visualization ---
    cloud = pv.PolyData(coords)
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
    layer_label[bottom_mask] = 0  # Bottom
    layer_label[top_mask] = 1  # Top
    cloud["layer"] = layer_label
    
    vtp_filename = f"coupling_column_{case_name.lower()}.vtp"
    cloud.save(vtp_filename)
    print(f"Saved VTP: {vtp_filename}")
    
    return {
        'case_name': case_name,
        'n_bottom': n_bottom,
        'n_top': n_top,
        'bottom_mass': (density[bottom_mask] * particle_volume[bottom_mask]).sum(),
        'top_mass': (density[top_mask] * particle_volume[top_mask]).sum(),
        'expected_stress': expected_stress_from_top,
    }

# --- Create all three cases ---
results = []

# Case 1: SAME (control - no transition)
result1 = create_column_coupling_test(
    case_name="SAME",
    top_density=3000.0,  # Same as bottom
    top_ys=1e9,          # Same as bottom (unyielding)
    top_strain_criteria=1.0  # High (won't convert)
)
results.append(result1)

# Case 2: WEAK_SAME_DENSITY (transitions to XPBD, same density)
result2 = create_column_coupling_test(
    case_name="WEAK_SAME_DENSITY",
    top_density=3000.0,  # Same as bottom
    top_ys=1e4,          # 10 kPa (weak - will fail)
    top_strain_criteria=0.0  # Zero - instant conversion
)
results.append(result2)

# Case 3: WEAK_LIGHT (transitions to XPBD, 0.1× density)
result3 = create_column_coupling_test(
    case_name="WEAK_LIGHT",
    top_density=300.0,   # 0.1× bottom density
    top_ys=1e4,          # 10 kPa (weak - will fail)
    top_strain_criteria=0.0  # Zero - instant conversion
)
results.append(result3)

# --- Summary comparison ---
print(f"\n{'='*70}")
print("SUMMARY - EXPECTED STRESS COMPARISON AT INTERFACE (z=100m)")
print(f"{'='*70}")
print(f"{'Case':<20} {'Top Mass (kg)':<15} {'Top Weight (N)':<15} {'Expected σ_zz (kPa)':<20}")
print("-"*70)

for r in results:
    top_weight = r['top_mass'] * 9.81
    print(f"{r['case_name']:<20} {r['top_mass']:<15.1f} {top_weight:<15.1f} {r['expected_stress']/1e3:<20.2f}")

print("-"*70)
print(f"\nExpected ratio WEAK_LIGHT / WEAK_SAME_DENSITY: {results[2]['expected_stress'] / results[1]['expected_stress']:.3f}")
print(f"(Should be 0.100 since density is 0.1×)")
print(f"{'='*70}\n")

print("Next steps:")
print("1. Create config files for each case")
print("2. Run simulations to t=10s")
print("3. Analyze stress at interface (z=100m) in bottom half")
print("4. Compare WEAK_LIGHT vs WEAK_SAME_DENSITY ratios")

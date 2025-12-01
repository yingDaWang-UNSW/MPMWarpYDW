"""
Stress comparison accounting for slab self-weight.
Isolates XPBD contribution by subtracting expected geostatic stress.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def read_vtp(filepath):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    positions = vtk_to_numpy(points.GetData())
    
    point_data = polydata.GetPointData()
    data = {'positions': positions}
    
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        array = vtk_to_numpy(point_data.GetArray(i))
        data[name] = array
    
    return data

# Load both files
solid_file = "output_coupling_test_solid/sim_step_0000010000_particles.vtp"
light_file = "output_coupling_test_light/sim_step_0000010000_particles.vtp"

print("Loading data...")
solid = read_vtp(solid_file)
light = read_vtp(light_file)

print(f"\n{'='*70}")
print("STRESS ANALYSIS WITH SLAB SELF-WEIGHT CORRECTION")
print(f"{'='*70}")

# Slab properties (from createCouplingTest.py)
slab_height = 20.0  # meters
slab_density = 3000.0  # kg/m³
g = 9.81  # m/s²

def analyze_with_correction(name, data, slab_height, slab_density, g):
    """Analyze stress accounting for slab self-weight."""
    pos = data['positions']
    stress_tensor = data['stress_tensor']
    sigma_zz = stress_tensor[:, 2]
    z = pos[:, 2]
    
    # Identify slab particles (original slab region: z < slab_height)
    slab_mask = z < slab_height
    slab_pos = pos[slab_mask]
    slab_sigma_zz = sigma_zz[slab_mask]
    slab_z = z[slab_mask]
    
    # Calculate expected geostatic stress for slab self-weight
    # σ_zz(depth) = -ρ * g * depth
    # depth = distance from top of slab
    z_top_slab = slab_height  # Top of slab at 20m
    depth_from_top = z_top_slab - slab_z
    expected_geostatic = -slab_density * g * depth_from_top  # Pa
    
    # Subtract geostatic to get XPBD contribution
    xpbd_contribution = slab_sigma_zz - expected_geostatic
    
    # Bottom 20% of SLAB only
    z_min_slab = slab_z.min()
    z_max_slab = slab_z.max()
    bottom_threshold = z_min_slab + 0.2 * (z_max_slab - z_min_slab)
    bottom_mask = slab_z < bottom_threshold
    
    bottom_sigma_zz = slab_sigma_zz[bottom_mask]
    bottom_expected = expected_geostatic[bottom_mask]
    bottom_xpbd = xpbd_contribution[bottom_mask]
    
    print(f"\n{name}:")
    print(f"  Slab particles: {np.sum(slab_mask)}")
    print(f"  Slab Z range: [{z_min_slab:.2f}, {z_max_slab:.2f}] m")
    print(f"  Bottom 20% slab particles: {np.sum(bottom_mask)}")
    
    print(f"\n  Bottom stress breakdown (kPa):")
    print(f"    Total observed σ_zz:        {np.mean(bottom_sigma_zz)/1e3:.2f} ± {np.std(bottom_sigma_zz)/1e3:.2f}")
    print(f"    Expected geostatic (slab):  {np.mean(bottom_expected)/1e3:.2f} ± {np.std(bottom_expected)/1e3:.2f}")
    print(f"    XPBD contribution:          {np.mean(bottom_xpbd)/1e3:.2f} ± {np.std(bottom_xpbd)/1e3:.2f}")
    
    # Also analyze top of slab (where XPBD contacts)
    top_threshold = z_max_slab - 0.2 * (z_max_slab - z_min_slab)
    top_mask = slab_z > top_threshold
    
    top_sigma_zz = slab_sigma_zz[top_mask]
    top_expected = expected_geostatic[top_mask]
    top_xpbd = xpbd_contribution[top_mask]
    
    print(f"\n  Top 20% stress breakdown (kPa, where XPBD contacts):")
    print(f"    Total observed σ_zz:        {np.mean(top_sigma_zz)/1e3:.2f} ± {np.std(top_sigma_zz)/1e3:.2f}")
    print(f"    Expected geostatic (slab):  {np.mean(top_expected)/1e3:.2f} ± {np.std(top_expected)/1e3:.2f}")
    print(f"    XPBD contribution:          {np.mean(top_xpbd)/1e3:.2f} ± {np.std(top_xpbd)/1e3:.2f}")
    
    return {
        'name': name,
        'bottom_total': np.mean(bottom_sigma_zz),
        'bottom_geostatic': np.mean(bottom_expected),
        'bottom_xpbd': np.mean(bottom_xpbd),
        'top_total': np.mean(top_sigma_zz),
        'top_geostatic': np.mean(top_expected),
        'top_xpbd': np.mean(top_xpbd),
    }

# Analyze both cases
solid_results = analyze_with_correction("SOLID", solid, slab_height, slab_density, g)
light_results = analyze_with_correction("LIGHT", light, slab_height, slab_density, g)

# Compare XPBD contributions
print(f"\n{'='*70}")
print("XPBD CONTRIBUTION COMPARISON (isolated from slab self-weight)")
print(f"{'='*70}")

# Bottom of slab
bottom_diff = light_results['bottom_xpbd'] - solid_results['bottom_xpbd']
if abs(solid_results['bottom_xpbd']) > 1e-6:
    bottom_pct = 100 * bottom_diff / solid_results['bottom_xpbd']
else:
    bottom_pct = 0.0

print(f"\nBottom of slab (XPBD contribution only):")
print(f"  SOLID XPBD contribution: {solid_results['bottom_xpbd']/1e3:.2f} kPa")
print(f"  LIGHT XPBD contribution: {light_results['bottom_xpbd']/1e3:.2f} kPa")
print(f"  Difference:              {bottom_diff/1e3:.2f} kPa ({bottom_pct:.1f}%)")

# Top of slab (where XPBD actually contacts)
top_diff = light_results['top_xpbd'] - solid_results['top_xpbd']
if abs(solid_results['top_xpbd']) > 1e-6:
    top_pct = 100 * top_diff / solid_results['top_xpbd']
else:
    top_pct = 0.0

print(f"\nTop of slab (where XPBD contacts):")
print(f"  SOLID XPBD contribution: {solid_results['top_xpbd']/1e3:.2f} kPa")
print(f"  LIGHT XPBD contribution: {light_results['top_xpbd']/1e3:.2f} kPa")
print(f"  Difference:              {top_diff/1e3:.2f} kPa ({top_pct:.1f}%)")

# Expected XPBD contributions
print(f"\n{'='*70}")
print("THEORETICAL EXPECTED VALUES")
print(f"{'='*70}")

# Block properties
block_volume = 100 * 100 * 50  # m³
solid_block_density = 3000.0  # kg/m³
light_block_density = 300.0   # kg/m³
contact_area = 100 * 100      # m²

solid_weight = block_volume * solid_block_density * g
light_weight = block_volume * light_block_density * g

expected_solid_stress = solid_weight / contact_area  # Pa
expected_light_stress = light_weight / contact_area  # Pa

print(f"\nExpected XPBD stress at contact surface (top of slab):")
print(f"  SOLID (3000 kg/m³): {expected_solid_stress/1e3:.2f} kPa")
print(f"  LIGHT (300 kg/m³):  {expected_light_stress/1e3:.2f} kPa")
print(f"  Expected ratio:     {expected_light_stress/expected_solid_stress:.3f} (should be 0.1)")

# Expected at bottom of slab
# Bottom has: self-weight of slab (~630 kPa) + XPBD contribution transmitted through slab
print(f"\nExpected stress at bottom of slab:")
bottom_depth = slab_height  # Full slab depth
expected_bottom_geostatic = slab_density * g * bottom_depth  # Pa (positive value)
print(f"  Self-weight baseline: {expected_bottom_geostatic/1e3:.2f} kPa")
print(f"  SOLID XPBD contribution: {expected_solid_stress/1e3:.2f} kPa (transmitted from top)")
print(f"  LIGHT XPBD contribution: {expected_light_stress/1e3:.2f} kPa (transmitted from top)")
print(f"  Expected SOLID total: {(expected_bottom_geostatic + expected_solid_stress)/1e3:.2f} kPa")
print(f"  Expected LIGHT total: {(expected_bottom_geostatic + expected_light_stress)/1e3:.2f} kPa")
print(f"  Expected difference:  {(expected_solid_stress - expected_light_stress)/1e3:.2f} kPa")
print(f"  Expected ratio at bottom: {(expected_bottom_geostatic + expected_light_stress)/(expected_bottom_geostatic + expected_solid_stress):.3f}")

print(f"\n{'='*70}")
print("FINAL ASSESSMENT")
print(f"{'='*70}")

# Use top of slab for comparison (most relevant)
if abs(top_pct) < 10:
    print(f"\n✓ SIMILAR: Top XPBD contributions differ by {abs(top_pct):.1f}% (< 10%)")
    print("  -> Possible mass transfer issue OR both cases have similar artifacts")
elif top_pct < -50:
    print(f"\n✓ LARGE REDUCTION: LIGHT has {abs(top_pct):.1f}% less XPBD loading")
    print("  -> Mass coupling working! Lighter XPBD → less compression")
    
    observed_ratio = light_results['top_xpbd'] / solid_results['top_xpbd'] if solid_results['top_xpbd'] != 0 else 0
    print(f"  -> Observed ratio: {observed_ratio:.3f}")
    print(f"  -> Expected ratio: 0.1 (10x lighter)")
    
    if abs(observed_ratio - 0.1) < 0.2:
        print("  -> ✓✓ EXCELLENT: Ratio matches expectation!")
    elif abs(observed_ratio - 0.1) < 0.4:
        print("  -> ✓ GOOD: Ratio reasonably close to expectation")
    else:
        print("  -> ⚠ Ratio deviates from expectation, investigate further")
else:
    print(f"\n⚠ UNEXPECTED: {top_pct:+.1f}% change")
    print("  -> Investigate coupling mechanism")

print(f"\n{'='*70}\n")

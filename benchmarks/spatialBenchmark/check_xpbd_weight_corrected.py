"""
CORRECTED ANALYSIS - Proper sign convention.
Compression = NEGATIVE stress
More negative = more compression = more weight on top
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

print("Loading simulation results...")
same = read_vtp("output_coupling_column_same/sim_step_0000010000_particles.vtp")
weak_same = read_vtp("output_coupling_column_weak_same_density/sim_step_0000010000_particles.vtp")
weak_light = read_vtp("output_coupling_column_weak_light/sim_step_0000010000_particles.vtp")

print(f"\n{'='*70}")
print("XPBD WEIGHT CONTRIBUTION - CORRECTED SIGN ANALYSIS")
print(f"{'='*70}")
print("\nSign convention: COMPRESSION = NEGATIVE")
print("More negative σ_zz = More compression = More weight on top")

# Constants
rho = 3000.0  # kg/m³
g = 9.81

def analyze_bottom_stress(name, data):
    """Analyze stress in bottom half."""
    pos = data['positions']
    stress_tensor = data['stress_tensor']
    sigma_zz = stress_tensor[:, 2]
    z = pos[:, 2]
    
    # Bottom half only (z < 100m)
    bottom_mask = z < 100.0
    bottom_z = z[bottom_mask]
    bottom_sigma_zz = sigma_zz[bottom_mask]
    
    # Analyze at key locations
    depths = [
        ("Bottom (0-20m)", 0, 20),
        ("Middle (40-60m)", 40, 60),
        ("Near interface (80-100m)", 80, 100),
    ]
    
    print(f"\n{name}:")
    
    results = {}
    for label, z_min, z_max in depths:
        mask = (bottom_z >= z_min) & (bottom_z < z_max)
        if np.sum(mask) > 0:
            mean_stress = bottom_sigma_zz[mask].mean()
            mean_z = bottom_z[mask].mean()
            results[label] = {
                'z': mean_z,
                'stress': mean_stress,
                'n': np.sum(mask)
            }
            print(f"  {label}: σ_zz = {mean_stress/1e3:7.2f} kPa (z={mean_z:.1f}m)")
    
    return results

same_results = analyze_bottom_stress("SAME (Control - MPM on top)", same)
weak_same_results = analyze_bottom_stress("WEAK_SAME_DENSITY (XPBD on top, 3000 kg/m³)", weak_same)
weak_light_results = analyze_bottom_stress("WEAK_LIGHT (XPBD on top, 300 kg/m³)", weak_light)

print(f"\n{'='*70}")
print("STRESS COMPARISON - CORRECTED")
print(f"{'='*70}")

for label in same_results.keys():
    if label in weak_same_results and label in weak_light_results:
        same_stress = same_results[label]['stress']
        weak_same_stress = weak_same_results[label]['stress']
        weak_light_stress = weak_light_results[label]['stress']
        
        print(f"\n{label}:")
        print(f"  SAME:             {same_stress/1e3:8.2f} kPa (MPM column)")
        print(f"  WEAK_SAME_DENSITY:{weak_same_stress/1e3:8.2f} kPa")
        print(f"  WEAK_LIGHT:       {weak_light_stress/1e3:8.2f} kPa")
        
        # Comparison (remember: more negative = more compression)
        if weak_same_stress < same_stress:  # More negative
            diff_kpa = abs(weak_same_stress - same_stress) / 1e3
            print(f"  → WEAK_SAME has {diff_kpa:.0f} kPa MORE compression than SAME")
            print(f"     (XPBD is adding weight) ✓")
        elif weak_same_stress > same_stress:  # Less negative
            diff_kpa = abs(weak_same_stress - same_stress) / 1e3
            print(f"  → WEAK_SAME has {diff_kpa:.0f} kPa LESS compression than SAME")
            print(f"     (XPBD is lighter/softer than MPM)")
        else:
            print(f"  → Same compression (no difference)")
        
        if weak_light_stress < same_stress:  # More negative
            diff_kpa = abs(weak_light_stress - same_stress) / 1e3
            print(f"  → WEAK_LIGHT has {diff_kpa:.0f} kPa MORE compression than SAME")
            print(f"     (XPBD is adding weight) ✓")
        elif weak_light_stress > same_stress:  # Less negative
            diff_kpa = abs(weak_light_stress - same_stress) / 1e3
            print(f"  → WEAK_LIGHT has {diff_kpa:.0f} kPa LESS compression than SAME")
            print(f"     (XPBD is lighter/softer than MPM)")

# Overall average
print(f"\n{'='*70}")
print("OVERALL BOTTOM STRESS (average of all z < 100m)")
print(f"{'='*70}")

same_pos = same['positions']
weak_same_pos = weak_same['positions']
weak_light_pos = weak_light['positions']

same_sigma = same['stress_tensor'][:, 2]
weak_same_sigma = weak_same['stress_tensor'][:, 2]
weak_light_sigma = weak_light['stress_tensor'][:, 2]

same_bottom = same_sigma[same_pos[:, 2] < 100].mean()
weak_same_bottom = weak_same_sigma[weak_same_pos[:, 2] < 100].mean()
weak_light_bottom = weak_light_sigma[weak_light_pos[:, 2] < 100].mean()

print(f"\n  SAME:             {same_bottom/1e3:8.2f} kPa")
print(f"  WEAK_SAME_DENSITY:{weak_same_bottom/1e3:8.2f} kPa")
print(f"  WEAK_LIGHT:       {weak_light_bottom/1e3:8.2f} kPa")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

# Check if WEAK cases are MORE or LESS compressed than SAME
if weak_same_bottom < same_bottom and weak_light_bottom < same_bottom:
    # More negative = more compression
    print("\n✓ XPBD particles ARE adding weight (more compression in weak cases)")
    
    added_compression_same = abs(weak_same_bottom - same_bottom)
    added_compression_light = abs(weak_light_bottom - same_bottom)
    
    print(f"\nAdded compression from XPBD:")
    print(f"  WEAK_SAME_DENSITY: {added_compression_same/1e3:.0f} kPa")
    print(f"  WEAK_LIGHT:        {added_compression_light/1e3:.0f} kPa")
    print(f"  Ratio: {added_compression_light / added_compression_same:.3f}")
    
    print(f"\nExpected:")
    print(f"  If density works: ratio = 0.100 (10× lighter)")
    print(f"  If density broken: ratio ≈ 1.0 (same weight)")
    
    ratio = added_compression_light / added_compression_same
    if abs(ratio - 0.1) < 0.05:
        print(f"\n✓✓ EXCELLENT: Ratio = {ratio:.3f} ≈ 0.100")
        print(f"   Density IS working correctly!")
    elif abs(ratio - 1.0) < 0.1:
        print(f"\n✗✗ FAIL: Ratio = {ratio:.3f} ≈ 1.0")
        print(f"   Density is NOT working (both have same mass)")
    else:
        print(f"\n? UNCLEAR: Ratio = {ratio:.3f}")
        print(f"   Intermediate value, investigate further")
        
elif weak_same_bottom > same_bottom and weak_light_bottom > same_bottom:
    # Less negative = less compression
    print("\n⚠ XPBD particles cause LESS compression than MPM")
    print("  This means XPBD top is lighter/softer than MPM top")
    
    reduced_compression_same = abs(weak_same_bottom - same_bottom)
    reduced_compression_light = abs(weak_light_bottom - same_bottom)
    
    print(f"\nReduced compression (compared to SAME):")
    print(f"  WEAK_SAME_DENSITY: {reduced_compression_same/1e3:.0f} kPa less")
    print(f"  WEAK_LIGHT:        {reduced_compression_light/1e3:.0f} kPa less")
    print(f"  Ratio: {reduced_compression_light / reduced_compression_same:.3f}")
    
    print(f"\nThis could mean:")
    print(f"  1. XPBD particles are lighter than MPM (expected for weak material)")
    print(f"  2. XPBD particles spread out more (lower stress concentration)")
    print(f"  3. Top MPM (SAME case) has higher E, creates more compression")
    
    ratio = reduced_compression_light / reduced_compression_same
    if abs(ratio - 0.1) < 0.05:
        print(f"\n✓ Ratio suggests 10× density difference is working")
    elif abs(ratio - 1.0) < 0.1:
        print(f"\n✗ Ratio ≈ 1.0 suggests density is NOT working")
    
else:
    print("\n? MIXED: One case more compressed, one less - complex behavior")

print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

# Compare the two WEAK cases directly
if abs(weak_same_bottom - weak_light_bottom) < 50e3:  # < 50 kPa difference
    print(f"\n✗✗ DENSITY BUG CONFIRMED")
    print(f"   WEAK_SAME and WEAK_LIGHT have nearly identical stress:")
    print(f"   WEAK_SAME:  {weak_same_bottom/1e3:.0f} kPa")
    print(f"   WEAK_LIGHT: {weak_light_bottom/1e3:.0f} kPa")
    print(f"   Difference: {abs(weak_same_bottom - weak_light_bottom)/1e3:.0f} kPa")
    print(f"\n   Expected difference: ~2650 kPa (if 10× density worked)")
    print(f"   XPBD particles are ignoring HDF5 density!")
else:
    diff_pct = abs(weak_same_bottom - weak_light_bottom) / abs(weak_same_bottom) * 100
    print(f"\n? Stress difference exists: {diff_pct:.1f}%")
    print(f"   Investigate if this matches 10× density ratio")

print(f"\n{'='*70}\n")

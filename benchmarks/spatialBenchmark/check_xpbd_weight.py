"""
Check if XPBD particles contribute ANY weight at all.

Compare stress in bottom half between:
1. SAME: No XPBD on top (pure MPM column)
2. WEAK_SAME_DENSITY: XPBD on top (same density)
3. WEAK_LIGHT: XPBD on top (0.1× density)

If XPBD contributes NO weight, all three should have identical stress.
If XPBD contributes weight, stress should be: SAME > WEAK_SAME > WEAK_LIGHT (if density worked)
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
print("DO XPBD PARTICLES CONTRIBUTE WEIGHT?")
print(f"{'='*70}")

# Constants
rho = 3000.0  # kg/m³
g = 9.81

def analyze_bottom_stress(name, data):
    """Analyze stress in bottom half only."""
    pos = data['positions']
    stress_tensor = data['stress_tensor']
    sigma_zz = stress_tensor[:, 2]
    z = pos[:, 2]
    
    # Bottom half only (z < 100m)
    bottom_mask = z < 100.0
    bottom_z = z[bottom_mask]
    bottom_sigma_zz = sigma_zz[bottom_mask]
    
    # Analyze at different depths in bottom half
    # Bottom (z=0-20m), Middle (z=40-60m), Top (z=80-100m)
    
    depths = [
        ("Bottom (0-20m)", 0, 20),
        ("Middle (40-60m)", 40, 60),
        ("Near interface (80-100m)", 80, 100),
    ]
    
    print(f"\n{name}:")
    print(f"  Bottom half particles: {np.sum(bottom_mask)}")
    
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
            print(f"  {label}: σ_zz = {mean_stress/1e3:.2f} kPa (z={mean_z:.1f}m, n={np.sum(mask)})")
    
    return results

same_results = analyze_bottom_stress("SAME (Control - no XPBD on top)", same)
weak_same_results = analyze_bottom_stress("WEAK_SAME_DENSITY (XPBD on top, 3000 kg/m³)", weak_same)
weak_light_results = analyze_bottom_stress("WEAK_LIGHT (XPBD on top, 300 kg/m³)", weak_light)

# Compare stress differences
print(f"\n{'='*70}")
print("STRESS COMPARISON - Does top XPBD add compression?")
print(f"{'='*70}")

for label in same_results.keys():
    if label in weak_same_results and label in weak_light_results:
        same_stress = same_results[label]['stress']
        weak_same_stress = weak_same_results[label]['stress']
        weak_light_stress = weak_light_results[label]['stress']
        
        # Difference from SAME (control)
        weak_same_diff = weak_same_stress - same_stress
        weak_light_diff = weak_light_stress - same_stress
        
        print(f"\n{label}:")
        print(f"  SAME:             {same_stress/1e3:8.2f} kPa (baseline - top is MPM)")
        print(f"  WEAK_SAME_DENSITY:{weak_same_stress/1e3:8.2f} kPa (diff: {weak_same_diff/1e3:+7.2f} kPa)")
        print(f"  WEAK_LIGHT:       {weak_light_stress/1e3:8.2f} kPa (diff: {weak_light_diff/1e3:+7.2f} kPa)")
        
        # Interpretation
        if abs(weak_same_diff) < 50e3 and abs(weak_light_diff) < 50e3:  # < 50 kPa difference
            print(f"  → ⚠ Differences < 50 kPa - XPBD contributes MINIMAL or NO weight!")
        elif weak_same_diff > 100e3:  # > 100 kPa more compression
            print(f"  → ✓ WEAK_SAME has more compression - XPBD DOES contribute weight")
        elif weak_same_diff < -100e3:  # > 100 kPa less compression
            print(f"  → ⚠ WEAK_SAME has LESS compression - unexpected!")

# Expected differences if XPBD works correctly
print(f"\n{'='*70}")
print("EXPECTED BEHAVIOR")
print(f"{'='*70}")

print(f"\nIf XPBD particles contribute weight correctly:")
print(f"  1. SAME case: Bottom supports 100m of MPM material (100m × 3000 kg/m³)")
print(f"  2. WEAK_SAME_DENSITY: Bottom supports 100m MPM + 100m XPBD (both 3000 kg/m³)")
print(f"     → Should have +2943 kPa more compression than SAME")
print(f"  3. WEAK_LIGHT: Bottom supports 100m MPM + 100m XPBD (300 kg/m³)")
print(f"     → Should have +294 kPa more compression than SAME")

print(f"\nIf XPBD particles are WEIGHTLESS (bug):")
print(f"  1. SAME case: Bottom supports 100m of MPM material")
print(f"  2. WEAK_SAME_DENSITY: Bottom supports 100m MPM only (XPBD ignored)")
print(f"     → Should have ~0 kPa difference from SAME")
print(f"  3. WEAK_LIGHT: Bottom supports 100m MPM only (XPBD ignored)")
print(f"     → Should have ~0 kPa difference from SAME")

# Average stress across all bottom regions
print(f"\n{'='*70}")
print("OVERALL BOTTOM STRESS COMPARISON")
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

print(f"\nAverage bottom stress (all z < 100m):")
print(f"  SAME:             {same_bottom/1e3:.2f} kPa")
print(f"  WEAK_SAME_DENSITY:{weak_same_bottom/1e3:.2f} kPa (diff: {(weak_same_bottom-same_bottom)/1e3:+.2f} kPa)")
print(f"  WEAK_LIGHT:       {weak_light_bottom/1e3:.2f} kPa (diff: {(weak_light_bottom-same_bottom)/1e3:+.2f} kPa)")

# Final verdict
diff_weak_same = abs(weak_same_bottom - same_bottom)
diff_weak_light = abs(weak_light_bottom - same_bottom)

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

if diff_weak_same < 200e3 and diff_weak_light < 200e3:  # < 200 kPa difference
    print(f"\n✗✗ XPBD PARTICLES ARE WEIGHTLESS!")
    print(f"   Weak cases differ by < 200 kPa from control")
    print(f"   Expected difference: ~2943 kPa for WEAK_SAME, ~294 kPa for WEAK_LIGHT")
    print(f"   XPBD particles are NOT contributing their weight to MPM foundation")
elif diff_weak_same > 2000e3:  # > 2000 kPa difference
    print(f"\n✓ XPBD PARTICLES DO CONTRIBUTE WEIGHT")
    print(f"   Weak cases show {diff_weak_same/1e3:.0f} kPa more compression")
    print(f"   XPBD particles ARE loading the MPM foundation")
    
    # Now check density
    ratio = (weak_light_bottom - same_bottom) / (weak_same_bottom - same_bottom)
    print(f"\n   Ratio of added stress: {ratio:.3f} (expected 0.100)")
    if abs(ratio - 0.1) < 0.05:
        print(f"   ✓✓ Density is working correctly!")
    else:
        print(f"   ✗ Density is NOT working (ratio should be 0.100)")
else:
    print(f"\n? UNCLEAR - Differences are in intermediate range")
    print(f"   WEAK_SAME diff: {diff_weak_same/1e3:.0f} kPa")
    print(f"   WEAK_LIGHT diff: {diff_weak_light/1e3:.0f} kPa")
    print(f"   May have partial weight transmission or other effects")

print(f"\n{'='*70}\n")

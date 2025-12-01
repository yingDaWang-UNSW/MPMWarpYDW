"""
Analyze stress at interface (z=100m) for column coupling tests.

Compares three cases:
1. SAME: Control (no transition)
2. WEAK_SAME_DENSITY: Top converts to XPBD, same density
3. WEAK_LIGHT: Top converts to XPBD, 0.1× density

Expected result: WEAK_LIGHT should have 0.1× the XPBD contribution of WEAK_SAME_DENSITY
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

# Load all three cases at final timestep
print("Loading simulation results...")
same = read_vtp("output_coupling_column_same/sim_step_0000010000_particles.vtp")
weak_same = read_vtp("output_coupling_column_weak_same_density/sim_step_0000010000_particles.vtp")
weak_light = read_vtp("output_coupling_column_weak_light/sim_step_0000010000_particles.vtp")

print(f"\n{'='*70}")
print("COLUMN COUPLING TEST - INTERFACE STRESS ANALYSIS")
print(f"{'='*70}")

# Domain parameters
interface_z = 100.0  # meters (split between top and bottom)
bottom_density = 3000.0  # kg/m³
g = 9.81  # m/s²

def analyze_interface(name, data, interface_z, expected_top_weight_per_area):
    """Analyze stress at interface, accounting for bottom self-weight."""
    pos = data['positions']
    stress_tensor = data['stress_tensor']
    sigma_zz = stress_tensor[:, 2]
    z = pos[:, 2]
    
    # Identify bottom particles only (z < interface_z)
    bottom_mask = z < interface_z
    bottom_pos = pos[bottom_mask]
    bottom_sigma_zz = sigma_zz[bottom_mask]
    bottom_z = z[bottom_mask]
    
    # Calculate expected geostatic stress for bottom self-weight
    # At interface (z=100m), depth from bottom surface is 100m
    depth_from_bottom = bottom_z  # Assuming bottom starts at z=0
    expected_bottom_geostatic = -bottom_density * g * depth_from_bottom  # Pa
    
    # Subtract to get contribution from top (XPBD or MPM)
    top_contribution = bottom_sigma_zz - expected_bottom_geostatic
    
    # Analyze particles near interface (95-100m range to get good sample)
    interface_thickness = 5.0  # meters
    interface_mask = (bottom_z > (interface_z - interface_thickness)) & (bottom_z <= interface_z)
    
    interface_sigma_zz = bottom_sigma_zz[interface_mask]
    interface_geostatic = expected_bottom_geostatic[interface_mask]
    interface_top_contribution = top_contribution[interface_mask]
    
    n_interface = np.sum(interface_mask)
    
    print(f"\n{name}:")
    print(f"  Bottom particles: {np.sum(bottom_mask)}")
    print(f"  Interface particles (z ∈ [{interface_z-interface_thickness:.1f}, {interface_z:.1f}]m): {n_interface}")
    
    if n_interface > 0:
        print(f"\n  Interface stress breakdown (kPa):")
        print(f"    Total observed σ_zz:        {np.mean(interface_sigma_zz)/1e3:.2f} ± {np.std(interface_sigma_zz)/1e3:.2f}")
        print(f"    Expected geostatic (bottom):{np.mean(interface_geostatic)/1e3:.2f} ± {np.std(interface_geostatic)/1e3:.2f}")
        print(f"    Top contribution:           {np.mean(interface_top_contribution)/1e3:.2f} ± {np.std(interface_top_contribution)/1e3:.2f}")
        print(f"    Expected from top weight:   {expected_top_weight_per_area/1e3:.2f} kPa")
        
        return {
            'name': name,
            'interface_total': np.mean(interface_sigma_zz),
            'interface_geostatic': np.mean(interface_geostatic),
            'interface_top': np.mean(interface_top_contribution),
            'expected_top': expected_top_weight_per_area,
        }
    else:
        print(f"  WARNING: No particles found at interface!")
        return None

# Expected top weights (from domain creation)
contact_area = 50.0 * 50.0  # m²
top_volume = 50.0 * 50.0 * 100.0  # m³

expected_same_weight = top_volume * 3000.0 * g  # Same density
expected_weak_same_weight = top_volume * 3000.0 * g  # Same density
expected_weak_light_weight = top_volume * 300.0 * g  # 0.1× density

same_stress = expected_same_weight / contact_area
weak_same_stress = expected_weak_same_weight / contact_area
weak_light_stress = expected_weak_light_weight / contact_area

# Analyze all cases
same_results = analyze_interface("SAME (Control)", same, interface_z, same_stress)
weak_same_results = analyze_interface("WEAK_SAME_DENSITY", weak_same, interface_z, weak_same_stress)
weak_light_results = analyze_interface("WEAK_LIGHT (0.1× density)", weak_light, interface_z, weak_light_stress)

# Compare XPBD contributions
print(f"\n{'='*70}")
print("TOP CONTRIBUTION COMPARISON (isolated from bottom self-weight)")
print(f"{'='*70}")

if same_results and weak_same_results and weak_light_results:
    print(f"\nTop contribution at interface (z={interface_z}m):")
    print(f"  SAME:             {same_results['interface_top']/1e3:.2f} kPa")
    print(f"  WEAK_SAME_DENSITY:{weak_same_results['interface_top']/1e3:.2f} kPa")
    print(f"  WEAK_LIGHT:       {weak_light_results['interface_top']/1e3:.2f} kPa")
    
    # Calculate ratios
    if abs(weak_same_results['interface_top']) > 1e-6:
        observed_ratio = weak_light_results['interface_top'] / weak_same_results['interface_top']
        print(f"\nObserved ratio (WEAK_LIGHT / WEAK_SAME_DENSITY):")
        print(f"  Observed: {observed_ratio:.3f}")
        print(f"  Expected: 0.100 (from 10× density difference)")
        print(f"  Error:    {abs(observed_ratio - 0.1) / 0.1 * 100:.1f}%")
        
        if abs(observed_ratio - 0.1) < 0.02:
            print(f"  ✓✓ EXCELLENT: Ratio matches expectation!")
        elif abs(observed_ratio - 0.1) < 0.05:
            print(f"  ✓ GOOD: Ratio reasonably close to expectation")
        else:
            print(f"  ⚠ WARNING: Ratio deviates from expectation")
    
    # Compare with expected theoretical values
    print(f"\n{'='*70}")
    print("THEORETICAL vs OBSERVED")
    print(f"{'='*70}")
    print(f"\nExpected top contributions (from weight/area):")
    print(f"  SAME:             {same_stress/1e3:.2f} kPa")
    print(f"  WEAK_SAME_DENSITY:{weak_same_stress/1e3:.2f} kPa")
    print(f"  WEAK_LIGHT:       {weak_light_stress/1e3:.2f} kPa")
    
    print(f"\nTransmission efficiency (observed / expected):")
    print(f"  SAME:             {same_results['interface_top'] / same_stress * 100:.1f}%")
    print(f"  WEAK_SAME_DENSITY:{weak_same_results['interface_top'] / weak_same_stress * 100:.1f}%")
    print(f"  WEAK_LIGHT:       {weak_light_results['interface_top'] / weak_light_stress * 100:.1f}%")

print(f"\n{'='*70}")
print("FINAL ASSESSMENT")
print(f"{'='*70}")

if weak_same_results and weak_light_results:
    diff = weak_light_results['interface_top'] - weak_same_results['interface_top']
    if abs(weak_same_results['interface_top']) > 1e-6:
        pct = 100 * diff / weak_same_results['interface_top']
        
        if abs(pct + 90) < 5:  # Within 5% of -90%
            print(f"\n✓✓ EXCELLENT: WEAK_LIGHT has {abs(pct):.1f}% less loading (expected 90%)")
            print("  -> Mass coupling working perfectly!")
        elif abs(pct + 90) < 15:  # Within 15% of -90%
            print(f"\n✓ GOOD: WEAK_LIGHT has {abs(pct):.1f}% less loading (expected 90%)")
            print("  -> Mass coupling working reasonably well")
        else:
            print(f"\n⚠ WARNING: WEAK_LIGHT has {pct:+.1f}% difference (expected -90%)")
            print("  -> Investigate coupling mechanism")

print(f"\n{'='*70}\n")

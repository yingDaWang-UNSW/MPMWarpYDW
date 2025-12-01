"""
Detailed analysis of SAME control case.
Should show pure geostatic stress profile: σ_zz(z) = -ρgh
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

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

print("Analyzing SAME control case...")
same = read_vtp("output_coupling_column_same/sim_step_0000010000_particles.vtp")

pos = same['positions']
stress_tensor = same['stress_tensor']
sigma_zz = stress_tensor[:, 2]
z = pos[:, 2]

# Constants
rho = 3000.0  # kg/m³ (uniform throughout)
g = 9.81      # m/s²
H = 200.0     # Total column height

print(f"\n{'='*70}")
print("SAME CASE - GEOSTATIC ANALYSIS")
print(f"{'='*70}")

print(f"\nGeometry:")
print(f"  Total particles: {len(pos)}")
print(f"  Z range: {z.min():.2f} - {z.max():.2f} m")
print(f"  Column height: {z.max() - z.min():.2f} m (expected {H:.2f} m)")

print(f"\nStress statistics:")
print(f"  σ_zz min: {sigma_zz.min()/1e3:.2f} kPa")
print(f"  σ_zz max: {sigma_zz.max()/1e3:.2f} kPa")
print(f"  σ_zz mean: {sigma_zz.mean()/1e3:.2f} kPa")
print(f"  σ_zz std: {sigma_zz.std()/1e3:.2f} kPa")

# Expected geostatic stress at different heights
# σ_zz(z) = -ρ*g*(H-z) where H is top of column
z_top = z.max()

print(f"\n{'='*70}")
print("EXPECTED vs OBSERVED (at key locations)")
print(f"{'='*70}")

# Analyze at different heights
heights = [0, 25, 50, 75, 100, 125, 150, 175, z_top]

print(f"\n{'Height (m)':<12} {'Expected (kPa)':<20} {'Observed (kPa)':<20} {'Diff (%)':<15}")
print("-"*70)

for h in heights:
    # Find particles near this height (±2.5m)
    mask = np.abs(z - h) < 2.5
    if np.sum(mask) > 0:
        observed_mean = sigma_zz[mask].mean()
        observed_std = sigma_zz[mask].std()
        
        # Expected: stress due to weight of material ABOVE this height
        depth_from_top = z_top - h
        expected = -rho * g * depth_from_top  # Pa
        
        if abs(expected) > 1e-6:
            diff_pct = 100 * (observed_mean - expected) / abs(expected)
        else:
            diff_pct = 0.0
        
        print(f"{h:<12.1f} {expected/1e3:<20.2f} {observed_mean/1e3:<20.2f} {diff_pct:<15.1f}")

# Interface analysis (z=100m)
print(f"\n{'='*70}")
print("INTERFACE ANALYSIS (z=100m)")
print(f"{'='*70}")

interface_z = 100.0
interface_thick = 5.0

# Bottom of interface (95-100m)
bottom_interface_mask = (z >= (interface_z - interface_thick)) & (z < interface_z)
n_bottom_interface = np.sum(bottom_interface_mask)

# Top of interface (100-105m)
top_interface_mask = (z >= interface_z) & (z < (interface_z + interface_thick))
n_top_interface = np.sum(top_interface_mask)

if n_bottom_interface > 0:
    bottom_sigma = sigma_zz[bottom_interface_mask].mean()
    bottom_z_mean = z[bottom_interface_mask].mean()
    depth_from_top = z_top - bottom_z_mean
    expected_bottom = -rho * g * depth_from_top
    
    print(f"\nBottom side of interface (z ∈ [{interface_z-interface_thick:.1f}, {interface_z:.1f}]m):")
    print(f"  Particles: {n_bottom_interface}")
    print(f"  Mean z: {bottom_z_mean:.2f} m")
    print(f"  Depth from top: {depth_from_top:.2f} m")
    print(f"  Expected σ_zz: {expected_bottom/1e3:.2f} kPa")
    print(f"  Observed σ_zz: {bottom_sigma/1e3:.2f} kPa")
    print(f"  Difference: {(bottom_sigma - expected_bottom)/1e3:.2f} kPa")

if n_top_interface > 0:
    top_sigma = sigma_zz[top_interface_mask].mean()
    top_z_mean = z[top_interface_mask].mean()
    depth_from_top = z_top - top_z_mean
    expected_top = -rho * g * depth_from_top
    
    print(f"\nTop side of interface (z ∈ [{interface_z:.1f}, {interface_z+interface_thick:.1f}]m):")
    print(f"  Particles: {n_top_interface}")
    print(f"  Mean z: {top_z_mean:.2f} m")
    print(f"  Depth from top: {depth_from_top:.2f} m")
    print(f"  Expected σ_zz: {expected_top/1e3:.2f} kPa")
    print(f"  Observed σ_zz: {top_sigma/1e3:.2f} kPa")
    print(f"  Difference: {(top_sigma - expected_top)/1e3:.2f} kPa")

# Stress jump at interface
if n_bottom_interface > 0 and n_top_interface > 0:
    stress_jump = top_sigma - bottom_sigma
    print(f"\nStress jump across interface:")
    print(f"  Δσ_zz = {stress_jump/1e3:.2f} kPa")
    print(f"  Expected: ~0 kPa (continuous material)")
    
    if abs(stress_jump) < 100e3:  # < 100 kPa
        print(f"  ✓ Continuous (as expected)")
    else:
        print(f"  ⚠ Large discontinuity (unexpected)")

# Check if geostatic profile is correct
print(f"\n{'='*70}")
print("GEOSTATIC PROFILE ASSESSMENT")
print(f"{'='*70}")

# Expected: linear profile from -ρgH at bottom to 0 at top
z_bottom = z.min()
expected_bottom_stress = -rho * g * (z_top - z_bottom)
expected_top_stress = 0.0

actual_bottom_mask = z < (z_bottom + 5)
actual_top_mask = z > (z_top - 5)

if np.sum(actual_bottom_mask) > 0 and np.sum(actual_top_mask) > 0:
    actual_bottom_stress = sigma_zz[actual_bottom_mask].mean()
    actual_top_stress = sigma_zz[actual_top_mask].mean()
    
    print(f"\nBottom (z ≈ {z_bottom:.1f}m):")
    print(f"  Expected: {expected_bottom_stress/1e3:.2f} kPa")
    print(f"  Observed: {actual_bottom_stress/1e3:.2f} kPa")
    print(f"  Error: {100*(actual_bottom_stress-expected_bottom_stress)/abs(expected_bottom_stress):.1f}%")
    
    print(f"\nTop (z ≈ {z_top:.1f}m):")
    print(f"  Expected: {expected_top_stress/1e3:.2f} kPa (free surface)")
    print(f"  Observed: {actual_top_stress/1e3:.2f} kPa")
    
    # Overall assessment
    bottom_error = abs(100*(actual_bottom_stress-expected_bottom_stress)/abs(expected_bottom_stress))
    top_error = abs(actual_top_stress)
    
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")
    
    if bottom_error < 10 and top_error < 50e3:  # < 10% bottom error, < 50 kPa top stress
        print(f"\n✓✓ EXCELLENT: Geostatic profile matches theory")
        print(f"   Bottom error: {bottom_error:.1f}%")
        print(f"   Top stress: {top_error/1e3:.1f} kPa (should be ~0)")
    elif bottom_error < 20:
        print(f"\n✓ GOOD: Geostatic profile reasonable")
        print(f"   Bottom error: {bottom_error:.1f}%")
        print(f"   Top stress: {top_error/1e3:.1f} kPa")
    else:
        print(f"\n⚠ WARNING: Geostatic profile has issues")
        print(f"   Bottom error: {bottom_error:.1f}% (should be < 20%)")
        print(f"   Top stress: {top_error/1e3:.1f} kPa (should be ~0)")

print(f"\n{'='*70}\n")

# Try to create a simple plot
try:
    # Bin particles by height and average stress
    z_bins = np.linspace(z.min(), z.max(), 50)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    binned_stress = []
    
    for i in range(len(z_bins)-1):
        mask = (z >= z_bins[i]) & (z < z_bins[i+1])
        if np.sum(mask) > 0:
            binned_stress.append(sigma_zz[mask].mean())
        else:
            binned_stress.append(np.nan)
    
    # Expected profile
    expected_stress = -rho * g * (z_top - z_centers)
    
    plt.figure(figsize=(10, 8))
    plt.plot(np.array(binned_stress)/1e3, z_centers, 'b.-', label='Observed', linewidth=2, markersize=8)
    plt.plot(expected_stress/1e3, z_centers, 'r--', label='Expected (ρgh)', linewidth=2)
    plt.axhline(y=100, color='k', linestyle=':', alpha=0.5, label='Interface (z=100m)')
    plt.xlabel('Vertical Stress σ_zz (kPa)', fontsize=12)
    plt.ylabel('Height z (m)', fontsize=12)
    plt.title('SAME Case: Stress Profile', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('same_case_stress_profile.png', dpi=150)
    print("Saved plot: same_case_stress_profile.png")
except Exception as e:
    print(f"Could not create plot: {e}")

"""
Check if XPBD particles are COMPLETELY WEIGHTLESS.

Compare:
1. SAME case: 100m MPM bottom + 100m MPM top (both 3000 kg/m³)
2. WEAK cases: 100m MPM bottom + 100m XPBD top
3. Expected if XPBD is weightless: Only 100m MPM creates stress

If XPBD is completely weightless, WEAK cases should have stress as if 
only 100m tall column exists (half the stress of SAME case).
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
same = read_vtp("output_coupling_column_same/sim_step_0000050000_particles.vtp")
weak_same = read_vtp("output_coupling_column_weak_same_density/sim_step_0000050000_particles.vtp")
weak_light = read_vtp("output_coupling_column_weak_light/sim_step_0000050000_particles.vtp")

print(f"\n{'='*70}")
print("ARE XPBD PARTICLES COMPLETELY WEIGHTLESS?")
print(f"{'='*70}")

# Constants
rho = 3000.0  # kg/m³
g = 9.81

# Get average bottom stress for each case
same_pos = same['positions']
weak_same_pos = weak_same['positions']
weak_light_pos = weak_light['positions']

same_sigma = same['stress_tensor'][:, 2]
weak_same_sigma = weak_same['stress_tensor'][:, 2]
weak_light_sigma = weak_light['stress_tensor'][:, 2]

# Bottom half stress (z < 100m)
same_bottom = same_sigma[same_pos[:, 2] < 100].mean()
weak_same_bottom = weak_same_sigma[weak_same_pos[:, 2] < 100].mean()
weak_light_bottom = weak_light_sigma[weak_light_pos[:, 2] < 100].mean()

print(f"\nObserved average stress in bottom half (z < 100m):")
print(f"  SAME (MPM top):       {same_bottom/1e3:8.2f} kPa")
print(f"  WEAK_SAME (XPBD top): {weak_same_bottom/1e3:8.2f} kPa")
print(f"  WEAK_LIGHT (XPBD top):{weak_light_bottom/1e3:8.2f} kPa")

print(f"\n{'='*70}")
print("THEORETICAL EXPECTATIONS")
print(f"{'='*70}")

# Case 1: SAME (200m tall MPM column, 3000 kg/m³)
# At midpoint (z=50m), depth from top = 150m
# Expected: σ = -ρ*g*h = -3000 * 9.81 * 150 = -4.41 MPa
expected_same_at_50m = -rho * g * 150
print(f"\nSAME case (200m MPM column):")
print(f"  Expected stress at z=50m: {expected_same_at_50m/1e3:.2f} kPa")
print(f"  (depth from top = 150m)")

# Case 2: If XPBD is COMPLETELY WEIGHTLESS (0 kg/m³)
# Only bottom 100m MPM contributes
# At midpoint (z=50m), depth from top of BOTTOM = 50m
# Expected: σ = -ρ*g*h = -3000 * 9.81 * 50 = -1.47 MPa
expected_weightless_at_50m = -rho * g * 50
print(f"\nIf XPBD is WEIGHTLESS (0 kg/m³):")
print(f"  Expected stress at z=50m: {expected_weightless_at_50m/1e3:.2f} kPa")
print(f"  (only 50m of MPM above this point)")

# Case 3: If XPBD has CORRECT weight (3000 kg/m³ for WEAK_SAME)
# Both halves contribute: bottom 100m + top 100m
# At midpoint (z=50m), depth from top = 150m (same as SAME case)
# Expected: σ = -ρ*g*h = -3000 * 9.81 * 150 = -4.41 MPa
expected_full_weight_at_50m = -rho * g * 150
print(f"\nIf XPBD has FULL WEIGHT (3000 kg/m³ for WEAK_SAME):")
print(f"  Expected stress at z=50m: {expected_full_weight_at_50m/1e3:.2f} kPa")
print(f"  (same as SAME case)")

# Case 4: If XPBD has SOME weight (intermediate)
# e.g., 50% weight: σ = -ρ*g*(50 + 0.5*100) = -3.0 MPa
expected_half_weight_at_50m = -rho * g * (50 + 0.5*100)
print(f"\nIf XPBD has 50% WEIGHT:")
print(f"  Expected stress at z=50m: {expected_half_weight_at_50m/1e3:.2f} kPa")

print(f"\n{'='*70}")
print("COMPARISON AT z=40-60m (MIDPOINT OF BOTTOM)")
print(f"{'='*70}")

# Extract stress at midpoint
def get_midpoint_stress(data):
    pos = data['positions']
    sigma = data['stress_tensor'][:, 2]
    mask = (pos[:, 2] >= 40) & (pos[:, 2] < 60)
    return sigma[mask].mean()

same_mid = get_midpoint_stress(same)
weak_same_mid = get_midpoint_stress(weak_same)
weak_light_mid = get_midpoint_stress(weak_light)

print(f"\nObserved stress at z=40-60m:")
print(f"  SAME:       {same_mid/1e3:8.2f} kPa")
print(f"  WEAK_SAME:  {weak_same_mid/1e3:8.2f} kPa")
print(f"  WEAK_LIGHT: {weak_light_mid/1e3:8.2f} kPa")

print(f"\nExpected scenarios:")
print(f"  Full 200m MPM column:       {expected_same_at_50m/1e3:8.2f} kPa")
print(f"  XPBD weightless (100m MPM): {expected_weightless_at_50m/1e3:8.2f} kPa")
print(f"  XPBD full weight (200m):    {expected_full_weight_at_50m/1e3:8.2f} kPa")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

# Calculate what fraction of weight XPBD has
# Observed = -ρ*g*(h_bottom + α*h_top) where α is effective weight fraction
# σ_obs = -ρ*g*(50 + α*100)
# Solve for α: α = (σ_obs/(-ρ*g) - 50) / 100

alpha_weak_same = (weak_same_mid / (-rho * g) - 50) / 100
alpha_weak_light = (weak_light_mid / (-rho * g) - 50) / 100

print(f"\nEffective weight fraction of XPBD top:")
print(f"  WEAK_SAME:  {alpha_weak_same:.3f} (1.0 = full weight, 0.0 = weightless)")
print(f"  WEAK_LIGHT: {alpha_weak_light:.3f}")

print(f"\nInterpretation:")
if alpha_weak_same < 0.05:
    print(f"  ✗✗ XPBD is essentially WEIGHTLESS (< 5% of expected weight)")
    print(f"     XPBD particles contribute no mass to stress calculations")
elif alpha_weak_same > 0.95:
    print(f"  ✓ XPBD has FULL WEIGHT (> 95% of expected)")
    print(f"     XPBD particles correctly contribute their mass")
    if abs(alpha_weak_same - alpha_weak_light) < 0.05:
        print(f"  ✗ But WEAK_SAME and WEAK_LIGHT have same weight!")
        print(f"     Density values are being ignored (both use same default)")
    else:
        print(f"  ✓ And density difference is working!")
elif 0.3 < alpha_weak_same < 0.7:
    print(f"  ? XPBD has PARTIAL WEIGHT (~{alpha_weak_same*100:.0f}%)")
    print(f"     Some weight transmission, but reduced")
else:
    print(f"  ? XPBD weight is {alpha_weak_same*100:.0f}% of expected")

# Check if both WEAK cases have same weight
diff_alpha = abs(alpha_weak_same - alpha_weak_light)
print(f"\nDifference in weight fractions:")
print(f"  |α_SAME - α_LIGHT| = {diff_alpha:.3f}")

if diff_alpha < 0.05:
    print(f"  ✗ Both XPBD cases have SAME effective weight")
    print(f"     Density is NOT being used (uniform default)")
elif abs(alpha_weak_light / alpha_weak_same - 0.1) < 0.05:
    print(f"  ✓ LIGHT has ~10% of SAME weight")
    print(f"     Density IS working correctly!")
else:
    print(f"  ? Weight ratio = {alpha_weak_light / alpha_weak_same:.3f}")

print(f"\n{'='*70}")
print("FINAL ANSWER")
print(f"{'='*70}")

if alpha_weak_same < 0.1:
    print(f"\n✗✗ YES - XPBD particles are COMPLETELY WEIGHTLESS")
    print(f"   Effective weight: {alpha_weak_same*100:.1f}% (essentially zero)")
    print(f"   Stress pattern matches 100m MPM column only")
elif alpha_weak_same > 0.9:
    print(f"\n✓ NO - XPBD particles HAVE weight")
    print(f"   Effective weight: {alpha_weak_same*100:.1f}% of expected")
    print(f"   BUT: Density values are being ignored (ratio ≈ 1.0)")
else:
    print(f"\n? PARTIAL - XPBD particles have REDUCED weight")
    print(f"   Effective weight: {alpha_weak_same*100:.1f}% of expected")
    print(f"   Some mass coupling, but not full")

print(f"\n{'='*70}\n")

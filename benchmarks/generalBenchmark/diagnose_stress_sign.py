"""
Diagnostic script to understand stress sign conventions and component comparison.

This script loads a VTP file and examines:
1. Sign convention (compression positive vs negative)
2. Mean stress vs vertical stress component
3. K0 coefficient validation
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path

# Configuration
vtp_file = "./benchmarks/generalBenchmark/outputElastic/sim_step_0000005000_particles.vtp"
density = 5000.0  # kg/m³
g = 9.81  # m/s²
K0 = 0.5  # Lateral earth pressure coefficient
E = 1e7  # Pa
nu = 0.2

# Load VTP file
if not Path(vtp_file).exists():
    print(f"File not found: {vtp_file}")
    exit(1)

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(vtp_file)
reader.Update()
polydata = reader.GetOutput()

# Get positions
points = polydata.GetPoints()
positions = numpy_support.vtk_to_numpy(points.GetData())
z_coords = positions[:, 2]

# Get mean stress
point_data = polydata.GetPointData()
mean_stress = numpy_support.vtk_to_numpy(point_data.GetArray('mean_stress'))

# Analysis
z_min = z_coords.min()
z_max = z_coords.max()
height = z_max - z_min

print("="*70)
print("STRESS DIAGNOSTIC ANALYSIS")
print("="*70)

print(f"\nGeometry:")
print(f"  Height: {height:.2f} m")
print(f"  Z range: [{z_min:.2f}, {z_max:.2f}] m")

print(f"\nMaterial Properties:")
print(f"  Density: {density} kg/m³")
print(f"  Gravity: {g} m/s²")
print(f"  K0: {K0}")
print(f"  E: {E/1e6:.1f} MPa")
print(f"  ν: {nu}")

# Analyze stress at bottom
bottom_mask = z_coords < (z_min + 5.0)  # Bottom 5m
mean_stress_bottom = mean_stress[bottom_mask].mean()
depth_bottom = z_max - z_coords[bottom_mask].mean()

print(f"\nBottom Region Analysis (z < {z_min + 5:.1f} m):")
print(f"  Average depth: {depth_bottom:.2f} m")
print(f"  Mean stress from simulation: {mean_stress_bottom/1e3:.2f} kPa")
print(f"  Sign: {'NEGATIVE (compression)' if mean_stress_bottom < 0 else 'POSITIVE (tension)'}")

# Expected stresses
sigma_zz_expected = -density * g * depth_bottom  # Compression negative
sigma_xx_expected = K0 * sigma_zz_expected
sigma_yy_expected = K0 * sigma_zz_expected
mean_stress_expected = (sigma_xx_expected + sigma_yy_expected + sigma_zz_expected) / 3.0

print(f"\nExpected Analytical Stresses (compression negative):")
print(f"  σ_zz (vertical): {sigma_zz_expected/1e3:.2f} kPa")
print(f"  σ_xx (horizontal): {sigma_xx_expected/1e3:.2f} kPa")
print(f"  σ_yy (horizontal): {sigma_yy_expected/1e3:.2f} kPa")
print(f"  Mean stress (σ_xx+σ_yy+σ_zz)/3: {mean_stress_expected/1e3:.2f} kPa")
print(f"  Trace (σ_xx+σ_yy+σ_zz): {(sigma_xx_expected+sigma_yy_expected+sigma_zz_expected)/1e3:.2f} kPa")

# Comparison
print(f"\nComparison:")
print(f"  Simulation mean_stress: {mean_stress_bottom/1e3:.2f} kPa")
print(f"  Analytical mean_stress: {mean_stress_expected/1e3:.2f} kPa")
print(f"  Ratio (sim/analytical): {mean_stress_bottom/mean_stress_expected:.3f}")
print(f"  Error: {(mean_stress_bottom-mean_stress_expected)/mean_stress_expected*100:.1f}%")

print(f"\n  Simulation mean_stress: {mean_stress_bottom/1e3:.2f} kPa")
print(f"  Analytical σ_zz: {sigma_zz_expected/1e3:.2f} kPa")
print(f"  Ratio (sim_mean/analytical_sigma_zz): {mean_stress_bottom/sigma_zz_expected:.3f}")
print(f"  ⚠️  This is WRONG comparison! Mean stress ≠ vertical stress component")

print(f"\nKey Insight:")
print(f"  For K0 = {K0}:")
print(f"    mean_stress = (2·K0 + 1)·σ_zz / 3 = {(2*K0+1)/3:.3f}·σ_zz")
print(f"    With K0={K0}: mean_stress = {(2*K0+1)/3:.3f}·σ_zz")
print(f"    Expected ratio: {(2*K0+1)/3:.3f}")
print(f"    Actual ratio: {mean_stress_bottom/sigma_zz_expected:.3f}")

if abs(mean_stress_bottom/sigma_zz_expected - (2*K0+1)/3) < 0.05:
    print(f"  ✅ MATCH! Simulation is correct.")
    print(f"  ❌ ERROR: You're comparing mean_stress to σ_zz (wrong!)")
    print(f"\n  SOLUTION: Extract σ_zz component from stress tensor, not mean_stress")
else:
    factor = mean_stress_bottom/sigma_zz_expected
    print(f"  ⚠️ Unexpected ratio: {factor:.3f}")
    if abs(factor - 0.5) < 0.05:
        print(f"  This matches your 50% observation!")
        print(f"  Possible cause: σ_zz only has half the expected magnitude?")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("To properly validate elastic geostatic stress:")
print("1. Extract full stress tensor from VTP (not just mean_stress)")
print("2. Compare σ_zz component to -ρ·g·depth")
print("3. Compare σ_xx, σ_yy to K0·σ_zz")
print("4. Mean stress is for volumetric validation, not vertical stress!")
print("="*70)

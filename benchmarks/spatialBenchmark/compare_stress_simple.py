"""
Simple comparison of bottom stress in coupling test using basic VTK reading.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def extract_bottom_stress(filepath, case_name):
    """Extract bottom stress from VTP file."""
    print(f"\nReading {case_name}...")
    print(f"  File: {filepath}")
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    # Get positions
    points = polydata.GetPoints()
    positions = vtk_to_numpy(points.GetData())
    
    # Get stress tensor
    stress_array = polydata.GetPointData().GetArray('stress_tensor')
    stress_tensor = vtk_to_numpy(stress_array)
    
    # sigma_zz is index 2 in [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
    sigma_zz = stress_tensor[:, 2]
    
    # Get z coordinates
    z = positions[:, 2]
    z_min, z_max = z.min(), z.max()
    
    # Bottom 20%
    bottom_threshold = z_min + 0.2 * (z_max - z_min)
    bottom_mask = z < bottom_threshold
    
    sigma_zz_bottom = sigma_zz[bottom_mask]
    
    mean_stress = np.mean(sigma_zz_bottom)
    std_stress = np.std(sigma_zz_bottom)
    
    print(f"  Total particles: {len(positions)}")
    print(f"  Z range: [{z_min:.3f}, {z_max:.3f}] m")
    print(f"  Bottom 20% particles: {np.sum(bottom_mask)}")
    print(f"  Bottom sigma_zz mean: {mean_stress/1e3:.2f} kPa")
    print(f"  Bottom sigma_zz std:  {std_stress/1e3:.2f} kPa")
    
    return mean_stress, std_stress

# File paths
solid_file = "d:/sourceCodes/MPMWarpYDW/benchmarks/spatialBenchmark/output_coupling_test_solid/sim_step_0000010000_particles.vtp"
light_file = "d:/sourceCodes/MPMWarpYDW/benchmarks/spatialBenchmark/output_coupling_test_light/sim_step_0000010000_particles.vtp"

print("="*70)
print("COUPLING TEST STRESS COMPARISON")
print("="*70)

# Extract stresses
solid_mean, solid_std = extract_bottom_stress(solid_file, "SOLID (normal XPBD density)")
light_mean, light_std = extract_bottom_stress(light_file, "LIGHT (10x lighter XPBD)")

# Compare
print(f"\n{'='*70}")
print("COMPARISON RESULTS:")
print(f"{'='*70}")

diff = light_mean - solid_mean
pct = 100 * diff / solid_mean

print(f"  Solid bottom stress:  {solid_mean/1e3:.2f} +/- {solid_std/1e3:.2f} kPa")
print(f"  Light bottom stress:  {light_mean/1e3:.2f} +/- {light_std/1e3:.2f} kPa")
print(f"  Difference:           {diff/1e3:.2f} kPa")
print(f"  Relative change:      {pct:.1f}%")

print(f"\n{'='*70}")
if abs(pct) < 5:
    print("RESULT: SIMILAR (< 5% difference)")
    print("  -> XPBD mass properly coupled to MPM")
elif pct < -5:
    print(f"RESULT: REDUCED by {abs(pct):.1f}%")
    print("  -> Expected: lighter XPBD reduces compressive stress on block")
    print("  -> Mass coupling working correctly!")
else:
    print(f"RESULT: UNEXPECTED ({pct:+.1f}% change)")
    print("  -> Investigate coupling mechanism")

print(f"{'='*70}\n")

# Save results to file
output_file = "d:/sourceCodes/MPMWarpYDW/benchmarks/spatialBenchmark/stress_comparison_results.txt"
with open(output_file, 'w') as f:
    f.write("COUPLING TEST STRESS COMPARISON\n")
    f.write("="*70 + "\n\n")
    f.write(f"SOLID (normal XPBD density):\n")
    f.write(f"  Bottom stress: {solid_mean/1e3:.2f} +/- {solid_std/1e3:.2f} kPa\n\n")
    f.write(f"LIGHT (10x lighter XPBD):\n")
    f.write(f"  Bottom stress: {light_mean/1e3:.2f} +/- {light_std/1e3:.2f} kPa\n\n")
    f.write(f"DIFFERENCE:\n")
    f.write(f"  Delta: {diff/1e3:.2f} kPa\n")
    f.write(f"  Relative: {pct:.1f}%\n\n")
    
    if abs(pct) < 5:
        f.write("CONCLUSION: Similar stresses (<5%) - proper coupling\n")
    elif pct < -5:
        f.write(f"CONCLUSION: Reduced by {abs(pct):.1f}% - expected behavior\n")
    else:
        f.write(f"CONCLUSION: Unexpected {pct:+.1f}% change\n")

print(f"Results saved to: {output_file}")

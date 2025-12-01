"""
Extract full stress tensor components from VTP file to diagnose the 50% issue.
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path

# Configuration
vtp_file = "./benchmarks/generalBenchmark/outputElastic/sim_step_0000005000_particles.vtp"
density = 5000.0
g = 9.81
K0 = 0.5

if not Path(vtp_file).exists():
    print(f"File not found: {vtp_file}")
    exit(1)

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(vtp_file)
reader.Update()
polydata = reader.GetOutput()

positions = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
z_coords = positions[:, 2]

point_data = polydata.GetPointData()

print("Available arrays in VTP file:")
for i in range(point_data.GetNumberOfArrays()):
    array = point_data.GetArray(i)
    print(f"  {array.GetName()}: {array.GetNumberOfComponents()} components, {array.GetNumberOfTuples()} tuples")

# Check if we have full stress tensor or just mean_stress
if point_data.HasArray('stress_tensor'):
    stress_tensor = numpy_support.vtk_to_numpy(point_data.GetArray('stress_tensor'))
    print(f"\n✓ Full stress tensor available: shape {stress_tensor.shape}")
    
    # Bottom particles
    bottom_mask = z_coords < (z_coords.min() + 5.0)
    z_bottom_mean = z_coords[bottom_mask].mean()
    depth = z_coords.max() - z_bottom_mean
    
    # Average stress tensor at bottom
    stress_bottom = stress_tensor[bottom_mask].mean(axis=0)
    if stress_tensor.shape[1] == 9:  # Flattened 3x3
        stress_bottom = stress_bottom.reshape(3, 3)
    
    print(f"\nBottom region stress tensor (Pa):")
    print(stress_bottom)
    print(f"\nDiagonal components:")
    print(f"  σ_xx: {stress_bottom[0,0]/1e3:.2f} kPa")
    print(f"  σ_yy: {stress_bottom[1,1]/1e3:.2f} kPa")
    print(f"  σ_zz: {stress_bottom[2,2]/1e3:.2f} kPa")
    
    # Expected
    sigma_zz_exp = -density * g * depth
    sigma_h_exp = K0 * sigma_zz_exp
    
    print(f"\nExpected (depth={depth:.2f}m):")
    print(f"  σ_zz: {sigma_zz_exp/1e3:.2f} kPa")
    print(f"  σ_h (K0=0.5): {sigma_h_exp/1e3:.2f} kPa")
    
    print(f"\nRatios:")
    print(f"  σ_zz sim/expected: {stress_bottom[2,2]/sigma_zz_exp:.3f}")
    print(f"  σ_xx sim/expected: {stress_bottom[0,0]/sigma_h_exp:.3f}")
    
else:
    print("\n❌ Full stress tensor not available in VTP file")
    print("   Only mean_stress is exported")
    print("\n   The VTP export needs to include full stress tensor components!")
    print("   Check fs5PlotUtils.py save_particles_to_vtp() function")
    
    # Work with what we have
    mean_stress = numpy_support.vtk_to_numpy(point_data.GetArray('mean_stress'))
    bottom_mask = z_coords < (z_coords.min() + 5.0)
    mean_bottom = mean_stress[bottom_mask].mean()
    depth = z_coords.max() - z_coords[bottom_mask].mean()
    
    sigma_zz_exp = -density * g * depth
    mean_exp = (2*K0 + 1) * sigma_zz_exp / 3
    
    print(f"\nUsing mean_stress only:")
    print(f"  Depth: {depth:.2f} m")
    print(f"  Mean stress (sim): {mean_bottom/1e3:.2f} kPa")
    print(f"  Expected σ_zz: {sigma_zz_exp/1e3:.2f} kPa")
    print(f"  Expected mean: {mean_exp/1e3:.2f} kPa")
    print(f"  Ratio mean/sigma_zz: {mean_bottom/sigma_zz_exp:.3f} (should be {(2*K0+1)/3:.3f} for K0={K0})")
    
    if abs(mean_bottom/sigma_zz_exp - 0.5) < 0.05:
        print(f"\n  ⚠️  FOUND IT! Ratio is ~0.5, matching your observation!")
        print(f"  This suggests σ_zz itself is only half the expected value")
        print(f"  OR: mean_stress = σ_zz/2 instead of (2K0+1)σ_zz/3")

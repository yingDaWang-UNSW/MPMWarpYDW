"""
Quick stress comparison for coupling test: solid vs light XPBD particles.
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
print("COUPLING TEST STRESS COMPARISON: SOLID vs LIGHT (10x) XPBD PARTICLES")
print(f"{'='*70}")

for name, data in [("SOLID", solid), ("LIGHT", light)]:
    pos = data['positions']
    stress_tensor = data['stress_tensor']  # [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
    
    # Extract sigma_zz (vertical stress, index 2)
    sigma_zz = stress_tensor[:, 2]
    
    # Find bottom 20% of particles
    z = pos[:, 2]
    z_min, z_max = z.min(), z.max()
    bottom_threshold = z_min + 0.2 * (z_max - z_min)
    bottom_mask = z < bottom_threshold
    
    bottom_sigma_zz = sigma_zz[bottom_mask]
    
    print(f"\n{name}:")
    print(f"  Total particles: {len(pos)}")
    print(f"  Z range: [{z_min:.2f}, {z_max:.2f}] m")
    print(f"  Bottom 20% particles: {np.sum(bottom_mask)}")
    print(f"  Bottom sigma_zz (kPa, compression = negative):")
    print(f"    Mean:   {np.mean(bottom_sigma_zz)/1e3:.2f}")
    print(f"    Median: {np.median(bottom_sigma_zz)/1e3:.2f}")
    print(f"    Std:    {np.std(bottom_sigma_zz)/1e3:.2f}")

# Comparison
solid_bottom_z = solid['positions'][:, 2]
solid_bottom_mask = solid_bottom_z < (solid_bottom_z.min() + 0.2 * (solid_bottom_z.max() - solid_bottom_z.min()))
solid_sigma_zz_bottom = solid['stress_tensor'][solid_bottom_mask, 2]

light_bottom_z = light['positions'][:, 2]
light_bottom_mask = light_bottom_z < (light_bottom_z.min() + 0.2 * (light_bottom_z.max() - light_bottom_z.min()))
light_sigma_zz_bottom = light['stress_tensor'][light_bottom_mask, 2]

diff = np.mean(light_sigma_zz_bottom) - np.mean(solid_sigma_zz_bottom)
pct = 100 * diff / np.mean(solid_sigma_zz_bottom)

print(f"\n{'='*70}")
print("DIFFERENCE (LIGHT - SOLID):")
print(f"{'='*70}")
print(f"  Delta sigma_zz:  {diff/1e3:.2f} kPa")
print(f"  Relative change: {pct:.1f}%")

if abs(pct) < 5:
    print(f"\nResult: SIMILAR (< 5% difference)")
    print("  -> XPBD mass properly transferred to MPM block")
elif abs(pct) < 15:
    print(f"\nResult: MODERATE DIFFERENCE ({abs(pct):.1f}%)")
    print("  -> Expected behavior: lighter XPBD reduces bottom stress")
else:
    print(f"\nResult: LARGE DIFFERENCE ({abs(pct):.1f}%)")
    print("  -> Potential coupling issue")

print(f"\n{'='*70}\n")

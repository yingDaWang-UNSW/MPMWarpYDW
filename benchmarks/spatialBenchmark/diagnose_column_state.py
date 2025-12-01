"""
Quick diagnostic: Check particle distribution and stress state
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

print("Checking simulation state at step 10000...\n")

for case in ['same', 'weak_same_density', 'weak_light']:
    filename = f"output_coupling_column_{case}/sim_step_0000010000_particles.vtp"
    print(f"\n{'='*70}")
    print(f"CASE: {case.upper()}")
    print(f"{'='*70}")
    
    data = read_vtp(filename)
    pos = data['positions']
    z = pos[:, 2]
    
    print(f"\nParticle distribution:")
    print(f"  Total particles: {len(pos)}")
    print(f"  Z range: [{z.min():.2f}, {z.max():.2f}] m")
    print(f"  Z mean: {z.mean():.2f} m")
    
    # Check material labels
    if 'materialLabel' in data:
        labels = data['materialLabel']
        n_mpm = np.sum(labels == 1)
        n_xpbd = np.sum(labels == 2)
        print(f"\n  MPM particles (label=1): {n_mpm}")
        print(f"  XPBD particles (label=2): {n_xpbd}")
    
    # Check stress
    if 'stress_tensor' in data:
        stress = data['stress_tensor']
        sigma_zz = stress[:, 2]
        
        # Count non-zero stresses
        nonzero = np.abs(sigma_zz) > 1e-6
        print(f"\n  Particles with non-zero σ_zz: {np.sum(nonzero)} / {len(sigma_zz)}")
        print(f"  σ_zz range: [{sigma_zz.min()/1e3:.2f}, {sigma_zz.max()/1e3:.2f}] kPa")
        print(f"  σ_zz mean (all): {sigma_zz.mean()/1e3:.2f} kPa")
        print(f"  σ_zz mean (nonzero): {sigma_zz[nonzero].mean()/1e3 if np.sum(nonzero) > 0 else 0:.2f} kPa")
        
        # Check bottom half specifically
        bottom_mask = z < 100.0
        bottom_sigma = sigma_zz[bottom_mask]
        bottom_nonzero = np.abs(bottom_sigma) > 1e-6
        print(f"\n  Bottom half (z < 100m):")
        print(f"    Particles: {np.sum(bottom_mask)}")
        print(f"    Non-zero σ_zz: {np.sum(bottom_nonzero)}")
        print(f"    σ_zz mean: {bottom_sigma.mean()/1e3:.2f} kPa")
        print(f"    σ_zz mean (nonzero): {bottom_sigma[bottom_nonzero].mean()/1e3 if np.sum(bottom_nonzero) > 0 else 0:.2f} kPa")
    
    # Check velocities
    if 'v' in data:
        v = data['v']
        v_mag = np.linalg.norm(v, axis=1)
        print(f"\n  Velocity magnitude:")
        print(f"    Mean: {v_mag.mean():.4f} m/s")
        print(f"    Max: {v_mag.max():.4f} m/s")
        print(f"    Particles moving (v > 0.01 m/s): {np.sum(v_mag > 0.01)}")

print(f"\n{'='*70}\n")

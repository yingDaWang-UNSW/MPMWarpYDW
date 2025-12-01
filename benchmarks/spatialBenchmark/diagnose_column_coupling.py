"""
Diagnostic script to understand what's happening in column coupling tests.
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

def diagnose(name, data):
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {name}")
    print(f"{'='*70}")
    
    pos = data['positions']
    z = pos[:, 2]
    
    # Check if density field exists
    if 'density' in data:
        density = data['density']
    else:
        print("  ⚠ WARNING: No density field in output!")
        density = None
    
    # Check material labels
    if 'materialLabel' in data:
        material_label = data['materialLabel']
        n_mpm = np.sum(material_label == 1)
        n_xpbd = np.sum(material_label == 2)
    else:
        print("  ⚠ WARNING: No materialLabel field in output!")
        n_mpm = 0
        n_xpbd = 0
    
    # Particle volume
    if 'particle_volume' in data:
        volume = data['particle_volume']
    else:
        print("  ⚠ WARNING: No particle_volume field in output!")
        volume = None
    
    print(f"\nParticle counts:")
    print(f"  Total particles: {len(pos)}")
    print(f"  MPM particles (label=1): {n_mpm}")
    print(f"  XPBD particles (label=2): {n_xpbd}")
    
    print(f"\nZ-coordinate range:")
    print(f"  Min: {z.min():.2f} m")
    print(f"  Max: {z.max():.2f} m")
    print(f"  Mean: {z.mean():.2f} m")
    
    # Analyze top vs bottom
    interface_z = 100.0
    bottom_mask = z < interface_z
    top_mask = z >= interface_z
    
    print(f"\nBottom half (z < {interface_z}m): {np.sum(bottom_mask)} particles")
    print(f"Top half (z ≥ {interface_z}m): {np.sum(top_mask)} particles")
    
    if density is not None and volume is not None:
        print(f"\nDensity analysis:")
        print(f"  All particles: {density.min():.1f} - {density.max():.1f} kg/m³, mean={density.mean():.1f}")
        
        if np.sum(bottom_mask) > 0:
            print(f"  Bottom: {density[bottom_mask].min():.1f} - {density[bottom_mask].max():.1f} kg/m³, mean={density[bottom_mask].mean():.1f}")
        if np.sum(top_mask) > 0:
            print(f"  Top: {density[top_mask].min():.1f} - {density[top_mask].max():.1f} kg/m³, mean={density[top_mask].mean():.1f}")
        
        # Calculate masses
        total_mass = (density * volume).sum()
        bottom_mass = (density[bottom_mask] * volume[bottom_mask]).sum()
        top_mass = (density[top_mask] * volume[top_mask]).sum()
        
        print(f"\nMass analysis:")
        print(f"  Total mass: {total_mass:.2e} kg")
        print(f"  Bottom mass: {bottom_mass:.2e} kg")
        print(f"  Top mass: {top_mass:.2e} kg")
        print(f"  Top weight: {top_mass * 9.81:.2e} N")
        print(f"  Expected stress at interface: {top_mass * 9.81 / (50*50):.2f} kPa")
    
    # Material label analysis
    if n_xpbd > 0 and density is not None:
        xpbd_mask = material_label == 2
        print(f"\nXPBD particles:")
        print(f"  Count: {n_xpbd}")
        print(f"  Z range: {z[xpbd_mask].min():.2f} - {z[xpbd_mask].max():.2f} m")
        print(f"  Density: {density[xpbd_mask].min():.1f} - {density[xpbd_mask].max():.1f} kg/m³, mean={density[xpbd_mask].mean():.1f}")
        if volume is not None:
            xpbd_mass = (density[xpbd_mask] * volume[xpbd_mask]).sum()
            print(f"  Total XPBD mass: {xpbd_mass:.2e} kg")
            print(f"  XPBD weight: {xpbd_mass * 9.81:.2e} N")
    
    # Stress analysis
    if 'stress_tensor' in data:
        stress = data['stress_tensor']
        sigma_zz = stress[:, 2]
        
        print(f"\nStress analysis (σ_zz):")
        print(f"  All particles: {sigma_zz.min()/1e3:.2f} - {sigma_zz.max()/1e3:.2f} kPa")
        
        if n_mpm > 0 and n_xpbd > 0:
            mpm_mask = material_label == 1
            xpbd_mask = material_label == 2
            print(f"  MPM particles: {sigma_zz[mpm_mask].min()/1e3:.2f} - {sigma_zz[mpm_mask].max()/1e3:.2f} kPa, mean={sigma_zz[mpm_mask].mean()/1e3:.2f}")
            print(f"  XPBD particles: {sigma_zz[xpbd_mask].min()/1e3:.2f} - {sigma_zz[xpbd_mask].max()/1e3:.2f} kPa, mean={sigma_zz[xpbd_mask].mean()/1e3:.2f}")

diagnose("SAME (Control)", same)
diagnose("WEAK_SAME_DENSITY", weak_same)
diagnose("WEAK_LIGHT (10× lighter)", weak_light)

# Compare masses
print(f"\n{'='*70}")
print("MASS COMPARISON")
print(f"{'='*70}")

if 'density' in weak_same and 'density' in weak_light and 'particle_volume' in weak_same:
    weak_same_top_mask = weak_same['positions'][:, 2] >= 100
    weak_light_top_mask = weak_light['positions'][:, 2] >= 100
    
    weak_same_top_mass = (weak_same['density'][weak_same_top_mask] * weak_same['particle_volume'][weak_same_top_mask]).sum()
    weak_light_top_mass = (weak_light['density'][weak_light_top_mask] * weak_light['particle_volume'][weak_light_top_mask]).sum()
    
    print(f"\nTop half masses:")
    print(f"  WEAK_SAME_DENSITY: {weak_same_top_mass:.2e} kg")
    print(f"  WEAK_LIGHT:        {weak_light_top_mass:.2e} kg")
    print(f"  Ratio: {weak_light_top_mass / weak_same_top_mass:.3f} (expected 0.100)")
    
    if abs(weak_light_top_mass / weak_same_top_mass - 0.1) < 0.01:
        print("  ✓ Masses are correct!")
    else:
        print("  ⚠ Mass ratio is wrong!")

print(f"\n{'='*70}\n")

"""
Compare stress profiles between solid and light XPBD particle coupling tests.
Analyzes bottom stress in MPM block to verify proper loading from XPBD particles.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

def read_vtp_file(filepath):
    """Read VTP file and extract particle data."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    # Extract data
    points = polydata.GetPoints()
    positions = vtk_to_numpy(points.GetData())
    
    # Get point data arrays
    point_data = polydata.GetPointData()
    
    data = {'positions': positions}
    
    # Extract all available arrays
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        array = vtk_to_numpy(point_data.GetArray(i))
        data[array_name] = array
    
    return data

def compute_stress_magnitude(stress_tensor):
    """
    Compute von Mises equivalent stress from stress tensor.
    
    Args:
        stress_tensor: (N, 6) or (N, 9) array representing stress tensors
                      If 6 components: [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz] (Voigt notation)
                      If 9 components: full 3x3 matrix
    
    Returns:
        (N,) array of von Mises stress, mean stress, and sigma_zz
    """
    if stress_tensor.shape[1] == 6:
        # Voigt notation: [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
        s_xx = stress_tensor[:, 0]
        s_yy = stress_tensor[:, 1]
        s_zz = stress_tensor[:, 2]
        s_xy = stress_tensor[:, 3]
        s_xz = stress_tensor[:, 4]
        s_yz = stress_tensor[:, 5]
        
        # Mean stress (pressure)
        p = (s_xx + s_yy + s_zz) / 3.0
        
        # Deviatoric stress
        s_dev_xx = s_xx - p
        s_dev_yy = s_yy - p
        s_dev_zz = s_zz - p
        
        # von Mises stress
        vm = np.sqrt(0.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2 + 
                            2*(s_xy**2 + s_xz**2 + s_yz**2)))
        
        return vm, p, s_zz
        
    elif stress_tensor.shape[1] == 9:
        # Reshape to (N, 3, 3)
        stress = stress_tensor.reshape(-1, 3, 3)
        
        # Extract components
        s_xx = stress[:, 0, 0]
        s_yy = stress[:, 1, 1]
        s_zz = stress[:, 2, 2]
        s_xy = stress[:, 0, 1]
        s_xz = stress[:, 0, 2]
        s_yz = stress[:, 1, 2]
        
        # Mean stress (pressure)
        p = (s_xx + s_yy + s_zz) / 3.0
        
        # Deviatoric stress
        s_dev_xx = s_xx - p
        s_dev_yy = s_yy - p
        s_dev_zz = s_zz - p
        
        # von Mises stress
        vm = np.sqrt(0.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2 + 
                            2*(s_xy**2 + s_xz**2 + s_yz**2)))
        
        return vm, p, s_zz
    else:
        return None, None, None

def analyze_bottom_stress(filepath, case_name):
    """Analyze stress at bottom of MPM block."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {case_name}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    data = read_vtp_file(filepath)
    
    positions = data['positions']
    material_label = data.get('materialLabel', np.ones(len(positions)))
    
    # Get stress tensor - check both possible names
    stress_tensor = data.get('stress_tensor', data.get('stress', None))
    
    if stress_tensor is None:
        print("ERROR: No stress data found in VTP file!")
        return None
    
    # Separate MPM (label=1) and XPBD (label=2) particles
    mpm_mask = material_label == 1
    xpbd_mask = material_label == 2
    
    mpm_pos = positions[mpm_mask]
    xpbd_pos = positions[xpbd_mask]
    
    # Check if stress data is pre-computed
    if 'von_mises_stress' in data and 'mean_stress' in data:
        vm_stress_all = data['von_mises_stress']
        pressure_all = data['mean_stress']
        vm_stress = vm_stress_all[mpm_mask]
        pressure = pressure_all[mpm_mask]
        
        # Get sigma_zz from stress tensor
        mpm_stress = stress_tensor[mpm_mask]
        _, _, sigma_zz = compute_stress_magnitude(mpm_stress)
    else:
        # Compute from stress tensor
        mpm_stress = stress_tensor[mpm_mask]
        vm_stress, pressure, sigma_zz = compute_stress_magnitude(mpm_stress)
    
    print(f"\nParticle counts:")
    print(f"  MPM particles:  {np.sum(mpm_mask)}")
    print(f"  XPBD particles: {np.sum(xpbd_mask)}")
    
    # Find bottom region of MPM block (lowest 20% in z)
    z_min = mpm_pos[:, 2].min()
    z_max = mpm_pos[:, 2].max()
    z_range = z_max - z_min
    
    bottom_threshold = z_min + 0.2 * z_range
    bottom_mask = mpm_pos[:, 2] < bottom_threshold
    
    print(f"\nDomain bounds:")
    print(f"  Z range: [{z_min:.3f}, {z_max:.3f}] m")
    print(f"  Bottom 20% threshold: z < {bottom_threshold:.3f} m")
    print(f"  Particles in bottom 20%: {np.sum(bottom_mask)}")
    
    # Statistics for bottom region
    if np.sum(bottom_mask) > 0:
        bottom_vm = vm_stress[bottom_mask]
        bottom_pressure = pressure[bottom_mask]
        bottom_sigma_zz = sigma_zz[bottom_mask]
        
        print(f"\nBottom region stress (compression = negative):")
        print(f"  sigma_zz (vertical stress):")
        print(f"    Mean:   {np.mean(bottom_sigma_zz)/1e3:.2f} kPa")
        print(f"    Median: {np.median(bottom_sigma_zz)/1e3:.2f} kPa")
        print(f"    Std:    {np.std(bottom_sigma_zz)/1e3:.2f} kPa")
        print(f"    Min:    {np.min(bottom_sigma_zz)/1e3:.2f} kPa")
        print(f"    Max:    {np.max(bottom_sigma_zz)/1e3:.2f} kPa")
        
        print(f"  Mean stress (pressure, compression = negative):")
        print(f"    Mean:   {np.mean(bottom_pressure)/1e3:.2f} kPa")
        print(f"    Median: {np.median(bottom_pressure)/1e3:.2f} kPa")
        print(f"    Std:    {np.std(bottom_pressure)/1e3:.2f} kPa")
        
        print(f"  Von Mises equivalent stress:")
        print(f"    Mean:   {np.mean(bottom_vm)/1e3:.2f} kPa")
        print(f"    Median: {np.median(bottom_vm)/1e3:.2f} kPa")
        print(f"    Std:    {np.std(bottom_vm)/1e3:.2f} kPa")
        
        # Estimate expected stress from weight
        if 'density' in data:
            density_mpm = data['density'][mpm_mask]
            volume_mpm = data.get('volume', np.ones(np.sum(mpm_mask)))
            if isinstance(volume_mpm, np.ndarray) and len(volume_mpm) > np.sum(mpm_mask):
                volume_mpm = volume_mpm[mpm_mask]
            
            total_mass_mpm = np.sum(density_mpm * volume_mpm)
            
            # XPBD contribution (if they're resting on top)
            if 'density' in data and np.sum(xpbd_mask) > 0:
                density_xpbd = data['density'][xpbd_mask]
                volume_xpbd = data.get('volume', np.ones(np.sum(xpbd_mask)))
                if isinstance(volume_xpbd, np.ndarray) and len(volume_xpbd) > np.sum(xpbd_mask):
                    volume_xpbd = volume_xpbd[xpbd_mask]
                total_mass_xpbd = np.sum(density_xpbd * volume_xpbd)
            else:
                total_mass_xpbd = 0.0
            
            # Estimate contact area (assume roughly cubic block)
            x_extent = mpm_pos[:, 0].max() - mpm_pos[:, 0].min()
            y_extent = mpm_pos[:, 1].max() - mpm_pos[:, 1].min()
            contact_area = x_extent * y_extent
            
            # Expected bottom stress from total weight
            g = 9.81  # m/s²
            expected_stress = -(total_mass_mpm + total_mass_xpbd) * g / contact_area
            
            print(f"\nWeight analysis:")
            print(f"  MPM block mass:  {total_mass_mpm:.2f} kg")
            print(f"  XPBD mass:       {total_mass_xpbd:.2f} kg")
            print(f"  Total mass:      {total_mass_mpm + total_mass_xpbd:.2f} kg")
            print(f"  Contact area:    {contact_area:.4f} m²")
            print(f"  Expected σ_zz:   {expected_stress/1e3:.2f} kPa (from weight)")
            print(f"  Simulated σ_zz:  {np.mean(bottom_sigma_zz)/1e3:.2f} kPa")
            print(f"  Difference:      {(np.mean(bottom_sigma_zz) - expected_stress)/1e3:.2f} kPa")
            print(f"  Relative error:  {100*(np.mean(bottom_sigma_zz) - expected_stress)/expected_stress:.1f}%")
        
        return {
            'case': case_name,
            'bottom_sigma_zz_mean': np.mean(bottom_sigma_zz),
            'bottom_sigma_zz_median': np.median(bottom_sigma_zz),
            'bottom_sigma_zz_std': np.std(bottom_sigma_zz),
            'bottom_pressure_mean': np.mean(bottom_pressure),
            'bottom_vm_mean': np.mean(bottom_vm),
            'mpm_count': np.sum(mpm_mask),
            'xpbd_count': np.sum(xpbd_mask),
            'z_profile': mpm_pos[:, 2],
            'sigma_zz_profile': sigma_zz,
            'pressure_profile': pressure
        }
    else:
        print("WARNING: No particles found in bottom region!")
        return None

if __name__ == "__main__":
    # File paths
    solid_file = "./benchmarks/spatialBenchmark/output_coupling_test_solid/sim_step_0000010000_particles.vtp"
    light_file = "./benchmarks/spatialBenchmark/output_coupling_test_light/sim_step_0000010000_particles.vtp"
    
    # Analyze both cases
    solid_results = analyze_bottom_stress(solid_file, "SOLID (normal density XPBD)")
    light_results = analyze_bottom_stress(light_file, "LIGHT (10x lighter XPBD)")
    
    # Comparison
    if solid_results and light_results:
        print(f"\n{'='*60}")
        print("COMPARISON: Solid vs Light XPBD particles")
        print(f"{'='*60}")
        
        sigma_zz_diff = light_results['bottom_sigma_zz_mean'] - solid_results['bottom_sigma_zz_mean']
        sigma_zz_pct = 100 * sigma_zz_diff / solid_results['bottom_sigma_zz_mean']
        
        print(f"\nBottom stress difference (LIGHT - SOLID):")
        print(f"  Delta_sigma_zz:  {sigma_zz_diff/1e3:.2f} kPa")
        print(f"  Relative change: {sigma_zz_pct:.1f}%")
        
        if abs(sigma_zz_pct) < 5:
            print(f"\nSIMILAR: Stresses differ by less than 5% - coupling working correctly!")
        elif abs(sigma_zz_pct) < 15:
            print(f"\nMODERATE DIFFERENCE: {abs(sigma_zz_pct):.1f}% difference detected")
            print(f"   Expected: Lighter XPBD should reduce bottom stress")
        else:
            print(f"\nLARGE DIFFERENCE: {abs(sigma_zz_pct):.1f}% difference!")
            print(f"   This suggests XPBD weight may not be properly transferred to MPM")
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Stress vs depth profiles
        ax = axes[0]
        for results, label, color in [(solid_results, 'Solid', 'blue'), 
                                       (light_results, 'Light (10x less mass)', 'red')]:
            z = results['z_profile']
            sigma_zz = results['sigma_zz_profile'] / 1e3  # Convert to kPa
            
            # Sort by depth for cleaner plot
            sort_idx = np.argsort(z)
            z_sorted = z[sort_idx]
            sigma_sorted = sigma_zz[sort_idx]
            
            ax.scatter(sigma_sorted, z_sorted, alpha=0.3, s=1, label=label, color=color)
        
        ax.set_xlabel('Vertical Stress sigma_zz (kPa, compression = negative)', fontsize=12)
        ax.set_ylabel('Depth from Top (m)', fontsize=12)
        ax.set_title('Stress Profile Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
        
        # Plot 2: Bottom region statistics
        ax = axes[1]
        cases = ['Solid', 'Light (10x)']
        means = [solid_results['bottom_sigma_zz_mean']/1e3, 
                 light_results['bottom_sigma_zz_mean']/1e3]
        stds = [solid_results['bottom_sigma_zz_std']/1e3,
                light_results['bottom_sigma_zz_std']/1e3]
        
        x_pos = np.arange(len(cases))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                      color=['blue', 'red'], edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cases, fontsize=12)
        ax.set_ylabel('Bottom sigma_zz (kPa)', fontsize=12)
        ax.set_title('Bottom Stress Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height - std - 5,
                    f'{mean:.1f} +/- {std:.1f}',
                    ha='center', va='top', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./benchmarks/spatialBenchmark/coupling_stress_comparison.png', dpi=150)
        print(f"\nPlot saved: ./benchmarks/spatialBenchmark/coupling_stress_comparison.png")
        plt.show()


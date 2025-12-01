"""
Analytical Validation for Elastic Block Compression

Compares MPM simulation results to analytical elastic solutions for:
1. Geostatic stress distribution (sigma_zz = -rho*g*z)
2. Elastic compression/settlement under self-weight
3. Stress-strain relationships

Usage:
    python analytical_validation.py --E 1e6 --nu 0.2 --density 5000 --height 223
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analytical_geostatic_stress(z, z_top, density, g=9.81):
    """
    Analytical vertical stress for elastic column under self-weight.
    
    sigma_zz(z) = -rho * g * (z_top - z)
    (negative = compression)
    
    **IMPORTANT LIMITATION**: This solution assumes:
    - 1D column (infinite lateral extent)
    - Confined lateral boundaries (K0 condition: no lateral strain)
    - Homogeneous material
    
    For finite-width blocks, 3D effects cause:
    - Stress concentration at edges/corners
    - Non-uniform stress distribution across width
    - Lower average stress than 1D solution (especially near free surfaces)
    
    Parameters
    ----------
    z : array
        Vertical coordinates
    z_top : float
        Top surface elevation
    density : float
        Material density (kg/m³)
    g : float
        Gravity magnitude (m/s²)
    
    Returns
    -------
    sigma_zz : array
        Vertical stress (negative = compression)
        This is the CENTERLINE stress for an infinite column
    """
    depth = z_top - z
    sigma_zz = -density * g * depth
    return sigma_zz

def analytical_elastic_settlement(z, z_top, z_bottom, density, g, E, nu):
    """
    Analytical vertical settlement for elastic column under self-weight.
    
    For confined compression (K0 condition):
    strain_zz = sigma_zz / E_confined
    where E_confined = E*(1-nu)/((1+nu)*(1-2*nu))
    
    u_z(z) = integral of strain from z to z_top
    
    Parameters
    ----------
    z : array
        Vertical coordinates
    z_top : float
        Top surface elevation
    z_bottom : float
        Bottom surface elevation
    density : float
        Density (kg/m³)
    g : float
        Gravity (m/s²)
    E : float
        Young's modulus (Pa)
    nu : float
        Poisson's ratio
    
    Returns
    -------
    settlement : array
        Vertical displacement (negative = downward)
    """
    # Confined modulus (oedometric modulus)
    E_oed = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
    
    # Depth below top surface
    depth = z_top - z
    
    # Analytical settlement: u = -rho*g/(2*E_oed) * (H^2 - d^2)
    # where H = total height, d = depth from top
    H = z_top - z_bottom
    settlement = -density * g / (2 * E_oed) * (H**2 - depth**2)
    
    return settlement

def analytical_strain_at_depth(depth, density, g, E, nu):
    """Analytical vertical strain at given depth."""
    E_oed = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
    sigma_zz = -density * g * depth
    strain_zz = sigma_zz / E_oed
    return strain_zz

def load_block_geometry(hdf5_file):
    """Load block geometry from HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        x = np.array(f['x']).T
    
    print(f"Block Geometry:")
    print(f"  Particles: {x.shape[0]}")
    print(f"  X: [{x[:,0].min():.2f}, {x[:,0].max():.2f}] m")
    print(f"  Y: [{x[:,1].min():.2f}, {x[:,1].max():.2f}] m")
    print(f"  Z: [{x[:,2].min():.2f}, {x[:,2].max():.2f}] m")
    print(f"  Dimensions: {x[:,0].max()-x[:,0].min():.2f} × {x[:,1].max()-x[:,1].min():.2f} × {x[:,2].max()-x[:,2].min():.2f} m")
    
    return x

def load_simulation_results(output_folder, step):
    """Load VTP file from simulation."""
    import vtk
    from vtk.util import numpy_support
    
    vtp_file = f"{output_folder}/sim_step_0000{step:06d}_particles.vtp"
    
    if not Path(vtp_file).exists():
        raise FileNotFoundError(f"File not found: {vtp_file}")
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    polydata = reader.GetOutput()
    
    # Get positions
    points = polydata.GetPoints()
    positions = numpy_support.vtk_to_numpy(points.GetData())
    
    # Get stress
    point_data = polydata.GetPointData()
    
    result = {'positions': positions}
    
    # Try to get various fields
    field_names = ['mean_stress', 'von_mises', 'damage', 'ys', 'stress_tensor']
    for field in field_names:
        if point_data.HasArray(field):
            result[field] = numpy_support.vtk_to_numpy(point_data.GetArray(field))
    
    return result

def compare_stress_profiles(sim_data, analytical_params, output_path=None):
    """Compare simulation stress to analytical solution."""
    
    positions = sim_data['positions']
    z_coords = positions[:, 2]
    
    # Analytical solution
    z_top = analytical_params['z_top']
    density = analytical_params['density']
    g = analytical_params['g']
    E = analytical_params['E']
    nu = analytical_params['nu']
    K0 = analytical_params.get('K0', 0.5)  # Default K0 = 0.5 if not specified
    
    sigma_analytical = analytical_geostatic_stress(z_coords, z_top, density, g)
    
    # Compute expected vs actual stress at bottom
    z_bottom = z_coords.min()
    height = z_top - z_bottom
    sigma_max_analytical = density * g * height
    
    # Check block aspect ratio for 3D effects
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    width_x = x_coords.max() - x_coords.min()
    width_y = y_coords.max() - y_coords.min()
    aspect_ratio = height / min(width_x, width_y)
    
    print(f"\nBlock Analysis:")
    print(f"  Height: {height:.2f} m")
    print(f"  Width X: {width_x:.2f} m")
    print(f"  Width Y: {width_y:.2f} m")
    print(f"  Aspect ratio H/W: {aspect_ratio:.2f}")
    print(f"  Z range: [{z_bottom:.2f}, {z_top:.2f}] m")
    print(f"  Expected max stress (1D): {sigma_max_analytical/1e3:.2f} kPa")
    
    # Estimate 3D correction factor
    if aspect_ratio < 3:
        print(f"\n  ⚠️ WARNING: Low aspect ratio (H/W = {aspect_ratio:.2f})")
        print(f"     1D analytical solution may overestimate stress by 20-40%")
        print(f"     3D effects significant for H/W < 3")
        print(f"     Stress varies across block width (not uniform)")
    elif aspect_ratio < 5:
        print(f"\n  ℹ️ Moderate aspect ratio (H/W = {aspect_ratio:.2f})")
        print(f"     1D solution reasonable but 3D effects present (~10-20% error)")
    else:
        print(f"\n  ✓ High aspect ratio (H/W = {aspect_ratio:.2f})")
        print(f"     1D analytical solution valid (3D effects < 10%)")
    
    # Estimate 3D stress reduction factor (approximate)
    # Based on Saint-Venant's principle and aspect ratio
    if aspect_ratio < 1:
        stress_reduction = 0.6  # Very wide block
    elif aspect_ratio < 2:
        stress_reduction = 0.7  # Wide block
    elif aspect_ratio < 3:
        stress_reduction = 0.8  # Moderate block
    else:
        stress_reduction = 0.9  # Tall column (1D approx valid)
    
    print(f"  Estimated 3D stress reduction: {(1-stress_reduction)*100:.0f}%")
    print(f"  Expected max stress (3D corrected): {sigma_max_analytical*stress_reduction/1e3:.2f} kPa")
    
    # Check for plasticity indicators
    if 'von_mises' in sim_data:
        von_mises_mean = np.mean(sim_data['von_mises'])
        print(f"  Mean von Mises stress: {von_mises_mean/1e3:.2f} kPa")
    
    if 'damage' in sim_data:
        damage_max = np.max(sim_data['damage'])
        damage_mean = np.mean(sim_data['damage'])
        print(f"  Damage: max={damage_max:.4f}, mean={damage_mean:.4f}")
        if damage_max > 0.01:
            print(f"  ⚠️ WARNING: Damage detected! Simulation is NOT purely elastic")
    
    if 'ys' in sim_data:
        ys_mean = np.mean(sim_data['ys'])
        print(f"  Mean yield stress: {ys_mean/1e6:.2e} MPa")
        if ys_mean < 10 * sigma_max_analytical:
            print(f"  ⚠️ WARNING: Yield stress too low! Plasticity may be active")
            print(f"     Recommended: ys > {10*sigma_max_analytical/1e6:.2e} MPa")
    
    # Check stress at top and bottom
    print(f"\nStress Distribution Check:")
    
    # Top particles (should have minimal stress)
    top_mask = z_coords > (z_top - 5.0)  # Top 5m
    # Bottom particles (should have maximum stress)
    bottom_mask = z_coords < (z_coords.min() + 5.0)  # Bottom 5m
    
    # Plot comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # Stress vs depth - All particles
    ax = axes[0]
    depth = z_top - z_coords
    
    # Prefer stress_tensor[zz] if available, fall back to mean_stress
    stress_field_name = None
    if 'stress_tensor' in sim_data:
        stress_tensor = sim_data['stress_tensor']  # (N, 6): [xx, yy, zz, xy, xz, yz]
        sigma_sim = stress_tensor[:, 2]  # zz component
        stress_field_name = 'σ_zz'
        print(f"  Using stress tensor σ_zz component for validation")
        
        # Check top stress (should be near zero)
        sigma_top = sigma_sim[top_mask].mean()
        sigma_bottom = sigma_sim[bottom_mask].mean()
        depth_bottom = (z_top - z_coords[bottom_mask]).mean()
        
        print(f"  Top region (z > {z_top - 5:.1f}m):")
        print(f"    Mean σ_zz: {sigma_top/1e3:.2f} kPa (should be ≈0)")
        if abs(sigma_top) > sigma_max_analytical * 0.05:  # More than 5% of max
            print(f"    ⚠️ WARNING: Top stress unexpectedly high!")
        
        # Calculate expected strain at top (should be near zero)
        E_oed = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
        expected_strain_top = 0.0  # No load at top
        actual_strain_top = sigma_top / E_oed if abs(E_oed) > 1e-6 else 0
        
        print(f"    Expected strain: {expected_strain_top*100:.4f}%")
        print(f"    Actual strain (from stress): {actual_strain_top*100:.4f}%")
        if abs(actual_strain_top) > 0.001:  # More than 0.1%
            print(f"    ⚠️ WARNING: Top strain > 0.1% - check initial conditions!")
        
        print(f"  Bottom region (z < {z_coords.min() + 5:.1f}m):")
        print(f"    Mean σ_zz: {sigma_bottom/1e3:.2f} kPa")
        print(f"    Expected: {-density*g*depth_bottom/1e3:.2f} kPa")
        print(f"    Ratio: {sigma_bottom/(-density*g*depth_bottom):.3f}")
        
        # Check lateral stresses if full tensor available
        sigma_xx = stress_tensor[:, 0]
        sigma_yy = stress_tensor[:, 1]
        
        # K0 check at bottom
        sigma_xx_bottom = sigma_xx[bottom_mask].mean()
        sigma_yy_bottom = sigma_yy[bottom_mask].mean()
        K0_sim_x = sigma_xx_bottom / sigma_bottom if abs(sigma_bottom) > 1e-6 else 0
        K0_sim_y = sigma_yy_bottom / sigma_bottom if abs(sigma_bottom) > 1e-6 else 0
        
        print(f"\n  Lateral Earth Pressure (bottom):")
        print(f"    σ_xx: {sigma_xx_bottom/1e3:.2f} kPa")
        print(f"    σ_yy: {sigma_yy_bottom/1e3:.2f} kPa")
        print(f"    K0 (σ_xx/σ_zz): {K0_sim_x:.3f} (expected: {K0:.2f})")
        print(f"    K0 (σ_yy/σ_zz): {K0_sim_y:.3f} (expected: {K0:.2f})")
        
        if abs(K0_sim_x - K0) > 0.1 or abs(K0_sim_y - K0) > 0.1:
            print(f"    ⚠️ WARNING: K0 deviation > 10%!")
            print(f"       Check boundary conditions and convergence")

    elif 'mean_stress' in sim_data:
        sigma_sim = sim_data['mean_stress']
        stress_field_name = 'mean_stress'
        print(f"  ⚠️ WARNING: Using mean_stress instead of σ_zz")
        print(f"     mean_stress = (σ_xx+σ_yy+σ_zz)/3 ≠ σ_zz")
        print(f"     For K0={K0}: mean_stress ≈ {(2*K0+1)/3:.3f}·σ_zz")
        print(f"     This will give ~{(2*K0+1)/3*100:.0f}% of expected value!")
    else:
        print("  ❌ ERROR: No stress data available!")
        return
        
    # Convert to compression positive for comparison
    ax.scatter(-sigma_sim/1e3, depth, s=1, alpha=0.3, label=f'All particles ({stress_field_name})', c='blue')
    
    # Also plot centerline particles only
    x_center = (x_coords.max() + x_coords.min()) / 2
    y_center = (y_coords.max() + y_coords.min()) / 2
    tolerance = min(width_x, width_y) * 0.1  # 10% of width
    
    centerline_mask = (np.abs(x_coords - x_center) < tolerance) & (np.abs(y_coords - y_center) < tolerance)
    if np.sum(centerline_mask) > 10:
        sigma_centerline = sigma_sim[centerline_mask]
        depth_centerline = depth[centerline_mask]
        ax.scatter(-sigma_centerline/1e3, depth_centerline, s=10, alpha=0.8, 
                  label='Centerline ±10%', c='green', marker='+')
        print(f"  Centerline particles: {np.sum(centerline_mask)}")
    
    # Check actual stress range
    sigma_sim_max = np.max(-sigma_sim)
    sigma_sim_min = np.min(-sigma_sim)
    print(f"  Actual stress range (sim): {sigma_sim_min/1e3:.2f} to {sigma_sim_max/1e3:.2f} kPa")
    
    # Analytical line (1D solution)
    depth_line = np.linspace(0, depth.max(), 100)
    sigma_line = density * g * depth_line
    ax.plot(sigma_line/1e3, depth_line, 'r--', linewidth=2, label='Analytical (1D)')
    # Analytical with 3D correction
    ax.plot(sigma_line * stress_reduction / 1e3, depth_line, 'orange', 
           linewidth=2, label=f'Analytical (3D est.)')
    
    # Update label based on what we're actually plotting
    if stress_field_name == 'σ_zz':
        ax.set_xlabel('Vertical Stress σ_zz (kPa, compression positive)')
    else:
        ax.set_xlabel('Mean Stress (kPa, compression positive)')
    ax.set_ylabel('Depth from Top (m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{stress_field_name} vs Depth')
    ax.invert_yaxis()
    
    # Spatial stress distribution at mid-height
    ax = axes[1]
    if sigma_sim is not None:
        # Select particles near mid-height
        z_mid = (z_top + z_bottom) / 2
        tolerance_z = height * 0.05  # ±5% of height
        mid_mask = np.abs(z_coords - z_mid) < tolerance_z
        
        if np.sum(mid_mask) > 0:
            x_mid = x_coords[mid_mask]
            y_mid = y_coords[mid_mask]
            sigma_mid = -sigma_sim[mid_mask] / 1e3
            
            # Plot stress distribution across width
            scatter = ax.scatter(x_mid, y_mid, c=sigma_mid, s=20, cmap='viridis')
            colorbar_label = f'{stress_field_name} (kPa)' if stress_field_name else 'Stress (kPa)'
            plt.colorbar(scatter, ax=ax, label=colorbar_label)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{stress_field_name} at Mid-Height (z≈{z_mid:.1f}m)')
            ax.set_aspect('equal')
            
            print(f"  Mid-height stress variation: {sigma_mid.min():.2f} to {sigma_mid.max():.2f} kPa")
            print(f"  Coefficient of variation: {np.std(sigma_mid)/np.mean(sigma_mid)*100:.1f}%")
    
    # Strain profile vs depth
    ax = axes[2]
    if 'stress_tensor' in sim_data:
        # Calculate actual strain from stress
        E_oed = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
        strain_sim = sigma_sim / E_oed  # Elastic strain
        
        # Analytical strain profile
        depth_analytical = z_top - z_coords
        sigma_analytical_profile = -density * g * depth_analytical
        strain_analytical = sigma_analytical_profile / E_oed
        
        # Plot actual vs analytical strain
        ax.scatter(strain_sim * 100, depth, s=1, alpha=0.3, label='Simulation', c='blue')
        
        # Analytical line
        depth_line = np.linspace(0, depth.max(), 100)
        sigma_line_analytical = -density * g * depth_line
        strain_line_analytical = sigma_line_analytical / E_oed
        ax.plot(strain_line_analytical * 100, depth_line, 'r--', linewidth=2, label='Analytical')
        
        ax.set_xlabel('Vertical Strain ε_zz (%)')
        ax.set_ylabel('Depth from Top (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('Strain Profile vs Depth')
        ax.invert_yaxis()
        
        # Print strain statistics
        print(f"\nStrain Analysis:")
        print(f"  Top strain (sim): {strain_sim[top_mask].mean()*100:.4f}% (should be ≈0%)")
        print(f"  Bottom strain (sim): {strain_sim[bottom_mask].mean()*100:.4f}%")
        print(f"  Bottom strain (analytical): {strain_analytical[bottom_mask].mean()*100:.4f}%")
        print(f"  Max strain magnitude: {abs(strain_sim.min())*100:.4f}%")
        
    # Stress error
    ax = axes[3]
    if sigma_sim is not None:
        # Bin by depth for cleaner comparison
        n_bins = 20
        depth_bins = np.linspace(0, depth.max(), n_bins)
        bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
        
        sigma_sim_binned = []
        sigma_analytical_binned = []
        
        for i in range(len(depth_bins)-1):
            mask = (depth >= depth_bins[i]) & (depth < depth_bins[i+1])
            if np.any(mask):
                sigma_sim_binned.append(np.mean(-sigma_sim[mask]))  # Convert to compression positive
                sigma_analytical_binned.append(np.mean(-sigma_analytical[mask]))
        
        sigma_sim_binned = np.array(sigma_sim_binned)
        sigma_analytical_binned = np.array(sigma_analytical_binned)
        
        # Compute relative error
        rel_error = (sigma_sim_binned - sigma_analytical_binned) / (sigma_analytical_binned + 1e-10) * 100
        
        ax.plot(rel_error, bin_centers, 'o-', linewidth=2)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Depth from Top (m)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Stress Error')
        ax.invert_yaxis()
        
        # Print statistics
        print(f"\nStress Comparison:")
        print(f"  Mean error: {np.mean(rel_error):.2f}%")
        print(f"  RMS error: {np.sqrt(np.mean(rel_error**2)):.2f}%")
        print(f"  Max error: {np.max(np.abs(rel_error)):.2f}%")
        
        # Validation verdict
        mean_err = np.abs(np.mean(rel_error))
        rms_err = np.sqrt(np.mean(rel_error**2))
        
        if rms_err < 5:
            print(f"\n✅ VALIDATION PASSED: Excellent agreement (RMS < 5%)")
        elif rms_err < 10:
            print(f"\n✅ VALIDATION PASSED: Good agreement (RMS < 10%)")
        elif rms_err < 20:
            print(f"\n⚠️  VALIDATION MARGINAL: Acceptable agreement (RMS < 20%)")
        else:
            print(f"\n❌ VALIDATION FAILED: Poor agreement (RMS > 20%)")
            print(f"   Possible causes:")
            print(f"   - Simulation not converged (run longer)")
            print(f"   - Plasticity active (check ys, damage)")
            print(f"   - Wrong material parameters")
            print(f"   - Boundary effects")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved plot: {output_path}")
    
    plt.show()
    
    return fig

def generate_test_config(E, nu, density, height=None, g=9.81):
    """Generate optimal test configuration for elastic validation."""
    
    print("\n" + "="*60)
    print("ELASTIC VALIDATION TEST CONFIGURATION")
    print("="*60)
    
    # Material properties
    print(f"\nMaterial Properties:")
    print(f"  E = {E:.2e} Pa")
    print(f"  nu = {nu:.3f}")
    print(f"  density = {density:.1f} kg/m³")
    print(f"  gravity = {g:.2f} m/s²")
    
    # Compute derived properties
    K = E / (3 * (1 - 2*nu))  # Bulk modulus
    G = E / (2 * (1 + nu))     # Shear modulus
    lam = E * nu / ((1 + nu) * (1 - 2*nu))  # Lame's first parameter
    
    print(f"\nDerived Properties:")
    print(f"  Bulk modulus K = {K:.2e} Pa")
    print(f"  Shear modulus G = {G:.2e} Pa")
    print(f"  Lame lambda = {lam:.2e} Pa")
    
    # Characteristic time scale
    # For elastic waves: c_p = sqrt((K + 4G/3)/rho)
    c_p = np.sqrt((K + 4*G/3) / density)
    
    # Use provided height or default
    if height is None:
        H = 100.0  # Default assumption
        print(f"\n  Using default height H = {H}m")
    else:
        H = height
        print(f"\n  Using measured height H = {H:.2f}m")
    
    t_wave = H / c_p
    
    print(f"\nTime Scales:")
    print(f"  P-wave speed: {c_p:.1f} m/s")
    print(f"  Wave crossing time: {t_wave:.4f} s")
    print(f"  Recommended dt: {t_wave/100:.2e} s (1% of crossing time)")
    
    # Recommended settings
    dt_recommended = t_wave / 100
    
    # Expected maximum stress and strain
    sigma_max = density * g * H
    E_oed = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
    strain_max = sigma_max / E_oed
    settlement_max = strain_max * H / 2  # Approximate
    
    print(f"\nExpected Results (H={H:.1f}m):")
    print(f"  Oedometric modulus E_oed: {E_oed:.2e} Pa")
    print(f"  Max vertical stress: {sigma_max/1e3:.2f} kPa = {sigma_max/1e6:.4f} MPa")
    print(f"  Max vertical strain: {strain_max*100:.4f}%")
    print(f"  Approximate settlement: {settlement_max*1000:.2f} mm")
    
    print(f"\nRecommended Config Settings:")
    print(f"  \"dt\": {dt_recommended:.2e},")
    print(f"  \"E\": {E:.2e},")
    print(f"  \"nu\": {nu},")
    print(f"  \"density\": {density},")
    print(f"  \"ys\": 1e20,  // Very high (no plasticity)")
    print(f"  \"alpha\": 0.0,  // No pressure dependence")
    print(f"  \"hardening\": 0.0,")
    print(f"  \"softening\": 0.0,")
    print(f"  \"gravity\": -{g},")
    print(f"  \"boundFriction\": 0.0,  // Frictionless (elastic only)")
    print(f"  \"constitutive_model\": 1,  // Elastic")
    
    return {
        'dt': dt_recommended,
        'E': E,
        'nu': nu,
        'density': density,
        'c_p': c_p,
        'sigma_max': sigma_max,
        'strain_max': strain_max,
        'H': H
    }

def main():
    parser = argparse.ArgumentParser(description='Validate MPM elastic simulation against analytical solution')
    parser.add_argument('--E', type=float, default=1e9, help='Young\'s modulus (Pa)')
    parser.add_argument('--nu', type=float, default=0.2, help='Poisson\'s ratio')
    parser.add_argument('--density', type=float, default=5000, help='Density (kg/m³)')
    parser.add_argument('--g', type=float, default=9.81, help='Gravity (m/s²)')
    parser.add_argument('--output_folder', type=str, 
                       default='./benchmarks/generalBenchmark/outputElastic/',
                       help='Output folder with simulation results')
    parser.add_argument('--step', type=int, default=1000, help='Simulation step to analyze')
    parser.add_argument('--generate_config', action='store_true', 
                       help='Generate recommended configuration')
    
    args = parser.parse_args()
    
    # Load geometry first to get actual height
    block_file = './benchmarks/generalBenchmark/block_particles.h5'
    height = None
    z_top = None
    z_bottom = None
    
    if Path(block_file).exists():
        x = load_block_geometry(block_file)
        z_top = x[:, 2].max()
        z_bottom = x[:, 2].min()
        height = z_top - z_bottom
    else:
        print(f"Warning: Block file not found: {block_file}")
        print("Using default height assumptions")
    
    # Generate configuration with actual height
    config_params = generate_test_config(args.E, args.nu, args.density, height=height, g=args.g)
    
    if args.generate_config:
        return
    
    # Try to load simulation results
    try:
        print(f"\nLoading simulation results from step {args.step}...")
        sim_data = load_simulation_results(args.output_folder, args.step)
        
        if z_top is None:
            z_top = sim_data['positions'][:, 2].max()
        
        analytical_params = {
            'z_top': z_top,
            'density': args.density,
            'g': args.g,
            'E': args.E,
            'nu': args.nu
        }
        
        output_plot = f"{args.output_folder}/analytical_comparison_step{args.step}.png"
        compare_stress_profiles(sim_data, analytical_params, output_plot)
        
    except FileNotFoundError as e:
        print(f"\nSimulation results not found: {e}")
        print("Run simulation first, then use this script to validate.")

if __name__ == '__main__':
    main()

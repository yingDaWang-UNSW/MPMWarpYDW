#!/usr/bin/env python3
"""
Analysis script for Global Plane Strain Y tunnel simulation.
This simulation uses v_y = 0 EVERYWHERE (not just at boundaries) to enforce true plane strain.

Key hypothesis: With global v_y = 0, we should achieve:
  σ_yy = ν(σ_xx + σ_zz) = 0.25 × (-1.275 + -2.55) = -0.9575 MPa

Instead of the free-slip result where σ_yy ≈ σ_h = -1.28 MPa.
"""

import numpy as np
from pathlib import Path

# Try to import pyvista for VTP reading
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not available, will try VTK directly")

if not HAS_PYVISTA:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        HAS_VTK = True
    except ImportError:
        HAS_VTK = False
        print("Neither PyVista nor VTK available!")

# Problem parameters
TUNNEL_CENTER_X = 100.0  # m
TUNNEL_CENTER_Y = 20.0   # m (center of Y dimension for long tunnel)
TUNNEL_CENTER_Z = 100.0  # m
TUNNEL_RADIUS = 10.0     # m

# Material properties
E = 10e9           # Young's modulus (Pa)
NU = 0.25          # Poisson's ratio
RHO = 2600.0       # Density (kg/m³)
K0 = 0.5           # Horizontal stress ratio
Z_TOP = 200.0      # Top of domain (m)
G_ACCEL = 9.81     # Gravity (m/s²)

def read_vtp_file(filepath):
    """Read VTP file and extract particle data."""
    if HAS_PYVISTA:
        mesh = pv.read(filepath)
        points = np.array(mesh.points)
        
        # Get stress components - may be individual or as tensor
        stress_data = {}
        
        # Try individual stress components first
        for name in ['StressXX', 'StressYY', 'StressZZ', 'StressXY', 'StressXZ', 'StressYZ']:
            if name in mesh.point_data:
                stress_data[name] = np.array(mesh.point_data[name])
        
        # If not found, try stress_tensor (6 components: XX, YY, ZZ, XY, XZ, YZ or similar)
        if 'stress_tensor' in mesh.point_data and len(stress_data) == 0:
            tensor = np.array(mesh.point_data['stress_tensor'])
            print(f"  stress_tensor shape: {tensor.shape}")
            # Typically stored as [n, 6] with XX, YY, ZZ, XY, XZ, YZ
            # Or could be [n, 9] as full 3x3 matrix flattened
            if tensor.ndim == 2:
                if tensor.shape[1] == 6:
                    # Voigt notation: XX, YY, ZZ, XY, XZ, YZ  (or similar ordering)
                    stress_data['StressXX'] = tensor[:, 0]
                    stress_data['StressYY'] = tensor[:, 1]
                    stress_data['StressZZ'] = tensor[:, 2]
                    stress_data['StressXY'] = tensor[:, 3]
                    stress_data['StressXZ'] = tensor[:, 4]
                    stress_data['StressYZ'] = tensor[:, 5]
                elif tensor.shape[1] == 9:
                    # Full 3x3 matrix: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
                    stress_data['StressXX'] = tensor[:, 0]
                    stress_data['StressXY'] = tensor[:, 1]
                    stress_data['StressXZ'] = tensor[:, 2]
                    stress_data['StressYY'] = tensor[:, 4]
                    stress_data['StressYZ'] = tensor[:, 5]
                    stress_data['StressZZ'] = tensor[:, 8]
        
        return points, stress_data
    elif HAS_VTK:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(filepath))
        reader.Update()
        polydata = reader.GetOutput()
        
        points = vtk_to_numpy(polydata.GetPoints().GetData())
        
        stress_data = {}
        for name in ['StressXX', 'StressYY', 'StressZZ', 'StressXY', 'StressXZ', 'StressYZ']:
            arr = polydata.GetPointData().GetArray(name)
            if arr:
                stress_data[name] = vtk_to_numpy(arr)
        
        return points, stress_data
    else:
        raise RuntimeError("No VTK library available")


def kirsch_solution(r, theta, sigma_v, sigma_h, a):
    """
    Kirsch analytical solution for circular tunnel in infinite medium.
    
    Parameters:
    -----------
    r : float or array
        Radial distance from tunnel center
    theta : float or array  
        Angle from vertical (0 = crown, π/2 = springline)
    sigma_v : float
        Vertical far-field stress (negative = compression)
    sigma_h : float
        Horizontal far-field stress (negative = compression)
    a : float
        Tunnel radius
    
    Returns:
    --------
    sigma_rr, sigma_tt : Radial and tangential stresses
    """
    r_ratio = a / r
    r2 = r_ratio**2
    r4 = r_ratio**4
    
    cos2t = np.cos(2 * theta)
    
    # Mean and deviatoric far-field stress
    P = (sigma_v + sigma_h) / 2
    Q = (sigma_v - sigma_h) / 2
    
    # Kirsch solution
    sigma_rr = P * (1 - r2) + Q * (1 - 4*r2 + 3*r4) * cos2t
    sigma_tt = P * (1 + r2) - Q * (1 + 3*r4) * cos2t
    
    return sigma_rr, sigma_tt


def main():
    """Main analysis function."""
    
    # Find the latest output file
    output_dir = Path(r"D:\sourceCodes\MPMWarpYDW\benchmarks\tunnelBenchmark\output_elastic_planestrain_global")
    
    vtp_files = sorted(output_dir.glob("*_particles.vtp"))
    if not vtp_files:
        print(f"No VTP files found in {output_dir}")
        return
    
    latest_file = vtp_files[-1]
    print(f"Analyzing: {latest_file.name}")
    
    # Read data
    points, stress_data = read_vtp_file(latest_file)
    print(f"Loaded {len(points)} particles")
    
    # Check available stress components
    print(f"Available stress components: {list(stress_data.keys())}")
    
    if 'StressXX' not in stress_data:
        print("ERROR: Stress data not found in file!")
        return
    
    # Extract positions relative to tunnel center
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Calculate cylindrical coordinates (tunnel axis = Y)
    dx = x - TUNNEL_CENTER_X
    dz = z - TUNNEL_CENTER_Z
    r = np.sqrt(dx**2 + dz**2)
    theta = np.arctan2(dx, dz)  # 0 = crown (+Z direction), π/2 = springline (+X)
    
    # Extract stresses
    sigma_xx = stress_data['StressXX']
    sigma_yy = stress_data['StressYY']
    sigma_zz = stress_data['StressZZ']
    
    # Far-field stresses (from geostatic initialization)
    # At tunnel center depth (z = 100m, relative to z_top = 200m)
    depth = Z_TOP - TUNNEL_CENTER_Z
    sigma_v_expected = -RHO * G_ACCEL * depth  # Vertical stress
    sigma_h_expected = K0 * sigma_v_expected    # Horizontal stress
    
    print("\n" + "="*70)
    print("EXPECTED FAR-FIELD STRESSES (from geostatic initialization)")
    print("="*70)
    print(f"  Depth at tunnel center: {depth:.1f} m")
    print(f"  σ_v (vertical, σ_zz):   {sigma_v_expected/1e6:.4f} MPa")
    print(f"  σ_h (horizontal, σ_xx): {sigma_h_expected/1e6:.4f} MPa")
    
    # Expected plane strain σ_yy
    sigma_yy_planestrain = NU * (sigma_v_expected + sigma_h_expected)
    print(f"  σ_yy (plane strain):    {sigma_yy_planestrain/1e6:.4f} MPa")
    print(f"    = ν × (σ_xx + σ_zz) = {NU} × ({sigma_h_expected/1e6:.4f} + {sigma_v_expected/1e6:.4f})")
    
    # Select far-field particles (far from tunnel, middle Y slice)
    far_field_mask = (r > 5 * TUNNEL_RADIUS) & \
                     (np.abs(y - TUNNEL_CENTER_Y) < 5.0)  # Within 5m of center Y slice
    
    print("\n" + "="*70)
    print("MEASURED FAR-FIELD STRESSES (r > 5a, middle Y slice)")
    print("="*70)
    print(f"  Number of far-field particles: {np.sum(far_field_mask)}")
    
    sigma_xx_far = np.mean(sigma_xx[far_field_mask])
    sigma_yy_far = np.mean(sigma_yy[far_field_mask])
    sigma_zz_far = np.mean(sigma_zz[far_field_mask])
    
    print(f"  σ_xx = {sigma_xx_far/1e6:.4f} MPa (expected: {sigma_h_expected/1e6:.4f} MPa)")
    print(f"  σ_zz = {sigma_zz_far/1e6:.4f} MPa (expected: {sigma_v_expected/1e6:.4f} MPa)")
    print(f"  σ_yy = {sigma_yy_far/1e6:.4f} MPa")
    
    # Plane strain check
    print("\n" + "="*70)
    print("PLANE STRAIN CHECK: σ_yy = ν(σ_xx + σ_zz)")
    print("="*70)
    sigma_yy_expected = NU * (sigma_xx_far + sigma_zz_far)
    print(f"  Expected σ_yy: {sigma_yy_expected/1e6:.4f} MPa")
    print(f"  Actual σ_yy:   {sigma_yy_far/1e6:.4f} MPa")
    error_pct = abs(sigma_yy_far - sigma_yy_expected) / abs(sigma_yy_expected) * 100
    print(f"  Error:         {error_pct:.1f}%")
    
    if error_pct < 5:
        print("  ✅ PLANE STRAIN ACHIEVED! Error < 5%")
    elif error_pct < 10:
        print("  ⚠️ Plane strain approximately achieved. Error < 10%")
    else:
        print(f"  ❌ Plane strain NOT achieved. Error = {error_pct:.1f}%")
    
    # Compare with free-slip result
    sigma_yy_freeslip = sigma_h_expected  # Free-slip gives σ_yy ≈ σ_h
    improvement = abs(sigma_yy_far - sigma_yy_expected) < abs(sigma_yy_freeslip - sigma_yy_expected)
    print(f"\n  Free-slip σ_yy would be: {sigma_yy_freeslip/1e6:.4f} MPa")
    print(f"  Global plane strain gives: {sigma_yy_far/1e6:.4f} MPa")
    print(f"  Target (true plane strain): {sigma_yy_expected/1e6:.4f} MPa")
    if improvement:
        print("  ✅ Global plane strain is CLOSER to target than free-slip!")
    else:
        print("  ❌ No improvement over free-slip")
    
    # =========================================================================
    # WALL STRESS ANALYSIS - Kirsch comparison
    # =========================================================================
    print("\n" + "="*70)
    print("TUNNEL WALL STRESS COMPARISON (Kirsch Solution)")
    print("="*70)
    
    # Select particles near tunnel wall (middle Y slice)
    wall_mask = (r > TUNNEL_RADIUS * 0.95) & (r < TUNNEL_RADIUS * 1.5) & \
                (np.abs(y - TUNNEL_CENTER_Y) < 3.0)
    
    r_wall = r[wall_mask]
    theta_wall = theta[wall_mask]
    sigma_xx_wall = sigma_xx[wall_mask]
    sigma_zz_wall = sigma_zz[wall_mask]
    
    # Calculate radial and tangential stresses from simulation
    cos_t = np.cos(theta_wall)
    sin_t = np.sin(theta_wall)
    
    # Transform Cartesian to cylindrical
    # σ_rr = σ_xx sin²θ + σ_zz cos²θ + 2σ_xz sinθ cosθ
    # σ_θθ = σ_xx cos²θ + σ_zz sin²θ - 2σ_xz sinθ cosθ
    if 'StressXZ' in stress_data:
        sigma_xz_wall = stress_data['StressXZ'][wall_mask]
    else:
        sigma_xz_wall = np.zeros_like(sigma_xx_wall)
    
    sigma_rr_sim = sigma_xx_wall * sin_t**2 + sigma_zz_wall * cos_t**2 + 2*sigma_xz_wall * sin_t * cos_t
    sigma_tt_sim = sigma_xx_wall * cos_t**2 + sigma_zz_wall * sin_t**2 - 2*sigma_xz_wall * sin_t * cos_t
    
    # Kirsch solution
    sigma_rr_kirsch, sigma_tt_kirsch = kirsch_solution(
        r_wall, theta_wall, 
        sigma_zz_far, sigma_xx_far,  # Use measured far-field stresses
        TUNNEL_RADIUS
    )
    
    # Analyze by angular position
    print("\nWall stress by angular position (θ=0 is crown, θ=90° is springline):")
    print("-" * 70)
    
    angle_bins = np.linspace(-np.pi, np.pi, 9)
    angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    for i in range(len(angle_bins) - 1):
        bin_mask = (theta_wall >= angle_bins[i]) & (theta_wall < angle_bins[i+1])
        if np.sum(bin_mask) > 10:
            angle_deg = np.degrees(angle_centers[i])
            
            sim_rr = np.mean(sigma_rr_sim[bin_mask])
            sim_tt = np.mean(sigma_tt_sim[bin_mask])
            kirsch_rr = np.mean(sigma_rr_kirsch[bin_mask])
            kirsch_tt = np.mean(sigma_tt_kirsch[bin_mask])
            
            rr_err = abs(sim_rr - kirsch_rr) / abs(kirsch_rr) * 100 if kirsch_rr != 0 else 0
            tt_err = abs(sim_tt - kirsch_tt) / abs(kirsch_tt) * 100 if kirsch_tt != 0 else 0
            
            print(f"  θ={angle_deg:6.1f}°: σ_rr={sim_rr/1e6:7.3f} MPa (Kirsch: {kirsch_rr/1e6:7.3f}, err: {rr_err:5.1f}%)")
            print(f"             σ_θθ={sim_tt/1e6:7.3f} MPa (Kirsch: {kirsch_tt/1e6:7.3f}, err: {tt_err:5.1f}%)")
    
    # Key locations
    print("\n" + "="*70)
    print("KEY LOCATIONS (on tunnel wall at r = a)")
    print("="*70)
    
    # Crown (θ = 0) - should have max σ_θθ
    crown_mask = (np.abs(theta_wall) < np.radians(15))
    if np.sum(crown_mask) > 5:
        crown_tt_sim = np.mean(sigma_tt_sim[crown_mask])
        crown_tt_kirsch = np.mean(sigma_tt_kirsch[crown_mask])
        crown_err = abs(crown_tt_sim - crown_tt_kirsch) / abs(crown_tt_kirsch) * 100
        print(f"  CROWN (θ≈0°):      σ_θθ = {crown_tt_sim/1e6:.3f} MPa (Kirsch: {crown_tt_kirsch/1e6:.3f}, err: {crown_err:.1f}%)")
    
    # Springline (θ = ±90°) - should have min σ_θθ
    springline_mask = (np.abs(np.abs(theta_wall) - np.pi/2) < np.radians(15))
    if np.sum(springline_mask) > 5:
        spring_tt_sim = np.mean(sigma_tt_sim[springline_mask])
        spring_tt_kirsch = np.mean(sigma_tt_kirsch[springline_mask])
        spring_err = abs(spring_tt_sim - spring_tt_kirsch) / abs(spring_tt_kirsch) * 100
        print(f"  SPRINGLINE (θ≈90°): σ_θθ = {spring_tt_sim/1e6:.3f} MPa (Kirsch: {spring_tt_kirsch/1e6:.3f}, err: {spring_err:.1f}%)")
    
    # Expected Kirsch values at wall (r = a)
    print("\n  Expected from Kirsch solution at r = a:")
    print(f"    Crown (θ=0°):      σ_θθ = σ_v + σ_h - 2(σ_v - σ_h) = 3σ_h - σ_v")
    print(f"                           = 3×({sigma_xx_far/1e6:.3f}) - ({sigma_zz_far/1e6:.3f})")
    print(f"                           = {(3*sigma_xx_far - sigma_zz_far)/1e6:.3f} MPa")
    print(f"    Springline (θ=90°): σ_θθ = σ_v + σ_h + 2(σ_v - σ_h) = 3σ_v - σ_h")
    print(f"                           = 3×({sigma_zz_far/1e6:.3f}) - ({sigma_xx_far/1e6:.3f})")
    print(f"                           = {(3*sigma_zz_far - sigma_xx_far)/1e6:.3f} MPa")
    
    # Check stress concentration
    print("\n  Stress concentration analysis:")
    if crown_mask.sum() > 5 and springline_mask.sum() > 5:
        crown_tt = np.mean(sigma_tt_sim[crown_mask])
        spring_tt = np.mean(sigma_tt_sim[springline_mask])
        
        # For K0 < 1, crown should have LOWER σ_θθ than springline (more compressive at springline)
        # Wait - let me reconsider. With compression negative:
        # Crown: 3σ_h - σ_v = 3*(-1.28) - (-2.55) = -3.84 + 2.55 = -1.29 MPa
        # Springline: 3σ_v - σ_h = 3*(-2.55) - (-1.28) = -7.65 + 1.28 = -6.37 MPa
        # So springline should be MORE compressive (more negative)
        
        print(f"    Crown σ_θθ:      {crown_tt/1e6:.3f} MPa")
        print(f"    Springline σ_θθ: {spring_tt/1e6:.3f} MPa")
        
        if spring_tt < crown_tt:
            print("    ✅ CORRECT: Springline has higher compression (more negative σ_θθ)")
        else:
            print("    ❌ INCORRECT: Expected springline to have higher compression!")
            print("       This indicates INVERTED stress concentration pattern")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Plane strain error: {error_pct:.1f}%")
    if error_pct < 10:
        print("  → Global plane strain boundary condition is working!")
        print("  → σ_yy ≈ ν(σ_xx + σ_zz) as expected")
    else:
        print("  → Global plane strain may need more time to converge")
        print("  → Or there may be issues with the implementation")


if __name__ == "__main__":
    main()

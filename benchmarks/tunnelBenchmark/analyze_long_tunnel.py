"""
Analyze the long tunnel simulation results to verify plane strain behavior
Focus on central Y-slice (y=20.0m) where end effects should be minimal
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_long_tunnel():
    """Analyze the long tunnel central slice"""
    
    # Parameters
    output_dir = Path("./output_elastic_long")
    tunnel_center = np.array([100.0, 20.0, 100.0])  # Domain center
    tunnel_radius = 10.0  # m
    
    # Material properties
    E = 10e9  # Pa
    nu = 0.25
    
    # Geostatic stress at tunnel center (z=100m, depth = 200-100 = 100m)
    rho = 2600  # kg/m³
    g = 9.81
    depth = 100.0  # m (z_top=200, tunnel at z=100)
    sigma_v = -rho * g * depth  # -2.55 MPa (compressive = negative)
    K0 = 0.5
    sigma_h = K0 * sigma_v  # -1.275 MPa
    
    # Plane strain: sigma_yy = nu * (sigma_xx + sigma_zz) = nu * 2*p0 where p0 = average in-plane
    # At far field: sigma_xx = sigma_h, sigma_zz = sigma_v
    # sigma_yy_plane_strain = nu * (sigma_h + sigma_v) = 0.25 * (-1.275 + -2.55) = -0.956 MPa
    sigma_yy_plane_strain = nu * (sigma_h + sigma_v)
    
    print("="*60)
    print("LONG TUNNEL PLANE STRAIN VERIFICATION")
    print("="*60)
    print(f"Expected far-field stresses:")
    print(f"  σ_xx (horizontal): {sigma_h/1e6:.3f} MPa")
    print(f"  σ_zz (vertical):   {sigma_v/1e6:.3f} MPa")
    print(f"  σ_yy (plane strain): {sigma_yy_plane_strain/1e6:.3f} MPa")
    print()
    
    # Find latest VTP file
    vtp_files = sorted(output_dir.glob("*_particles.vtp"))
    if not vtp_files:
        print("No VTP files found!")
        return
    
    latest_vtp = vtp_files[-1]
    print(f"Analyzing: {latest_vtp.name}")
    
    # Load data
    mesh = pv.read(str(latest_vtp))
    positions = np.array(mesh.points)
    stress = np.array(mesh.point_data['stress_tensor'])  # (N, 6) Voigt notation
    
    # Extract stress components
    sigma_xx = stress[:, 0]
    sigma_yy = stress[:, 1] 
    sigma_zz = stress[:, 2]
    sigma_xy = stress[:, 3]
    sigma_yz = stress[:, 4]
    sigma_xz = stress[:, 5]
    
    # Central Y-slice: y = 20.0 ± 0.25m (one particle spacing)
    y_center = 20.0
    y_tolerance = 0.25  # Half particle spacing
    
    y_mask = np.abs(positions[:, 1] - y_center) < y_tolerance
    
    print(f"\nCentral Y-slice analysis (y = {y_center} ± {y_tolerance}m):")
    print(f"  Particles in slice: {np.sum(y_mask)}")
    
    if np.sum(y_mask) == 0:
        print("No particles in central slice!")
        # Find actual Y values
        unique_y = np.unique(np.round(positions[:, 1], 2))
        print(f"Available Y values: {unique_y[:20]}...")
        y_center_actual = unique_y[len(unique_y)//2]
        y_mask = np.abs(positions[:, 1] - y_center_actual) < y_tolerance
        print(f"Using y = {y_center_actual}, particles: {np.sum(y_mask)}")
    
    # Filter positions and stress to central slice
    pos_slice = positions[y_mask]
    stress_xx_slice = sigma_xx[y_mask]
    stress_yy_slice = sigma_yy[y_mask]
    stress_zz_slice = sigma_zz[y_mask]
    
    # Far-field region (r > 4a from tunnel center)
    x_rel = pos_slice[:, 0] - tunnel_center[0]
    z_rel = pos_slice[:, 2] - tunnel_center[2]
    r = np.sqrt(x_rel**2 + z_rel**2)
    
    far_field_mask = r > 4.0 * tunnel_radius
    
    print(f"\nFar-field region (r > 4a = {4*tunnel_radius}m):")
    print(f"  Particles: {np.sum(far_field_mask)}")
    
    if np.sum(far_field_mask) > 0:
        ff_sigma_xx = np.mean(stress_xx_slice[far_field_mask])
        ff_sigma_yy = np.mean(stress_yy_slice[far_field_mask])
        ff_sigma_zz = np.mean(stress_zz_slice[far_field_mask])
        
        print(f"\n  Mean σ_xx: {ff_sigma_xx/1e6:.4f} MPa (expected: {sigma_h/1e6:.4f} MPa)")
        print(f"  Mean σ_zz: {ff_sigma_zz/1e6:.4f} MPa (expected: {sigma_v/1e6:.4f} MPa)")
        print(f"  Mean σ_yy: {ff_sigma_yy/1e6:.4f} MPa")
        print()
        
        # Check plane strain condition
        sigma_yy_expected = nu * (ff_sigma_xx + ff_sigma_zz)
        print(f"  Plane strain check:")
        print(f"    σ_yy_expected = ν(σ_xx + σ_zz) = {sigma_yy_expected/1e6:.4f} MPa")
        print(f"    σ_yy_actual   = {ff_sigma_yy/1e6:.4f} MPa")
        print(f"    Difference:     {(ff_sigma_yy - sigma_yy_expected)/1e6:.4f} MPa")
        print(f"    Relative error: {100*abs(ff_sigma_yy - sigma_yy_expected)/abs(sigma_yy_expected):.1f}%")
        
        # Compare to thin domain
        sigma_yy_thin_approx = sigma_h  # What we observed in thin domain
        print(f"\n  Comparison to thin domain (Ly=4m):")
        print(f"    Thin domain σ_yy ≈ σ_h = {sigma_yy_thin_approx/1e6:.4f} MPa")
        print(f"    Long tunnel σ_yy =       {ff_sigma_yy/1e6:.4f} MPa")
        
        # Near-wall region for Kirsch comparison
        near_wall_mask = (r > 1.2 * tunnel_radius) & (r < 2.0 * tunnel_radius)
        
        if np.sum(near_wall_mask) > 0:
            # Calculate radial and tangential stresses
            theta = np.arctan2(z_rel, x_rel)  # Angle from horizontal
            
            # Stress transformation to radial/tangential
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # σ_rr = σ_xx*cos²θ + σ_zz*sin²θ + 2*σ_xz*sinθ*cosθ
            # For central slice, σ_xz should be small
            stress_xz_slice = stress[:, 5][y_mask]
            
            sigma_rr = (stress_xx_slice * cos_t**2 + stress_zz_slice * sin_t**2 + 
                        2*stress_xz_slice * sin_t * cos_t)
            sigma_tt = (stress_xx_slice * sin_t**2 + stress_zz_slice * cos_t**2 - 
                        2*stress_xz_slice * sin_t * cos_t)
            
            print(f"\n  Near-wall region (1.2a < r < 2.0a):")
            print(f"    Particles: {np.sum(near_wall_mask)}")
            print(f"    Mean σ_rr: {np.mean(sigma_rr[near_wall_mask])/1e6:.4f} MPa")
            print(f"    Mean σ_θθ: {np.mean(sigma_tt[near_wall_mask])/1e6:.4f} MPa")
    
    # Also check Y-variation of σ_yy
    print("\n" + "="*60)
    print("Y-VARIATION OF σ_yy (checking end effects)")
    print("="*60)
    
    # Sample at different Y positions
    y_positions = [5, 10, 15, 20, 25, 30, 35]  # m
    
    for y_pos in y_positions:
        y_mask_local = np.abs(positions[:, 1] - y_pos) < 0.25
        if np.sum(y_mask_local) == 0:
            continue
            
        pos_local = positions[y_mask_local]
        stress_yy_local = sigma_yy[y_mask_local]
        stress_xx_local = sigma_xx[y_mask_local]
        stress_zz_local = sigma_zz[y_mask_local]
        
        # Far-field at this Y
        x_rel = pos_local[:, 0] - tunnel_center[0]
        z_rel = pos_local[:, 2] - tunnel_center[2]
        r = np.sqrt(x_rel**2 + z_rel**2)
        ff_mask = r > 4.0 * tunnel_radius
        
        if np.sum(ff_mask) > 0:
            ff_yy = np.mean(stress_yy_local[ff_mask])
            ff_xx = np.mean(stress_xx_local[ff_mask])
            ff_zz = np.mean(stress_zz_local[ff_mask])
            expected_yy = nu * (ff_xx + ff_zz)
            error = 100*abs(ff_yy - expected_yy)/abs(expected_yy) if expected_yy != 0 else 0
            print(f"  Y={y_pos:5.1f}m: σ_yy = {ff_yy/1e6:7.4f} MPa, expected = {expected_yy/1e6:7.4f} MPa, error = {error:5.1f}%")

if __name__ == "__main__":
    analyze_long_tunnel()

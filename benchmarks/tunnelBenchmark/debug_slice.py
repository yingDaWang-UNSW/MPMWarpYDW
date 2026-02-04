"""
Debug script to verify Y-slice selection and investigate systematic offset.
"""
import numpy as np
import pyvista as pv
import glob

# Load the final VTP files
files = {
    'lores': sorted(glob.glob('./output_elastic/*_particles.vtp'))[-1],
    'hires': sorted(glob.glob('./output_elastic_hires/*_particles.vtp'))[-1],
    'vhires': sorted(glob.glob('./output_elastic_vhires/*_particles.vtp'))[-1]
}

tunnel_center = np.array([100.0, 2.0, 100.0])
tunnel_radius = 10.0

for name, f in files.items():
    mesh = pv.read(f)
    pos = np.array(mesh.points)
    stress = np.array(mesh.point_data['stress_tensor'])
    
    print(f'\n{"="*60}')
    print(f'{name.upper()}')
    print(f'{"="*60}')
    print(f'Total particles: {len(pos)}')
    
    # Y distribution
    y_vals = pos[:, 1]
    y_unique = np.unique(np.round(y_vals, 3))
    print(f'Unique Y values: {y_unique}')
    print(f'Y range: [{y_vals.min():.3f}, {y_vals.max():.3f}]')
    
    # Check central slice
    y_tol = {'lores': 0.6, 'hires': 0.3, 'vhires': 0.25}[name]
    y_mask = np.abs(y_vals - 2.0) < y_tol
    print(f'Y-slice tolerance={y_tol}: {np.sum(y_mask)} particles selected')
    print(f'Y values in slice: {np.unique(np.round(y_vals[y_mask], 3))}')
    
    # Get central slice data
    pos_slice = pos[y_mask]
    stress_slice = stress[y_mask]
    
    # Compute stress at θ=45° far-field (r > 4a)
    dx = pos_slice[:, 0] - tunnel_center[0]
    dz = pos_slice[:, 2] - tunnel_center[2]
    r = np.sqrt(dx**2 + dz**2)
    theta = np.arctan2(dx, dz)  # from vertical
    
    # Far-field particles (r > 4a)
    far_mask = r > 4 * tunnel_radius
    print(f'\nFar-field (r > 4a): {np.sum(far_mask)} particles')
    print(f'  Mean σ_xx: {np.mean(stress_slice[far_mask, 0])/1e6:.4f} MPa')
    print(f'  Mean σ_yy: {np.mean(stress_slice[far_mask, 1])/1e6:.4f} MPa')  
    print(f'  Mean σ_zz: {np.mean(stress_slice[far_mask, 2])/1e6:.4f} MPa')
    print(f'  Mean σ_xz: {np.mean(stress_slice[far_mask, 4])/1e6:.4f} MPa')
    
    # Expected far-field (depth = 100m)
    density = 2600.0
    gravity = 9.81
    K0 = 0.5
    depth = 100.0
    sigma_v_expected = -density * gravity * depth  # Compression negative
    sigma_h_expected = K0 * sigma_v_expected
    print(f'\n  Expected σ_zz (vertical): {sigma_v_expected/1e6:.4f} MPa')
    print(f'  Expected σ_xx (horizontal): {sigma_h_expected/1e6:.4f} MPa')
    
    # Check σ_yy (out-of-plane) - should be ν(σ_xx + σ_zz) for plane strain
    nu = 0.25
    sigma_yy_expected = nu * (sigma_h_expected + sigma_v_expected)
    print(f'  Expected σ_yy (plane strain): {sigma_yy_expected/1e6:.4f} MPa')
    print(f'  Actual σ_yy: {np.mean(stress_slice[far_mask, 1])/1e6:.4f} MPa')
    
    # Check at θ=45° specifically
    theta_target = np.radians(45)
    angle_tol = np.radians(10)
    theta_mask = np.abs(theta - theta_target) < angle_tol
    r_mask = (r > 1.2 * tunnel_radius) & (r < 2.0 * tunnel_radius)
    combined = theta_mask & r_mask
    
    if np.sum(combined) > 0:
        print(f'\nAt θ=45°, r=1.2-2.0a: {np.sum(combined)} particles')
        print(f'  Mean σ_xx: {np.mean(stress_slice[combined, 0])/1e6:.4f} MPa')
        print(f'  Mean σ_zz: {np.mean(stress_slice[combined, 2])/1e6:.4f} MPa')
        print(f'  Mean σ_xz: {np.mean(stress_slice[combined, 4])/1e6:.4f} MPa')
        
        # Transform to polar
        cos_t = np.cos(theta[combined])
        sin_t = np.sin(theta[combined])
        sigma_xx = stress_slice[combined, 0]
        sigma_zz = stress_slice[combined, 2]
        sigma_xz = stress_slice[combined, 4]
        
        sigma_rr = sigma_xx * sin_t**2 + sigma_zz * cos_t**2 + 2 * sigma_xz * sin_t * cos_t
        sigma_tt = sigma_xx * cos_t**2 + sigma_zz * sin_t**2 - 2 * sigma_xz * sin_t * cos_t
        
        print(f'\n  Transformed to polar:')
        print(f'  Mean σ_rr: {np.mean(sigma_rr)/1e6:.4f} MPa')
        print(f'  Mean σ_θθ: {np.mean(sigma_tt)/1e6:.4f} MPa')
        
        # Kirsch analytical at r=1.5a, θ=45°
        r_check = 1.5 * tunnel_radius
        a = tunnel_radius
        sigma_v = -density * gravity * depth
        sigma_h = K0 * sigma_v
        
        sigma_mean = (sigma_v + sigma_h) / 2
        sigma_dev = (sigma_v - sigma_h) / 2
        a2_r2 = (a / r_check) ** 2
        a4_r4 = a2_r2 ** 2
        cos2t = np.cos(2 * theta_target)
        
        ana_sigma_rr = sigma_mean * (1 - a2_r2) + sigma_dev * (1 - 4*a2_r2 + 3*a4_r4) * cos2t
        ana_sigma_tt = sigma_mean * (1 + a2_r2) - sigma_dev * (1 + 3*a4_r4) * cos2t
        
        print(f'\n  Kirsch analytical at r=1.5a, θ=45°:')
        print(f'  σ_rr: {-ana_sigma_rr/1e6:.4f} MPa')  # flip sign for comparison
        print(f'  σ_θθ: {-ana_sigma_tt/1e6:.4f} MPa')

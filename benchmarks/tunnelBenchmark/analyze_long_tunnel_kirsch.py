"""
Comprehensive analysis of long tunnel simulation vs Kirsch analytical solution.
Focuses on central Y-slice (y=20.0m) to minimize boundary effects.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

def kirsch_solution(r, theta, a, sigma_v, sigma_h):
    """
    Kirsch analytical solution for circular tunnel in infinite elastic medium.
    
    Parameters:
    -----------
    r : array-like
        Radial distance from tunnel center
    theta : array-like
        Angle from horizontal (x-axis), counter-clockwise positive
    a : float
        Tunnel radius
    sigma_v : float
        Vertical far-field stress (negative = compression)
    sigma_h : float
        Horizontal far-field stress (negative = compression)
    
    Returns:
    --------
    sigma_rr : array-like
        Radial stress
    sigma_tt : array-like
        Tangential (hoop) stress
    sigma_rt : array-like
        Shear stress
    """
    p0 = (sigma_v + sigma_h) / 2  # Mean stress
    q0 = (sigma_v - sigma_h) / 2  # Deviatoric stress
    
    ratio = a / r
    ratio2 = ratio ** 2
    ratio4 = ratio ** 4
    
    # Kirsch solution (note: theta measured from horizontal)
    # For vertical stress > horizontal, theta=0 is crown/invert direction
    sigma_rr = p0 * (1 - ratio2) + q0 * (1 - 4*ratio2 + 3*ratio4) * np.cos(2*theta)
    sigma_tt = p0 * (1 + ratio2) - q0 * (1 + 3*ratio4) * np.cos(2*theta)
    sigma_rt = -q0 * (1 + 2*ratio2 - 3*ratio4) * np.sin(2*theta)
    
    return sigma_rr, sigma_tt, sigma_rt


def analyze_and_plot():
    """Main analysis and plotting function."""
    
    # Parameters
    output_dir = Path("./output_elastic_long")
    tunnel_center = np.array([100.0, 20.0, 100.0])
    tunnel_radius = 10.0  # m
    
    # Material properties
    E = 10e9  # Pa
    nu = 0.25
    
    # Geostatic stress at tunnel center
    rho = 2600  # kg/m³
    g = 9.81
    depth = 100.0  # m
    sigma_v = -rho * g * depth  # -2.55 MPa
    K0 = 0.5
    sigma_h = K0 * sigma_v  # -1.275 MPa
    
    print("="*70)
    print("LONG TUNNEL vs KIRSCH ANALYTICAL SOLUTION")
    print("="*70)
    print(f"Tunnel radius: {tunnel_radius} m")
    print(f"Far-field σ_v: {sigma_v/1e6:.3f} MPa")
    print(f"Far-field σ_h: {sigma_h/1e6:.3f} MPa")
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
    stress = np.array(mesh.point_data['stress_tensor'])
    
    # Extract stress components
    sigma_xx = stress[:, 0]
    sigma_yy = stress[:, 1]
    sigma_zz = stress[:, 2]
    sigma_xy = stress[:, 3]
    sigma_yz = stress[:, 4]
    sigma_xz = stress[:, 5]
    
    # Central Y-slice
    y_center = 20.0
    y_tolerance = 0.25
    y_mask = np.abs(positions[:, 1] - y_center) < y_tolerance
    
    print(f"\nCentral Y-slice (y = {y_center} ± {y_tolerance}m):")
    print(f"  Particles: {np.sum(y_mask)}")
    
    # Extract slice data
    pos_slice = positions[y_mask]
    stress_xx_slice = sigma_xx[y_mask]
    stress_yy_slice = sigma_yy[y_mask]
    stress_zz_slice = sigma_zz[y_mask]
    stress_xz_slice = sigma_xz[y_mask]
    
    # Calculate polar coordinates relative to tunnel center
    x_rel = pos_slice[:, 0] - tunnel_center[0]
    z_rel = pos_slice[:, 2] - tunnel_center[2]
    r = np.sqrt(x_rel**2 + z_rel**2)
    theta = np.arctan2(z_rel, x_rel)  # Angle from +x axis (horizontal)
    
    # Transform Cartesian stresses to polar (radial/tangential)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_2t = np.cos(2*theta)
    sin_2t = np.sin(2*theta)
    
    # Stress transformation:
    # σ_rr = σ_xx*cos²θ + σ_zz*sin²θ + 2*σ_xz*sinθ*cosθ
    # σ_θθ = σ_xx*sin²θ + σ_zz*cos²θ - 2*σ_xz*sinθ*cosθ
    # σ_rθ = (σ_zz - σ_xx)*sinθ*cosθ + σ_xz*(cos²θ - sin²θ)
    sigma_rr_sim = stress_xx_slice * cos_t**2 + stress_zz_slice * sin_t**2 + 2*stress_xz_slice * sin_t * cos_t
    sigma_tt_sim = stress_xx_slice * sin_t**2 + stress_zz_slice * cos_t**2 - 2*stress_xz_slice * sin_t * cos_t
    sigma_rt_sim = (stress_zz_slice - stress_xx_slice) * sin_t * cos_t + stress_xz_slice * (cos_t**2 - sin_t**2)
    
    # Filter to exclude very near-wall particles (r > 1.2a) and far-field (r < 10a)
    r_min = 1.2 * tunnel_radius
    r_max = 8.0 * tunnel_radius
    valid_mask = (r >= r_min) & (r <= r_max)
    
    print(f"\nAnalysis region: {r_min/tunnel_radius:.1f}a < r < {r_max/tunnel_radius:.1f}a")
    print(f"  Valid particles: {np.sum(valid_mask)}")
    
    # Calculate analytical solution at simulation points
    sigma_rr_ana, sigma_tt_ana, sigma_rt_ana = kirsch_solution(
        r[valid_mask], theta[valid_mask], tunnel_radius, sigma_v, sigma_h
    )
    
    # Extract simulation values at valid points
    r_valid = r[valid_mask]
    theta_valid = theta[valid_mask]
    sigma_rr_sim_valid = sigma_rr_sim[valid_mask]
    sigma_tt_sim_valid = sigma_tt_sim[valid_mask]
    sigma_rt_sim_valid = sigma_rt_sim[valid_mask]
    
    # =========================================================================
    # PLOT 1: Stress vs r/a at different angles
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.25)
    
    # Define angle bins (in degrees)
    angle_bins = [
        (0, 'θ=0° (horizontal)', 'C0'),
        (45, 'θ=45°', 'C1'),
        (90, 'θ=90° (vertical)', 'C2'),
    ]
    
    # Analytical curves
    r_ana = np.linspace(tunnel_radius, r_max, 200)
    
    for idx, (angle_deg, label, color) in enumerate(angle_bins):
        angle_rad = np.radians(angle_deg)
        
        # Analytical
        sigma_rr_curve, sigma_tt_curve, _ = kirsch_solution(
            r_ana, angle_rad, tunnel_radius, sigma_v, sigma_h
        )
        
        # Filter simulation data near this angle
        angle_tol = np.radians(10)  # ±10° tolerance
        
        # Handle angle wrapping for θ=0°
        if angle_deg == 0:
            angle_mask = (np.abs(theta_valid) < angle_tol) | (np.abs(theta_valid - 2*np.pi) < angle_tol) | (np.abs(theta_valid + 2*np.pi) < angle_tol)
        else:
            angle_mask = np.abs(theta_valid - angle_rad) < angle_tol
            # Also check negative angle (symmetry)
            angle_mask |= np.abs(theta_valid + angle_rad) < angle_tol
        
        r_at_angle = r_valid[angle_mask]
        sigma_rr_at_angle = sigma_rr_sim_valid[angle_mask]
        sigma_tt_at_angle = sigma_tt_sim_valid[angle_mask]
        
        # σ_rr subplot
        ax1 = fig1.add_subplot(gs[0, 0]) if idx == 0 else fig1.axes[0]
        ax1.plot(r_ana/tunnel_radius, sigma_rr_curve/1e6, '-', color=color, linewidth=2, 
                 label=f'{label} (Kirsch)' if idx == 0 else None)
        ax1.scatter(r_at_angle/tunnel_radius, sigma_rr_at_angle/1e6, s=8, alpha=0.5, color=color,
                    label=f'{label} (Sim)' if idx == 0 else None)
        
        # σ_θθ subplot
        ax2 = fig1.add_subplot(gs[0, 1]) if idx == 0 else fig1.axes[1]
        ax2.plot(r_ana/tunnel_radius, sigma_tt_curve/1e6, '-', color=color, linewidth=2)
        ax2.scatter(r_at_angle/tunnel_radius, sigma_tt_at_angle/1e6, s=8, alpha=0.5, color=color)
    
    # Add legends and labels
    ax1 = fig1.axes[0]
    ax1.set_xlabel('r/a', fontsize=12)
    ax1.set_ylabel('σ_rr (MPa)', fontsize=12)
    ax1.set_title('Radial Stress vs Distance', fontsize=14)
    ax1.axhline(y=sigma_h/1e6, color='gray', linestyle='--', alpha=0.5, label='σ_h')
    ax1.axhline(y=sigma_v/1e6, color='gray', linestyle=':', alpha=0.5, label='σ_v')
    ax1.set_xlim([1, 8])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    ax2 = fig1.axes[1]
    ax2.set_xlabel('r/a', fontsize=12)
    ax2.set_ylabel('σ_θθ (MPa)', fontsize=12)
    ax2.set_title('Tangential (Hoop) Stress vs Distance', fontsize=14)
    ax2.axhline(y=sigma_h/1e6, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=sigma_v/1e6, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlim([1, 8])
    ax2.grid(True, alpha=0.3)
    
    # Legend for angles
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=2) for _, _, c in angle_bins]
    ax2.legend(custom_lines, [l for _, l, _ in angle_bins], loc='upper right', fontsize=9)
    
    # =========================================================================
    # PLOT 2: Error analysis
    # =========================================================================
    # Calculate errors
    error_rr = (sigma_rr_sim_valid - sigma_rr_ana) / np.abs(sigma_rr_ana) * 100
    error_tt = (sigma_tt_sim_valid - sigma_tt_ana) / np.abs(sigma_tt_ana) * 100
    
    # Clip extreme errors for plotting
    error_rr_clipped = np.clip(error_rr, -100, 100)
    error_tt_clipped = np.clip(error_tt, -100, 100)
    
    ax3 = fig1.add_subplot(gs[1, 0])
    scatter3 = ax3.scatter(r_valid/tunnel_radius, error_rr_clipped, c=np.degrees(theta_valid), 
                           cmap='twilight', s=5, alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('r/a', fontsize=12)
    ax3.set_ylabel('σ_rr Error (%)', fontsize=12)
    ax3.set_title('Radial Stress Error vs Distance', fontsize=14)
    ax3.set_xlim([1, 8])
    ax3.set_ylim([-50, 50])
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='θ (degrees)')
    
    ax4 = fig1.add_subplot(gs[1, 1])
    scatter4 = ax4.scatter(r_valid/tunnel_radius, error_tt_clipped, c=np.degrees(theta_valid), 
                           cmap='twilight', s=5, alpha=0.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5)
    ax4.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('r/a', fontsize=12)
    ax4.set_ylabel('σ_θθ Error (%)', fontsize=12)
    ax4.set_title('Tangential Stress Error vs Distance', fontsize=14)
    ax4.set_xlim([1, 8])
    ax4.set_ylim([-50, 50])
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='θ (degrees)')
    
    fig1.suptitle(f'Long Tunnel (Ly=40m) Central Slice Analysis\nSimulation vs Kirsch Solution', 
                  fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig('long_tunnel_kirsch_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: long_tunnel_kirsch_comparison.png")
    
    # =========================================================================
    # PLOT 3: Stress distribution around tunnel at specific r/a
    # =========================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    r_ratios = [1.5, 2.0, 3.0]  # r/a values to analyze
    
    for ax, r_ratio in zip(axes, r_ratios):
        r_target = r_ratio * tunnel_radius
        r_band = 0.3 * tunnel_radius  # ±0.3a band
        
        band_mask = np.abs(r_valid - r_target) < r_band
        
        theta_band = theta_valid[band_mask]
        sigma_rr_band = sigma_rr_sim_valid[band_mask]
        sigma_tt_band = sigma_tt_sim_valid[band_mask]
        
        # Sort by angle for plotting
        sort_idx = np.argsort(theta_band)
        theta_sorted = theta_band[sort_idx]
        sigma_rr_sorted = sigma_rr_band[sort_idx]
        sigma_tt_sorted = sigma_tt_band[sort_idx]
        
        # Analytical
        theta_ana = np.linspace(-np.pi, np.pi, 361)
        sigma_rr_ana_curve, sigma_tt_ana_curve, _ = kirsch_solution(
            r_target, theta_ana, tunnel_radius, sigma_v, sigma_h
        )
        
        ax.plot(np.degrees(theta_ana), sigma_rr_ana_curve/1e6, 'b-', linewidth=2, label='σ_rr (Kirsch)')
        ax.plot(np.degrees(theta_ana), sigma_tt_ana_curve/1e6, 'r-', linewidth=2, label='σ_θθ (Kirsch)')
        ax.scatter(np.degrees(theta_sorted), sigma_rr_sorted/1e6, s=10, c='blue', alpha=0.4, label='σ_rr (Sim)')
        ax.scatter(np.degrees(theta_sorted), sigma_tt_sorted/1e6, s=10, c='red', alpha=0.4, label='σ_θθ (Sim)')
        
        ax.set_xlabel('θ (degrees)', fontsize=12)
        ax.set_ylabel('Stress (MPa)', fontsize=12)
        ax.set_title(f'r/a = {r_ratio:.1f}', fontsize=14)
        ax.set_xlim([-180, 180])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=sigma_h/1e6, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=sigma_v/1e6, color='gray', linestyle=':', alpha=0.3)
    
    fig2.suptitle('Stress Distribution Around Tunnel at Different Distances', fontsize=14)
    plt.tight_layout()
    plt.savefig('long_tunnel_angular_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: long_tunnel_angular_distribution.png")
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    print("\n" + "="*70)
    print("ERROR STATISTICS")
    print("="*70)
    
    # Overall stats
    print(f"\nOverall (r/a = {r_min/tunnel_radius:.1f} to {r_max/tunnel_radius:.1f}):")
    print(f"  σ_rr: Mean error = {np.mean(error_rr):+.2f}%, Std = {np.std(error_rr):.2f}%")
    print(f"  σ_θθ: Mean error = {np.mean(error_tt):+.2f}%, Std = {np.std(error_tt):.2f}%")
    
    # Stats by region
    for r_lo, r_hi in [(1.2, 2.0), (2.0, 4.0), (4.0, 8.0)]:
        region_mask = (r_valid >= r_lo*tunnel_radius) & (r_valid < r_hi*tunnel_radius)
        if np.sum(region_mask) > 0:
            err_rr_region = error_rr[region_mask]
            err_tt_region = error_tt[region_mask]
            print(f"\n  Region r/a = {r_lo:.1f} to {r_hi:.1f} ({np.sum(region_mask)} particles):")
            print(f"    σ_rr: Mean = {np.mean(err_rr_region):+.2f}%, Std = {np.std(err_rr_region):.2f}%")
            print(f"    σ_θθ: Mean = {np.mean(err_tt_region):+.2f}%, Std = {np.std(err_tt_region):.2f}%")
    
    # Stats by angle
    print("\n  By angle (r/a = 1.2 to 4.0):")
    near_mask = (r_valid >= 1.2*tunnel_radius) & (r_valid < 4.0*tunnel_radius)
    for angle_deg in [0, 45, 90]:
        angle_rad = np.radians(angle_deg)
        angle_tol = np.radians(15)
        
        if angle_deg == 0:
            angle_mask = (np.abs(theta_valid) < angle_tol) | (np.abs(np.abs(theta_valid) - np.pi) < angle_tol)
        elif angle_deg == 90:
            angle_mask = np.abs(np.abs(theta_valid) - np.pi/2) < angle_tol
        else:
            angle_mask = (np.abs(theta_valid - angle_rad) < angle_tol) | (np.abs(theta_valid + angle_rad) < angle_tol)
            angle_mask |= (np.abs(theta_valid - (np.pi - angle_rad)) < angle_tol) | (np.abs(theta_valid + (np.pi - angle_rad)) < angle_tol)
        
        combined_mask = near_mask & angle_mask
        if np.sum(combined_mask) > 0:
            err_rr_ang = error_rr[combined_mask]
            err_tt_ang = error_tt[combined_mask]
            print(f"    θ ≈ {angle_deg}°: σ_rr = {np.mean(err_rr_ang):+.1f}% ± {np.std(err_rr_ang):.1f}%, "
                  f"σ_θθ = {np.mean(err_tt_ang):+.1f}% ± {np.std(err_tt_ang):.1f}%")
    
    # =========================================================================
    # Check plane strain condition
    # =========================================================================
    print("\n" + "="*70)
    print("PLANE STRAIN VERIFICATION (σ_yy check)")
    print("="*70)
    
    far_field_mask = r > 5.0 * tunnel_radius
    if np.sum(far_field_mask) > 0:
        ff_sigma_xx = np.mean(stress_xx_slice[far_field_mask])
        ff_sigma_yy = np.mean(stress_yy_slice[far_field_mask])
        ff_sigma_zz = np.mean(stress_zz_slice[far_field_mask])
        
        sigma_yy_plane_strain = nu * (ff_sigma_xx + ff_sigma_zz)
        
        print(f"\nFar-field stresses (r > 5a):")
        print(f"  σ_xx = {ff_sigma_xx/1e6:.4f} MPa (expected σ_h = {sigma_h/1e6:.4f} MPa)")
        print(f"  σ_zz = {ff_sigma_zz/1e6:.4f} MPa (expected σ_v = {sigma_v/1e6:.4f} MPa)")
        print(f"  σ_yy = {ff_sigma_yy/1e6:.4f} MPa")
        print(f"\n  Plane strain check: σ_yy = ν(σ_xx + σ_zz)")
        print(f"    Expected: {sigma_yy_plane_strain/1e6:.4f} MPa")
        print(f"    Actual:   {ff_sigma_yy/1e6:.4f} MPa")
        print(f"    Error:    {100*abs(ff_sigma_yy - sigma_yy_plane_strain)/abs(sigma_yy_plane_strain):.1f}%")
        
        if abs(ff_sigma_yy - sigma_yy_plane_strain)/abs(sigma_yy_plane_strain) > 0.1:
            print(f"\n  ⚠️  WARNING: σ_yy does not match plane strain assumption!")
            print(f"     This suggests the Y boundaries allow expansion/contraction.")
            print(f"     σ_yy ≈ σ_h indicates plane-stress-like behavior in Y.")
    
    plt.show()
    
    return {
        'mean_error_rr': np.mean(error_rr),
        'std_error_rr': np.std(error_rr),
        'mean_error_tt': np.mean(error_tt),
        'std_error_tt': np.std(error_tt),
    }


if __name__ == "__main__":
    results = analyze_and_plot()

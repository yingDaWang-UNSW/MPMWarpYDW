"""
Analyze plane strain Y boundary simulation vs Kirsch solution.
Compare with the free-slip Y boundary results to verify improvement.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

def kirsch_solution(r, theta, a, sigma_v, sigma_h):
    """Standard Kirsch solution for circular tunnel."""
    p0 = (sigma_v + sigma_h) / 2
    q0 = (sigma_v - sigma_h) / 2
    
    ratio2 = (a / r) ** 2
    ratio4 = (a / r) ** 4
    
    sigma_rr = p0 * (1 - ratio2) + q0 * (1 - 4*ratio2 + 3*ratio4) * np.cos(2*theta)
    sigma_tt = p0 * (1 + ratio2) - q0 * (1 + 3*ratio4) * np.cos(2*theta)
    
    return sigma_rr, sigma_tt


def analyze_planestrain():
    """Analyze the plane strain Y simulation results."""
    
    # Parameters
    output_dir = Path("./output_elastic_planestrain")
    tunnel_center = np.array([100.0, 20.0, 100.0])
    a = 10.0  # tunnel radius
    nu = 0.25
    
    rho, g, depth = 2600, 9.81, 100.0
    sigma_v = -rho * g * depth  # -2.55 MPa
    sigma_h = 0.5 * sigma_v     # -1.275 MPa
    
    # Expected plane strain σ_yy
    sigma_yy_plane_strain = nu * (sigma_h + sigma_v)  # -0.957 MPa
    
    print("="*70)
    print("PLANE STRAIN Y BOUNDARY SIMULATION ANALYSIS")
    print("="*70)
    print(f"Expected far-field stresses:")
    print(f"  σ_xx (horizontal): {sigma_h/1e6:.4f} MPa")
    print(f"  σ_zz (vertical):   {sigma_v/1e6:.4f} MPa")
    print(f"  σ_yy (plane strain): {sigma_yy_plane_strain/1e6:.4f} MPa")
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
    
    sigma_xx = stress[:, 0]
    sigma_yy = stress[:, 1]
    sigma_zz = stress[:, 2]
    sigma_xz = stress[:, 5]
    
    # Central Y-slice
    y_mask = np.abs(positions[:, 1] - 20.0) < 0.25
    pos_slice = positions[y_mask]
    stress_xx_slice = sigma_xx[y_mask]
    stress_yy_slice = sigma_yy[y_mask]
    stress_zz_slice = sigma_zz[y_mask]
    stress_xz_slice = sigma_xz[y_mask]
    
    print(f"\nCentral Y-slice (y = 20.0 ± 0.25m): {np.sum(y_mask)} particles")
    
    # Polar coordinates
    x_rel = pos_slice[:, 0] - tunnel_center[0]
    z_rel = pos_slice[:, 2] - tunnel_center[2]
    r = np.sqrt(x_rel**2 + z_rel**2)
    theta = np.arctan2(z_rel, x_rel)
    
    # Transform to polar stresses
    c, s = np.cos(theta), np.sin(theta)
    sigma_rr = stress_xx_slice*c**2 + stress_zz_slice*s**2 + 2*stress_xz_slice*s*c
    sigma_tt = stress_xx_slice*s**2 + stress_zz_slice*c**2 - 2*stress_xz_slice*s*c
    
    # =========================================================================
    # CHECK PLANE STRAIN CONDITION (σ_yy)
    # =========================================================================
    print("\n" + "="*70)
    print("PLANE STRAIN VERIFICATION")
    print("="*70)
    
    far_field_mask = r > 5.0 * a
    ff_sigma_xx = np.mean(stress_xx_slice[far_field_mask])
    ff_sigma_yy = np.mean(stress_yy_slice[far_field_mask])
    ff_sigma_zz = np.mean(stress_zz_slice[far_field_mask])
    
    sigma_yy_expected = nu * (ff_sigma_xx + ff_sigma_zz)
    
    print(f"\nFar-field stresses (r > 5a):")
    print(f"  σ_xx = {ff_sigma_xx/1e6:.4f} MPa (expected: {sigma_h/1e6:.4f} MPa)")
    print(f"  σ_zz = {ff_sigma_zz/1e6:.4f} MPa (expected: {sigma_v/1e6:.4f} MPa)")
    print(f"  σ_yy = {ff_sigma_yy/1e6:.4f} MPa")
    print()
    print(f"  Plane strain check: σ_yy = ν(σ_xx + σ_zz)")
    print(f"    Expected: {sigma_yy_expected/1e6:.4f} MPa")
    print(f"    Actual:   {ff_sigma_yy/1e6:.4f} MPa")
    error_yy = abs(ff_sigma_yy - sigma_yy_expected) / abs(sigma_yy_expected) * 100
    print(f"    Error:    {error_yy:.1f}%")
    
    if error_yy < 10:
        print(f"\n  ✅ PLANE STRAIN ACHIEVED! σ_yy matches expected value within {error_yy:.1f}%")
    else:
        print(f"\n  ⚠️ Plane strain not fully achieved. Error = {error_yy:.1f}%")
    
    # Compare to free-slip result
    print(f"\n  Comparison to free-slip Y boundaries:")
    print(f"    Free-slip σ_yy ≈ σ_h = -1.277 MPa (33% error)")
    print(f"    Plane strain σ_yy =    {ff_sigma_yy/1e6:.4f} MPa ({error_yy:.1f}% error)")
    
    # =========================================================================
    # WALL STRESS COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("WALL STRESS COMPARISON (r ≈ 1.35a)")
    print("="*70)
    
    wall_mask = (r >= 1.2*a) & (r <= 1.5*a)
    
    for ang_deg, name in [(0, 'Springline'), (45, '45°'), (90, 'Crown')]:
        ang_rad = np.radians(ang_deg)
        
        if ang_deg == 0:
            ang_mask = (np.abs(theta) < np.radians(15)) | (np.abs(np.abs(theta) - np.pi) < np.radians(15))
        elif ang_deg == 90:
            ang_mask = np.abs(np.abs(theta) - np.pi/2) < np.radians(15)
        else:
            ang_mask = np.zeros_like(theta, dtype=bool)
            for t in [ang_rad, np.pi-ang_rad, -ang_rad, -(np.pi-ang_rad)]:
                ang_mask |= np.abs(theta - t) < np.radians(15)
        
        combined = wall_mask & ang_mask
        if np.sum(combined) > 0:
            r_mean = np.mean(r[combined])
            _, kirsch_tt = kirsch_solution(r_mean, ang_rad, a, sigma_v, sigma_h)
            sim_tt = np.mean(sigma_tt[combined])
            sim_rr = np.mean(sigma_rr[combined])
            
            kirsch_rr, _ = kirsch_solution(r_mean, ang_rad, a, sigma_v, sigma_h)
            
            err_tt = (sim_tt - kirsch_tt) / abs(kirsch_tt) * 100
            print(f"\n{name} (n={np.sum(combined)}, r_mean={r_mean/a:.2f}a):")
            print(f"  Kirsch: σ_rr={kirsch_rr/1e6:.3f}, σ_θθ={kirsch_tt/1e6:.3f} MPa")
            print(f"  Sim:    σ_rr={sim_rr/1e6:.3f}, σ_θθ={sim_tt/1e6:.3f} MPa")
            print(f"  σ_θθ error: {err_tt:+.1f}%")
    
    # =========================================================================
    # GENERATE COMPARISON PLOTS
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    r_ana = np.linspace(a, 8*a, 200)
    angles = [(0, 'Springline (θ=0°)'), (45, 'θ=45°'), (90, 'Crown (θ=90°)')]
    
    for col, (ang_deg, title) in enumerate(angles):
        ang_rad = np.radians(ang_deg)
        
        # Analytical
        sigma_rr_ana, sigma_tt_ana = kirsch_solution(r_ana, ang_rad, a, sigma_v, sigma_h)
        
        # Filter simulation data
        tol = np.radians(15)
        if ang_deg == 0:
            mask = (np.abs(theta) < tol) | (np.abs(np.abs(theta) - np.pi) < tol)
        elif ang_deg == 90:
            mask = np.abs(np.abs(theta) - np.pi/2) < tol
        else:
            mask = (np.abs(theta - ang_rad) < tol) | (np.abs(theta + ang_rad) < tol)
            mask |= (np.abs(theta - (np.pi-ang_rad)) < tol) | (np.abs(theta + (np.pi-ang_rad)) < tol)
        
        r_mask = (r >= 1.2*a) & (r <= 8*a)
        mask = mask & r_mask
        
        # Plot σ_rr
        ax = axes[0, col]
        ax.plot(r_ana/a, sigma_rr_ana/1e6, 'b-', lw=2, label='Kirsch')
        ax.scatter(r[mask]/a, sigma_rr[mask]/1e6, s=6, c='red', alpha=0.3, label='Simulation')
        ax.axhline(sigma_h/1e6, color='gray', ls='--', alpha=0.5)
        ax.axhline(sigma_v/1e6, color='gray', ls=':', alpha=0.5)
        ax.set_xlim([1, 8])
        ax.set_ylabel('σ_rr (MPa)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
        
        # Plot σ_θθ
        ax = axes[1, col]
        ax.plot(r_ana/a, sigma_tt_ana/1e6, 'b-', lw=2, label='Kirsch')
        ax.scatter(r[mask]/a, sigma_tt[mask]/1e6, s=6, c='red', alpha=0.3, label='Simulation')
        ax.axhline(sigma_h/1e6, color='gray', ls='--', alpha=0.5, label='σ_h')
        ax.axhline(sigma_v/1e6, color='gray', ls=':', alpha=0.5, label='σ_v')
        ax.set_xlim([1, 8])
        ax.set_xlabel('r/a')
        ax.set_ylabel('σ_θθ (MPa)')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc='lower right')
    
    fig.suptitle(f'Plane Strain Y Boundary: Simulation vs Kirsch\n'
                 f'σ_yy error = {error_yy:.1f}% (vs 33% with free-slip)', fontsize=12)
    plt.tight_layout()
    plt.savefig('planestrain_kirsch_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: planestrain_kirsch_comparison.png")
    
    # =========================================================================
    # ANGULAR DISTRIBUTION
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    r_target = 1.5 * a
    r_band = 0.3 * a
    band_mask = np.abs(r - r_target) < r_band
    
    theta_deg = np.degrees(theta[band_mask])
    sigma_rr_band = sigma_rr[band_mask]
    sigma_tt_band = sigma_tt[band_mask]
    
    theta_ana = np.linspace(-180, 180, 361)
    sigma_rr_ana, sigma_tt_ana = kirsch_solution(r_target, np.radians(theta_ana), a, sigma_v, sigma_h)
    
    ax1.plot(theta_ana, sigma_rr_ana/1e6, 'b-', lw=2, label='Kirsch')
    ax1.scatter(theta_deg, sigma_rr_band/1e6, s=8, c='red', alpha=0.3, label='Simulation')
    ax1.set_xlabel('θ (degrees)')
    ax1.set_ylabel('σ_rr (MPa)')
    ax1.set_title(f'Radial Stress at r = {r_target/a:.1f}a')
    ax1.set_xlim([-180, 180])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(theta_ana, sigma_tt_ana/1e6, 'b-', lw=2, label='Kirsch')
    ax2.scatter(theta_deg, sigma_tt_band/1e6, s=8, c='red', alpha=0.3, label='Simulation')
    ax2.set_xlabel('θ (degrees)')
    ax2.set_ylabel('σ_θθ (MPa)')
    ax2.set_title(f'Tangential Stress at r = {r_target/a:.1f}a')
    ax2.set_xlim([-180, 180])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig2.suptitle('Angular Stress Distribution (Plane Strain Y)', fontsize=12)
    plt.tight_layout()
    plt.savefig('planestrain_angular_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: planestrain_angular_distribution.png")
    
    plt.show()


if __name__ == "__main__":
    analyze_planestrain()

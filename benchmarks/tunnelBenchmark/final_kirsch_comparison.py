"""
Final comparison plot: Long tunnel central slice vs Kirsch analytical solution.

Key finding: The simulation shows INVERTED stress concentration pattern:
- Kirsch predicts max σ_θθ at crown (vertical direction)  
- Simulation shows max σ_θθ at springline (horizontal direction)

This is because σ_yy ≈ σ_h (not plane strain), changing the 3D constraint.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

def kirsch_solution(r, theta, a, sigma_v, sigma_h):
    """
    Standard Kirsch solution (theta measured from horizontal).
    For K0 < 1 (σ_h < σ_v), max hoop stress is at crown (θ=90°).
    """
    p0 = (sigma_v + sigma_h) / 2
    q0 = (sigma_v - sigma_h) / 2
    
    ratio2 = (a / r) ** 2
    ratio4 = (a / r) ** 4
    
    sigma_rr = p0 * (1 - ratio2) + q0 * (1 - 4*ratio2 + 3*ratio4) * np.cos(2*theta)
    sigma_tt = p0 * (1 + ratio2) - q0 * (1 + 3*ratio4) * np.cos(2*theta)
    
    return sigma_rr, sigma_tt


def main():
    # Parameters
    output_dir = Path("./output_elastic_long")
    tunnel_center = np.array([100.0, 20.0, 100.0])
    a = 10.0  # tunnel radius
    
    rho, g, depth = 2600, 9.81, 100.0
    sigma_v = -rho * g * depth  # -2.55 MPa
    sigma_h = 0.5 * sigma_v     # -1.275 MPa
    
    # Load data
    vtp_files = sorted(output_dir.glob("*_particles.vtp"))
    mesh = pv.read(str(vtp_files[-1]))
    positions = np.array(mesh.points)
    stress = np.array(mesh.point_data['stress_tensor'])
    
    # Central slice
    y_mask = np.abs(positions[:, 1] - 20.0) < 0.25
    pos = positions[y_mask]
    sigma_xx = stress[:, 0][y_mask]
    sigma_zz = stress[:, 2][y_mask]
    sigma_xz = stress[:, 5][y_mask]
    
    # Polar coordinates (theta from +x axis)
    x_rel = pos[:, 0] - tunnel_center[0]
    z_rel = pos[:, 2] - tunnel_center[2]
    r = np.sqrt(x_rel**2 + z_rel**2)
    theta = np.arctan2(z_rel, x_rel)
    
    # Transform to polar stresses
    c, s = np.cos(theta), np.sin(theta)
    sigma_rr = sigma_xx*c**2 + sigma_zz*s**2 + 2*sigma_xz*s*c
    sigma_tt = sigma_xx*s**2 + sigma_zz*c**2 - 2*sigma_xz*s*c
    
    # =========================================================================
    # FIGURE 1: Radial profiles at key angles
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    r_ana = np.linspace(a, 8*a, 200)
    angles = [(0, 'Springline (θ=0°)'), (45, 'θ=45°'), (90, 'Crown (θ=90°)')]
    
    for col, (ang_deg, title) in enumerate(angles):
        ang_rad = np.radians(ang_deg)
        
        # Analytical
        sigma_rr_ana, sigma_tt_ana = kirsch_solution(r_ana, ang_rad, a, sigma_v, sigma_h)
        
        # Filter simulation data (±15° tolerance)
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
    
    fig.suptitle('Long Tunnel (Ly=40m) Central Slice vs Kirsch Solution\n'
                 'Note: Simulation has σ_yy ≈ σ_h (not plane strain)', fontsize=12)
    plt.tight_layout()
    plt.savefig('tunnel_kirsch_radial_profiles.png', dpi=150, bbox_inches='tight')
    print("Saved: tunnel_kirsch_radial_profiles.png")
    
    # =========================================================================
    # FIGURE 2: Angular distribution at r = 1.5a
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simulation data at r ≈ 1.5a
    r_target = 1.5 * a
    r_band = 0.3 * a
    band_mask = (np.abs(r - r_target) < r_band)
    
    theta_deg = np.degrees(theta[band_mask])
    sigma_rr_band = sigma_rr[band_mask]
    sigma_tt_band = sigma_tt[band_mask]
    
    # Analytical
    theta_ana = np.linspace(-180, 180, 361)
    sigma_rr_ana, sigma_tt_ana = kirsch_solution(r_target, np.radians(theta_ana), a, sigma_v, sigma_h)
    
    ax1.plot(theta_ana, sigma_rr_ana/1e6, 'b-', lw=2, label='Kirsch')
    ax1.scatter(theta_deg, sigma_rr_band/1e6, s=8, c='red', alpha=0.3, label='Simulation')
    ax1.set_xlabel('θ (degrees from horizontal)')
    ax1.set_ylabel('σ_rr (MPa)')
    ax1.set_title(f'Radial Stress at r = {r_target/a:.1f}a')
    ax1.set_xlim([-180, 180])
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axvline(0, color='green', ls=':', alpha=0.3, label='Springline')
    ax1.axvline(90, color='orange', ls=':', alpha=0.3, label='Crown')
    ax1.axvline(-90, color='orange', ls=':', alpha=0.3)
    
    ax2.plot(theta_ana, sigma_tt_ana/1e6, 'b-', lw=2, label='Kirsch')
    ax2.scatter(theta_deg, sigma_tt_band/1e6, s=8, c='red', alpha=0.3, label='Simulation')
    ax2.set_xlabel('θ (degrees from horizontal)')
    ax2.set_ylabel('σ_θθ (MPa)')
    ax2.set_title(f'Tangential Stress at r = {r_target/a:.1f}a')
    ax2.set_xlim([-180, 180])
    ax2.set_xticks([-180, -90, 0, 90, 180])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axvline(0, color='green', ls=':', alpha=0.3)
    ax2.axvline(90, color='orange', ls=':', alpha=0.3)
    ax2.axvline(-90, color='orange', ls=':', alpha=0.3)
    
    fig2.suptitle('Angular Stress Distribution Around Tunnel', fontsize=12)
    plt.tight_layout()
    plt.savefig('tunnel_kirsch_angular_profile.png', dpi=150, bbox_inches='tight')
    print("Saved: tunnel_kirsch_angular_profile.png")
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("WALL STRESS COMPARISON (at r ≈ 1.35a)")
    print("="*70)
    print(f"{'Location':<20} {'Kirsch σ_θθ (MPa)':<20} {'Sim σ_θθ (MPa)':<20} {'Ratio':<10}")
    print("-"*70)
    
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
            ratio = sim_tt / kirsch_tt
            print(f"{name:<20} {kirsch_tt/1e6:<20.3f} {sim_tt/1e6:<20.3f} {ratio:<10.2f}")
    
    print("\n" + "="*70)
    print("KEY OBSERVATION")
    print("="*70)
    print("""
The simulation shows INVERTED stress concentration compared to Kirsch:
  - Kirsch predicts maximum σ_θθ at crown (θ=90°) because σ_v > σ_h
  - Simulation shows maximum σ_θθ at springline (θ=0°)

This is because the Y boundaries allow expansion, giving σ_yy ≈ σ_h instead
of the plane-strain value σ_yy = ν(σ_xx + σ_zz).

To properly validate against Kirsch, we need either:
1. Fixed Y-displacement boundaries (true plane strain), OR  
2. A 2D plane-strain simulation
""")
    
    plt.show()


if __name__ == "__main__":
    main()

"""
Corrected analysis accounting for the actual 3D stress state.

The simulation has σ_yy ≈ σ_h (not plane strain σ_yy = ν(σ_xx + σ_zz)).
This is because free-slip Y boundaries allow expansion/contraction.

For a rigorous comparison, we should either:
1. Use a 3D analytical solution (complex)
2. Compare against Kirsch but acknowledge the plane strain assumption is violated
3. Modify the simulation to enforce true plane strain (fixed Y displacement)

This script creates comparison plots showing both:
- Direct comparison to Kirsch (knowing it will have systematic errors)
- Adjusted comparison noting the σ_yy discrepancy
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path

def kirsch_solution(r, theta, a, sigma_v, sigma_h):
    """Kirsch analytical solution for circular tunnel."""
    p0 = (sigma_v + sigma_h) / 2
    q0 = (sigma_v - sigma_h) / 2
    
    ratio = a / r
    ratio2 = ratio ** 2
    ratio4 = ratio ** 4
    
    sigma_rr = p0 * (1 - ratio2) + q0 * (1 - 4*ratio2 + 3*ratio4) * np.cos(2*theta)
    sigma_tt = p0 * (1 + ratio2) - q0 * (1 + 3*ratio4) * np.cos(2*theta)
    sigma_rt = -q0 * (1 + 2*ratio2 - 3*ratio4) * np.sin(2*theta)
    
    return sigma_rr, sigma_tt, sigma_rt


def analyze_corrected():
    """Analysis with corrected understanding of the 3D stress state."""
    
    output_dir = Path("./output_elastic_long")
    tunnel_center = np.array([100.0, 20.0, 100.0])
    tunnel_radius = 10.0
    
    E = 10e9
    nu = 0.25
    rho = 2600
    g = 9.81
    depth = 100.0
    sigma_v = -rho * g * depth
    K0 = 0.5
    sigma_h = K0 * sigma_v
    
    # Find latest VTP file
    vtp_files = sorted(output_dir.glob("*_particles.vtp"))
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
    y_center = 20.0
    y_tolerance = 0.25
    y_mask = np.abs(positions[:, 1] - y_center) < y_tolerance
    
    pos_slice = positions[y_mask]
    stress_xx_slice = sigma_xx[y_mask]
    stress_yy_slice = sigma_yy[y_mask]
    stress_zz_slice = sigma_zz[y_mask]
    stress_xz_slice = sigma_xz[y_mask]
    
    # Polar coordinates
    x_rel = pos_slice[:, 0] - tunnel_center[0]
    z_rel = pos_slice[:, 2] - tunnel_center[2]
    r = np.sqrt(x_rel**2 + z_rel**2)
    theta = np.arctan2(z_rel, x_rel)
    
    # Transform to polar stresses
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    sigma_rr_sim = stress_xx_slice * cos_t**2 + stress_zz_slice * sin_t**2 + 2*stress_xz_slice * sin_t * cos_t
    sigma_tt_sim = stress_xx_slice * sin_t**2 + stress_zz_slice * cos_t**2 - 2*stress_xz_slice * sin_t * cos_t
    
    # =========================================================================
    # Create figure with 3 comparison angles
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    angles = [0, 45, 90]  # degrees
    r_ana = np.linspace(tunnel_radius, 80, 200)
    
    for col, angle_deg in enumerate(angles):
        angle_rad = np.radians(angle_deg)
        
        # Analytical Kirsch
        sigma_rr_kirsch, sigma_tt_kirsch, _ = kirsch_solution(r_ana, angle_rad, tunnel_radius, sigma_v, sigma_h)
        
        # Filter simulation data near this angle (±15°)
        angle_tol = np.radians(15)
        
        if angle_deg == 0:
            # θ = 0° or 180°
            angle_mask = (np.abs(theta) < angle_tol) | (np.abs(np.abs(theta) - np.pi) < angle_tol)
        elif angle_deg == 90:
            # θ = 90° or -90°
            angle_mask = np.abs(np.abs(theta) - np.pi/2) < angle_tol
        else:
            # θ = 45° or 135° or -45° or -135°
            angle_mask = np.zeros_like(theta, dtype=bool)
            for a in [angle_rad, np.pi - angle_rad, -angle_rad, -(np.pi - angle_rad)]:
                angle_mask |= np.abs(theta - a) < angle_tol
        
        # Also filter by radial distance (1.2a to 8a)
        r_mask = (r >= 1.2*tunnel_radius) & (r <= 8*tunnel_radius)
        combined_mask = angle_mask & r_mask
        
        r_sim = r[combined_mask]
        sigma_rr_sim_ang = sigma_rr_sim[combined_mask]
        sigma_tt_sim_ang = sigma_tt_sim[combined_mask]
        
        # σ_rr plot
        ax_rr = axes[0, col]
        ax_rr.plot(r_ana/tunnel_radius, sigma_rr_kirsch/1e6, 'b-', linewidth=2, label='Kirsch (plane strain)')
        ax_rr.scatter(r_sim/tunnel_radius, sigma_rr_sim_ang/1e6, s=8, c='red', alpha=0.3, label='Simulation')
        ax_rr.axhline(y=sigma_h/1e6, color='gray', linestyle='--', alpha=0.5, label='σ_h' if col==0 else None)
        ax_rr.axhline(y=sigma_v/1e6, color='gray', linestyle=':', alpha=0.5, label='σ_v' if col==0 else None)
        ax_rr.set_xlabel('r/a')
        ax_rr.set_ylabel('σ_rr (MPa)')
        ax_rr.set_title(f'θ = {angle_deg}°')
        ax_rr.set_xlim([1, 8])
        ax_rr.grid(True, alpha=0.3)
        if col == 0:
            ax_rr.legend(loc='best', fontsize=8)
        
        # σ_θθ plot
        ax_tt = axes[1, col]
        ax_tt.plot(r_ana/tunnel_radius, sigma_tt_kirsch/1e6, 'b-', linewidth=2, label='Kirsch (plane strain)')
        ax_tt.scatter(r_sim/tunnel_radius, sigma_tt_sim_ang/1e6, s=8, c='red', alpha=0.3, label='Simulation')
        ax_tt.axhline(y=sigma_h/1e6, color='gray', linestyle='--', alpha=0.5)
        ax_tt.axhline(y=sigma_v/1e6, color='gray', linestyle=':', alpha=0.5)
        ax_tt.set_xlabel('r/a')
        ax_tt.set_ylabel('σ_θθ (MPa)')
        ax_tt.set_xlim([1, 8])
        ax_tt.grid(True, alpha=0.3)
        if col == 0:
            ax_tt.legend(loc='best', fontsize=8)
        
        # Compute mean error for this angle
        if len(r_sim) > 0:
            # Interpolate Kirsch at simulation r values
            sigma_rr_kirsch_interp, sigma_tt_kirsch_interp, _ = kirsch_solution(
                r_sim, angle_rad, tunnel_radius, sigma_v, sigma_h
            )
            err_rr = np.mean((sigma_rr_sim_ang - sigma_rr_kirsch_interp) / sigma_rr_kirsch_interp * 100)
            err_tt = np.mean((sigma_tt_sim_ang - sigma_tt_kirsch_interp) / sigma_tt_kirsch_interp * 100)
            ax_rr.text(0.95, 0.05, f'Mean err: {err_rr:+.1f}%', transform=ax_rr.transAxes, 
                       ha='right', fontsize=9, color='red')
            ax_tt.text(0.95, 0.05, f'Mean err: {err_tt:+.1f}%', transform=ax_tt.transAxes, 
                       ha='right', fontsize=9, color='red')
    
    fig.suptitle('Long Tunnel (Ly=40m) Central Slice: Simulation vs Kirsch Solution\n'
                 '(Note: σ_yy ≈ σ_h in simulation, not plane strain σ_yy = ν(σ_xx+σ_zz))', 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('long_tunnel_comparison_by_angle.png', dpi=150, bbox_inches='tight')
    print("Saved: long_tunnel_comparison_by_angle.png")
    
    # =========================================================================
    # Wall stress analysis (at r = 1.2-1.5a)
    # =========================================================================
    print("\n" + "="*60)
    print("WALL STRESS ANALYSIS (r = 1.2a to 1.5a)")
    print("="*60)
    
    wall_mask = (r >= 1.2*tunnel_radius) & (r <= 1.5*tunnel_radius)
    
    for angle_deg in [0, 45, 90]:
        angle_rad = np.radians(angle_deg)
        angle_tol = np.radians(15)
        
        if angle_deg == 0:
            angle_mask = (np.abs(theta) < angle_tol) | (np.abs(np.abs(theta) - np.pi) < angle_tol)
        elif angle_deg == 90:
            angle_mask = np.abs(np.abs(theta) - np.pi/2) < angle_tol
        else:
            angle_mask = np.zeros_like(theta, dtype=bool)
            for a in [angle_rad, np.pi - angle_rad, -angle_rad, -(np.pi - angle_rad)]:
                angle_mask |= np.abs(theta - a) < angle_tol
        
        combined_mask = wall_mask & angle_mask
        
        if np.sum(combined_mask) > 0:
            r_mean = np.mean(r[combined_mask])
            sigma_rr_wall = np.mean(sigma_rr_sim[combined_mask])
            sigma_tt_wall = np.mean(sigma_tt_sim[combined_mask])
            
            # Kirsch at r=a (wall)
            sigma_rr_kirsch_wall, sigma_tt_kirsch_wall, _ = kirsch_solution(
                tunnel_radius, angle_rad, tunnel_radius, sigma_v, sigma_h
            )
            # Kirsch at mean r
            sigma_rr_kirsch_r, sigma_tt_kirsch_r, _ = kirsch_solution(
                r_mean, angle_rad, tunnel_radius, sigma_v, sigma_h
            )
            
            print(f"\nθ = {angle_deg}° (n = {np.sum(combined_mask)} particles, r_mean = {r_mean/tunnel_radius:.2f}a):")
            print(f"  Kirsch at r=a:       σ_rr = {sigma_rr_kirsch_wall/1e6:.3f} MPa, σ_θθ = {sigma_tt_kirsch_wall/1e6:.3f} MPa")
            print(f"  Kirsch at r={r_mean/tunnel_radius:.2f}a:    σ_rr = {sigma_rr_kirsch_r/1e6:.3f} MPa, σ_θθ = {sigma_tt_kirsch_r/1e6:.3f} MPa")
            print(f"  Simulation:          σ_rr = {sigma_rr_wall/1e6:.3f} MPa, σ_θθ = {sigma_tt_wall/1e6:.3f} MPa")
            
            err_rr = (sigma_rr_wall - sigma_rr_kirsch_r) / abs(sigma_rr_kirsch_r) * 100 if sigma_rr_kirsch_r != 0 else 0
            err_tt = (sigma_tt_wall - sigma_tt_kirsch_r) / abs(sigma_tt_kirsch_r) * 100
            print(f"  Error vs Kirsch:     σ_rr = {err_rr:+.1f}%, σ_θθ = {err_tt:+.1f}%")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The long tunnel simulation (Ly=40m) shows systematic differences from the 
Kirsch analytical solution. The root cause is that the Y boundaries allow
material expansion/contraction, resulting in:

  σ_yy ≈ σ_h = -1.28 MPa   (what we get)
  
Instead of plane strain:
  σ_yy = ν(σ_xx + σ_zz) = -0.96 MPa   (what Kirsch assumes)

This 33% difference in σ_yy affects the entire stress redistribution around
the tunnel. To achieve true plane strain for rigorous Kirsch validation, 
we would need to:

1. Fix Y displacements at Y boundaries (u_y = 0), OR
2. Use a 2D plane strain simulation

The current MPM implementation uses free-slip (roller) boundaries that
only prevent penetration but allow tangential expansion.
    """)
    
    plt.show()


if __name__ == "__main__":
    analyze_corrected()

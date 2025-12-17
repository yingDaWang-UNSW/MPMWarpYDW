"""
Compare low-resolution and high-resolution tunnel simulations against Kirsch solution.

This script generates comparison plots showing:
1. Both resolutions vs analytical Kirsch solution
2. Error convergence with resolution
3. Wall stress accuracy at different resolutions
"""

import numpy as np
import os
import sys
import glob
import argparse

# Import from the main comparison script
from compare_kirsch_solution import (
    kirsch_stress_polar,
    read_vtp_file,
    analyze_tunnel_stress,
    HAS_MATPLOTLIB,
    HAS_PYVISTA
)

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D


def compare_resolutions(lores_dir, hires_dir, output_dir, 
                        tunnel_radius=10.0, tunnel_center=(100.0, 2.0, 100.0),
                        density=2600.0, gravity=9.81, z_top=200.0, K0=0.5):
    """
    Compare low and high resolution simulations.
    """
    
    # Find VTP files
    lores_files = sorted(glob.glob(os.path.join(lores_dir, '*_particles.vtp')))
    hires_files = sorted(glob.glob(os.path.join(hires_dir, '*_particles.vtp')))
    
    if not lores_files:
        print(f"No VTP files found in {lores_dir}")
        return
    if not hires_files:
        print(f"No VTP files found in {hires_dir}")
        return
    
    lores_file = lores_files[-1]
    hires_file = hires_files[-1]
    
    print(f"Low-res file:  {lores_file}")
    print(f"High-res file: {hires_file}")
    
    # Compute far-field stresses at tunnel center
    depth = z_top - tunnel_center[2]
    sigma_v = density * gravity * depth
    sigma_h = K0 * sigma_v
    
    print(f"\nFar-field stresses at tunnel depth:")
    print(f"  σ_v = {sigma_v/1e6:.3f} MPa")
    print(f"  σ_h = {sigma_h/1e6:.3f} MPa")
    
    # Read and analyze both resolutions
    print("\nAnalyzing low-resolution simulation...")
    pos_lo, stress_lo = read_vtp_file(lores_file)
    results_lo = analyze_tunnel_stress(
        pos_lo, stress_lo, tunnel_center, tunnel_radius,
        sigma_v, sigma_h, y_slice_tolerance=2.5,
        density=density, gravity=gravity, z_top=z_top, K0=K0
    )
    
    print("\nAnalyzing high-resolution simulation...")
    pos_hi, stress_hi = read_vtp_file(hires_file)
    results_hi = analyze_tunnel_stress(
        pos_hi, stress_hi, tunnel_center, tunnel_radius,
        sigma_v, sigma_h, y_slice_tolerance=2.5,
        density=density, gravity=gravity, z_top=z_top, K0=K0
    )
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("RESOLUTION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<40} {'Low-Res (1.0m)':<15} {'High-Res (0.5m)':<15}")
    print("-" * 70)
    print(f"{'Particles analyzed':<40} {len(results_lo['r']):<15,} {len(results_hi['r']):<15,}")
    print(f"{'Mean rel. error (radial stress)':<40} {results_lo['mean_rel_err_rr']*100:<15.2f}% {results_hi['mean_rel_err_rr']*100:<15.2f}%")
    print(f"{'Mean rel. error (hoop stress)':<40} {results_lo['mean_rel_err_tt']*100:<15.2f}% {results_hi['mean_rel_err_tt']*100:<15.2f}%")
    print(f"{'Max rel. error (radial stress)':<40} {results_lo['max_rel_err_rr']*100:<15.2f}% {results_hi['max_rel_err_rr']*100:<15.2f}%")
    print(f"{'Max rel. error (hoop stress)':<40} {results_lo['max_rel_err_tt']*100:<15.2f}% {results_hi['max_rel_err_tt']*100:<15.2f}%")
    print("=" * 70)
    
    # Generate comparison plots
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return results_lo, results_hi
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Radial profiles at different angles
    for angle_deg in [0, 45, 90]:
        plot_radial_comparison(results_lo, results_hi, angle_deg, 
                              tunnel_radius, density, gravity, z_top, K0, tunnel_center,
                              os.path.join(output_dir, f'comparison_radial_{angle_deg}deg.png'))
    
    # Plot 2: Error vs radius
    plot_error_vs_radius(results_lo, results_hi, tunnel_radius,
                        os.path.join(output_dir, 'comparison_error_vs_radius.png'))
    
    # Plot 3: Wall stress comparison
    plot_wall_stress_comparison(results_lo, results_hi, tunnel_radius, sigma_v, sigma_h,
                               os.path.join(output_dir, 'comparison_wall_stress.png'))
    
    # Plot 4: 2D error comparison
    plot_2d_error_comparison(results_lo, results_hi, tunnel_radius, tunnel_center,
                            os.path.join(output_dir, 'comparison_2d_error.png'))
    
    return results_lo, results_hi


def plot_radial_comparison(results_lo, results_hi, angle_deg, 
                          tunnel_radius, density, gravity, z_top, K0, tunnel_center,
                          output_path):
    """Plot radial stress profile comparing both resolutions."""
    
    a = tunnel_radius
    zc = tunnel_center[2]
    theta_ana = np.radians(angle_deg)
    
    # Analytical profile with depth-varying stress
    r_ana = np.linspace(a, 5*a, 100)
    z_ana = zc + r_ana * np.cos(theta_ana)
    depth_ana = z_top - z_ana
    sigma_v_ana = density * gravity * depth_ana
    sigma_h_ana = K0 * sigma_v_ana
    
    ana_rr = np.zeros_like(r_ana)
    ana_tt = np.zeros_like(r_ana)
    for i in range(len(r_ana)):
        ana_rr[i], ana_tt[i], _ = kirsch_stress_polar(r_ana[i], theta_ana, a,
                                                       sigma_v_ana[i], sigma_h_ana[i])
    
    # Filter simulation data near this angle
    angle_tol = np.radians(15)
    
    def filter_by_angle(results, theta_target):
        angle_diff = np.abs(np.mod(results['theta'] - theta_target + np.pi, 2*np.pi) - np.pi)
        mask = angle_diff < angle_tol
        return results['r'][mask], results['sim_sigma_rr'][mask], results['sim_sigma_tt'][mask]
    
    r_lo, rr_lo, tt_lo = filter_by_angle(results_lo, theta_ana)
    r_hi, rr_hi, tt_hi = filter_by_angle(results_hi, theta_ana)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Radial stress
    ax = axes[0]
    ax.plot(r_ana/a, ana_rr/1e6, 'k-', linewidth=2.5, label='Kirsch analytical')
    ax.scatter(r_lo/a, rr_lo/1e6, c='blue', alpha=0.4, s=15, label='Low-res (1.0m)')
    ax.scatter(r_hi/a, rr_hi/1e6, c='red', alpha=0.4, s=15, label='High-res (0.5m)')
    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('σ_rr [MPa]', fontsize=12)
    ax.set_title(f'Radial Stress (θ = {angle_deg}°)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    
    # Tangential stress
    ax = axes[1]
    ax.plot(r_ana/a, ana_tt/1e6, 'k-', linewidth=2.5, label='Kirsch analytical')
    ax.scatter(r_lo/a, tt_lo/1e6, c='blue', alpha=0.4, s=15, label='Low-res (1.0m)')
    ax.scatter(r_hi/a, tt_hi/1e6, c='red', alpha=0.4, s=15, label='High-res (0.5m)')
    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('σ_θθ [MPa]', fontsize=12)
    ax.set_title(f'Tangential (Hoop) Stress (θ = {angle_deg}°)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    
    plt.suptitle(f'Resolution Comparison at θ = {angle_deg}°', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_vs_radius(results_lo, results_hi, tunnel_radius, output_path):
    """Plot error as function of normalized radius."""
    
    a = tunnel_radius
    
    # Bin errors by radius
    r_bins = np.linspace(1.0, 5.0, 20)
    
    def bin_errors(results, r_bins):
        r_norm = results['r'] / a
        err_rr = np.abs(results['rel_err_rr'])
        err_tt = np.abs(results['rel_err_tt'])
        
        mean_err_rr = []
        mean_err_tt = []
        r_centers = []
        
        for i in range(len(r_bins) - 1):
            mask = (r_norm >= r_bins[i]) & (r_norm < r_bins[i+1])
            if np.any(mask):
                mean_err_rr.append(np.mean(err_rr[mask]) * 100)
                mean_err_tt.append(np.mean(err_tt[mask]) * 100)
                r_centers.append((r_bins[i] + r_bins[i+1]) / 2)
        
        return np.array(r_centers), np.array(mean_err_rr), np.array(mean_err_tt)
    
    r_lo, err_rr_lo, err_tt_lo = bin_errors(results_lo, r_bins)
    r_hi, err_rr_hi, err_tt_hi = bin_errors(results_hi, r_bins)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Radial stress error
    ax = axes[0]
    ax.plot(r_lo, err_rr_lo, 'b-o', linewidth=2, markersize=6, label='Low-res (1.0m)')
    ax.plot(r_hi, err_rr_hi, 'r-s', linewidth=2, markersize=6, label='High-res (0.5m)')
    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('Mean Relative Error [%]', fontsize=12)
    ax.set_title('Radial Stress Error vs Distance', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    
    # Tangential stress error
    ax = axes[1]
    ax.plot(r_lo, err_tt_lo, 'b-o', linewidth=2, markersize=6, label='Low-res (1.0m)')
    ax.plot(r_hi, err_tt_hi, 'r-s', linewidth=2, markersize=6, label='High-res (0.5m)')
    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('Mean Relative Error [%]', fontsize=12)
    ax.set_title('Hoop Stress Error vs Distance', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    
    plt.suptitle('Error Convergence with Resolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_wall_stress_comparison(results_lo, results_hi, tunnel_radius, sigma_v, sigma_h, output_path):
    """Plot wall stress around the tunnel perimeter."""
    
    a = tunnel_radius
    
    # Analytical wall stress
    theta_ana = np.linspace(-np.pi, np.pi, 180)
    _, wall_tt_ana, _ = kirsch_stress_polar(a * 1.001, theta_ana, a, sigma_v, sigma_h)
    
    # Filter wall particles (r < 1.3a)
    def get_wall_stress(results):
        wall_mask = results['r'] < 1.3 * a
        return results['theta'][wall_mask], results['sim_sigma_tt'][wall_mask]
    
    theta_lo, tt_lo = get_wall_stress(results_lo)
    theta_hi, tt_hi = get_wall_stress(results_hi)
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    ax.plot(theta_ana, wall_tt_ana/1e6, 'k-', linewidth=2.5, label='Kirsch analytical')
    ax.scatter(theta_lo, tt_lo/1e6, c='blue', alpha=0.5, s=20, label='Low-res (1.0m)')
    ax.scatter(theta_hi, tt_hi/1e6, c='red', alpha=0.5, s=20, label='High-res (0.5m)')
    
    ax.set_title('Wall Hoop Stress σ_θθ [MPa]\n(θ=0° is crown, θ=90° is springline)', 
                fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_2d_error_comparison(results_lo, results_hi, tunnel_radius, tunnel_center, output_path):
    """Plot 2D error fields for both resolutions."""
    
    a = tunnel_radius
    xc, _, zc = tunnel_center
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine common error scale
    all_errors = np.concatenate([results_lo['err_tt'], results_hi['err_tt']]) / 1e6
    vmax = np.percentile(np.abs(all_errors), 95)
    
    # Low-res error
    ax = axes[0]
    pos_lo = results_lo['positions']
    err_lo = results_lo['err_tt'] / 1e6
    sc = ax.scatter(pos_lo[:, 0], pos_lo[:, 2], c=err_lo, cmap='RdBu', s=5,
                   vmin=-vmax, vmax=vmax)
    circle = plt.Circle((xc, zc), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title(f'Low-Res Error (1.0m spacing)\nMean: {results_lo["mean_rel_err_tt"]*100:.2f}%', fontsize=12)
    plt.colorbar(sc, ax=ax, label='Error [MPa]')
    
    # High-res error
    ax = axes[1]
    pos_hi = results_hi['positions']
    err_hi = results_hi['err_tt'] / 1e6
    sc = ax.scatter(pos_hi[:, 0], pos_hi[:, 2], c=err_hi, cmap='RdBu', s=2,
                   vmin=-vmax, vmax=vmax)
    circle = plt.Circle((xc, zc), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Z [m]', fontsize=12)
    ax.set_title(f'High-Res Error (0.5m spacing)\nMean: {results_hi["mean_rel_err_tt"]*100:.2f}%', fontsize=12)
    plt.colorbar(sc, ax=ax, label='Error [MPa]')
    
    plt.suptitle('Hoop Stress Error (Simulation - Kirsch)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare low and high resolution tunnel simulations')
    parser.add_argument('--lores_dir', type=str, default='./output_elastic',
                       help='Directory with low-resolution output')
    parser.add_argument('--hires_dir', type=str, default='./output_elastic_hires',
                       help='Directory with high-resolution output')
    parser.add_argument('--output_dir', type=str, default='./plots_comparison',
                       help='Directory for comparison plots')
    parser.add_argument('--tunnel_radius', type=float, default=10.0)
    parser.add_argument('--tunnel_center', type=float, nargs=3, default=[100.0, 2.0, 100.0])
    parser.add_argument('--density', type=float, default=2600.0)
    parser.add_argument('--K0', type=float, default=0.5)
    parser.add_argument('--z_top', type=float, default=200.0)
    parser.add_argument('--gravity', type=float, default=9.81)
    
    args = parser.parse_args()
    
    compare_resolutions(
        args.lores_dir, args.hires_dir, args.output_dir,
        tunnel_radius=args.tunnel_radius,
        tunnel_center=tuple(args.tunnel_center),
        density=args.density,
        gravity=args.gravity,
        z_top=args.z_top,
        K0=args.K0
    )


if __name__ == '__main__':
    main()

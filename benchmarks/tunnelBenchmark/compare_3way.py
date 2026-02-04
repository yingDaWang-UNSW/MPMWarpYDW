"""
Compare THREE tunnel resolutions against Kirsch analytical solution:
1. Low-res: 1.0m uniform grid spacing (~200k particles)
2. High-res: 0.5m uniform grid spacing (~1.4M particles)
3. Very High-res: 0.4m uniform grid spacing (~2.7M particles)

This demonstrates h-convergence: as grid spacing decreases, error decreases.
"""

import numpy as np
import os
import sys
import glob
import argparse

# Try to import from the main comparison script
try:
    from compare_kirsch_solution import (
        kirsch_stress_polar,
        read_vtp_file,
        analyze_tunnel_stress,
        HAS_MATPLOTLIB,
        HAS_PYVISTA
    )
except ImportError:
    print("Error: compare_kirsch_solution.py not found")
    sys.exit(1)

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch


def parse_args():
    parser = argparse.ArgumentParser(description='Compare 3 resolution levels')
    parser.add_argument('--lores_dir', type=str, default='./output_elastic',
                        help='Low-resolution output directory')
    parser.add_argument('--hires_dir', type=str, default='./output_elastic_hires',
                        help='High-resolution output directory')
    parser.add_argument('--adaptive_dir', type=str, default='./output_elastic_vhires',
                        help='Very high-res mesh output directory')
    parser.add_argument('--output_dir', type=str, default='./plots_3way',
                        help='Directory for output plots')
    parser.add_argument('--tunnel_radius', type=float, default=10.0)
    parser.add_argument('--tunnel_center', type=float, nargs=3, default=[100.0, 2.0, 100.0])
    parser.add_argument('--density', type=float, default=2600.0)
    parser.add_argument('--K0', type=float, default=0.5)
    parser.add_argument('--z_top', type=float, default=200.0)
    parser.add_argument('--gravity', type=float, default=9.81)
    return parser.parse_args()


def compare_three_resolutions(lores_dir, hires_dir, adaptive_dir, output_dir,
                              tunnel_radius=10.0, tunnel_center=(100.0, 2.0, 100.0),
                              density=2600.0, gravity=9.81, z_top=200.0, K0=0.5):
    """
    Compare three different mesh approaches.
    """
    
    # Find VTP files (take final timestep from each)
    def find_final_vtp(directory, name):
        files = sorted(glob.glob(os.path.join(directory, '*_particles.vtp')))
        if not files:
            print(f"WARNING: No VTP files found in {directory}")
            return None
        print(f"{name} file: {files[-1]}")
        return files[-1]
    
    lores_file = find_final_vtp(lores_dir, "Low-res")
    hires_file = find_final_vtp(hires_dir, "High-res")
    adaptive_file = find_final_vtp(adaptive_dir, "Adaptive")
    
    available_files = [(f, n) for f, n in [(lores_file, 'Low-res'), 
                                            (hires_file, 'High-res'),
                                            (adaptive_file, 'VHires')] if f is not None]
    
    if len(available_files) < 2:
        print("Need at least 2 simulations to compare")
        return
    
    # Compute far-field stresses at tunnel center
    depth = z_top - tunnel_center[2]
    sigma_v = density * gravity * depth
    sigma_h = K0 * sigma_v
    
    print(f"\nFar-field stresses at tunnel depth (z={tunnel_center[2]}m, depth={depth}m):")
    print(f"  σ_v = {sigma_v/1e6:.3f} MPa (compression positive for display)")
    print(f"  σ_h = {sigma_h/1e6:.3f} MPa")
    
    # Analyze each available simulation
    # Use tight Y-slice to get only the central layer of particles
    # This reduces scatter from multiple Y-layers having slightly different stress states
    results = {}
    
    if lores_file:
        print("\nAnalyzing LOW-RESOLUTION simulation (1.0m uniform grid)...")
        pos, stress = read_vtp_file(lores_file)
        # For 1.0m spacing, use tolerance of 0.6m to get single central layer
        y_tol = 0.6  # Slightly more than half the spacing
        results['lores'] = analyze_tunnel_stress(
            pos, stress, tunnel_center, tunnel_radius,
            sigma_v, sigma_h, y_slice_tolerance=y_tol,
            density=density, gravity=gravity, z_top=z_top, K0=K0
        )
        results['lores']['name'] = 'Low-res (1.0m grid)'
        results['lores']['color'] = 'blue'
        results['lores']['marker'] = 'o'
        results['lores']['spacing'] = 1.0
    
    if hires_file:
        print("\nAnalyzing HIGH-RESOLUTION simulation (0.5m uniform grid)...")
        pos, stress = read_vtp_file(hires_file)
        # For 0.5m spacing, use tolerance of 0.3m to get single central layer
        y_tol = 0.3
        results['hires'] = analyze_tunnel_stress(
            pos, stress, tunnel_center, tunnel_radius,
            sigma_v, sigma_h, y_slice_tolerance=y_tol,
            density=density, gravity=gravity, z_top=z_top, K0=K0
        )
        results['hires']['name'] = 'High-res (0.5m grid)'
        results['hires']['color'] = 'red'
        results['hires']['marker'] = 's'
        results['hires']['spacing'] = 0.5
    
    if adaptive_file:
        print("\nAnalyzing VERY HIGH-RES simulation (0.4m uniform grid)...")
        pos, stress = read_vtp_file(adaptive_file)
        # For 0.4m spacing, use tolerance of 0.25m to get single central layer
        y_tol = 0.25
        results['adaptive'] = analyze_tunnel_stress(
            pos, stress, tunnel_center, tunnel_radius,
            sigma_v, sigma_h, y_slice_tolerance=y_tol,
            density=density, gravity=gravity, z_top=z_top, K0=K0
        )
        results['adaptive']['name'] = 'VHires (0.4m grid)'
        results['adaptive']['color'] = 'green'
        results['adaptive']['marker'] = '^'
        results['adaptive']['spacing'] = 0.4
    
    # Print summary table
    print("\n" + "=" * 90)
    print("THREE-WAY RESOLUTION COMPARISON")
    print("=" * 90)
    
    header = f"{'Metric':<35}"
    for key in ['lores', 'hires', 'adaptive']:
        if key in results:
            header += f" {results[key]['name']:<20}"
    print(header)
    print("-" * 90)
    
    metrics = [
        ('Particles analyzed', 'n_particles', ''),
        ('Mean rel. error (σ_rr)', 'mean_rel_err_rr', '%'),
        ('Mean rel. error (σ_θθ)', 'mean_rel_err_tt', '%'),
        ('Max rel. error (σ_rr)', 'max_rel_err_rr', '%'),
        ('Max rel. error (σ_θθ)', 'max_rel_err_tt', '%'),
    ]
    
    for metric_name, metric_key, unit in metrics:
        row = f"{metric_name:<35}"
        for key in ['lores', 'hires', 'adaptive']:
            if key in results:
                val = results[key].get(metric_key, len(results[key]['r']))
                if metric_key == 'n_particles':
                    val = len(results[key]['r'])
                if unit == '%':
                    row += f" {val*100:<20.2f}"
                else:
                    row += f" {val:<20,}"
        row += unit
        print(row)
    
    print("=" * 90)
    
    # Generate comparison plots
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return results
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Radial profiles at different angles (3-way)
    for angle_deg in [0, 45, 90]:
        plot_radial_3way(results, angle_deg, tunnel_radius, density, gravity, 
                        z_top, K0, tunnel_center,
                        os.path.join(output_dir, f'3way_radial_{angle_deg}deg.png'))
    
    # Plot 2: Wall stress comparison (hoop stress vs angle)
    plot_wall_stress_3way(results, tunnel_radius, sigma_v, sigma_h,
                         os.path.join(output_dir, '3way_wall_stress.png'))
    
    # Plot 3: Error bar chart comparison
    plot_error_bars_3way(results, os.path.join(output_dir, '3way_error_bars.png'))
    
    # Plot 4: 2D stress field comparison (side by side)
    plot_2d_stress_3way(results, tunnel_radius, tunnel_center,
                       os.path.join(output_dir, '3way_2d_stress.png'))
    
    # Plot 5: Error convergence (particles vs error)
    plot_convergence(results, os.path.join(output_dir, '3way_convergence.png'))
    
    return results


def plot_radial_3way(results, angle_deg, tunnel_radius, density, gravity, 
                     z_top, K0, tunnel_center, output_path):
    """Plot radial stress profile comparing all three resolutions."""
    
    a = tunnel_radius
    zc = tunnel_center[2]
    theta_ana = np.radians(angle_deg)
    
    # Analytical profile with depth-varying stress
    # Start from 1.2a to match the analysis filter
    r_ana = np.linspace(1.2*a, 5*a, 100)
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
    
    def filter_by_angle(res, theta_target):
        angle_diff = np.abs(np.mod(res['theta'] - theta_target + np.pi, 2*np.pi) - np.pi)
        mask = angle_diff < angle_tol
        return res['r'][mask], res['sim_sigma_rr'][mask], res['sim_sigma_tt'][mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Radial stress
    ax = axes[0]
    ax.plot(r_ana/a, ana_rr/1e6, 'k-', linewidth=2.5, label='Kirsch analytical', zorder=10)
    
    for key in ['lores', 'hires', 'adaptive']:
        if key in results:
            r_sim, rr_sim, _ = filter_by_angle(results[key], theta_ana)
            ax.scatter(r_sim/a, rr_sim/1e6, c=results[key]['color'], alpha=0.4, s=15,
                      marker=results[key]['marker'], label=results[key]['name'])
    
    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('σ_rr [MPa]', fontsize=12)
    ax.set_title(f'Radial Stress (θ = {angle_deg}°)', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.2, 5)
    
    # Tangential stress
    ax = axes[1]
    ax.plot(r_ana/a, ana_tt/1e6, 'k-', linewidth=2.5, label='Kirsch analytical', zorder=10)
    
    for key in ['lores', 'hires', 'adaptive']:
        if key in results:
            r_sim, _, tt_sim = filter_by_angle(results[key], theta_ana)
            ax.scatter(r_sim/a, tt_sim/1e6, c=results[key]['color'], alpha=0.4, s=15,
                      marker=results[key]['marker'], label=results[key]['name'])
    
    ax.set_xlabel('r/a (normalized radius)', fontsize=12)
    ax.set_ylabel('σ_θθ [MPa]', fontsize=12)
    ax.set_title(f'Tangential (Hoop) Stress (θ = {angle_deg}°)', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.2, 5)
    
    plt.suptitle(f'3-Way Resolution Comparison at θ = {angle_deg}°', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_wall_stress_3way(results, tunnel_radius, sigma_v, sigma_h, output_path):
    """Plot hoop stress at tunnel wall vs angle for all three."""
    
    a = tunnel_radius
    
    # Analytical wall stress
    theta_ana = np.linspace(0, 2*np.pi, 361)
    sigma_tt_wall = []
    for theta in theta_ana:
        # Use r=1.25a for analytical (center of the analysis zone 1.2a-1.3a)
        _, sig_tt, _ = kirsch_stress_polar(a * 1.25, theta, a, sigma_v, sigma_h)
        sigma_tt_wall.append(sig_tt)
    sigma_tt_wall = np.array(sigma_tt_wall)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(np.degrees(theta_ana), sigma_tt_wall/1e6, 'k-', linewidth=2.5, 
            label='Kirsch analytical (r=1.25a)')
    
    # Extract near-wall particles for each resolution (using 1.2a-1.5a zone)
    for key in ['lores', 'hires', 'adaptive']:
        if key in results:
            res = results[key]
            wall_mask = (res['r'] < 1.5 * a) & (res['r'] >= 1.2 * a)
            if np.sum(wall_mask) > 0:
                theta_wall = np.degrees(res['theta'][wall_mask])
                stress_wall = res['sim_sigma_tt'][wall_mask] / 1e6
                ax.scatter(theta_wall, stress_wall, c=res['color'], alpha=0.5, s=20,
                          marker=res['marker'], label=res['name'])
    
    ax.set_xlabel('Angle θ from vertical [degrees]', fontsize=12)
    ax.set_ylabel('Hoop stress σ_θθ [MPa]', fontsize=12)
    ax.set_title('Hoop Stress Near Tunnel Wall (r ≈ 1.2-1.5a)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Add vertical lines for reference angles
    for angle in [0, 90, 180, 270]:
        ax.axvline(angle, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_bars_3way(results, output_path):
    """Bar chart comparing errors across resolutions."""
    
    keys = [k for k in ['lores', 'hires', 'adaptive'] if k in results]
    n_res = len(keys)
    
    if n_res == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(n_res)
    width = 0.35
    
    # Mean errors
    ax = axes[0]
    err_rr = [results[k]['mean_rel_err_rr'] * 100 for k in keys]
    err_tt = [results[k]['mean_rel_err_tt'] * 100 for k in keys]
    
    bars1 = ax.bar(x - width/2, err_rr, width, label='Radial σ_rr', color='steelblue')
    bars2 = ax.bar(x + width/2, err_tt, width, label='Hoop σ_θθ', color='coral')
    
    ax.set_ylabel('Mean Relative Error [%]', fontsize=12)
    ax.set_title('Mean Relative Error by Resolution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([results[k]['name'] for k in keys], fontsize=10, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Max errors
    ax = axes[1]
    err_rr = [results[k]['max_rel_err_rr'] * 100 for k in keys]
    err_tt = [results[k]['max_rel_err_tt'] * 100 for k in keys]
    
    bars1 = ax.bar(x - width/2, err_rr, width, label='Radial σ_rr', color='steelblue')
    bars2 = ax.bar(x + width/2, err_tt, width, label='Hoop σ_θθ', color='coral')
    
    ax.set_ylabel('Max Relative Error [%]', fontsize=12)
    ax.set_title('Maximum Relative Error by Resolution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([results[k]['name'] for k in keys], fontsize=10, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Error Comparison Across Mesh Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_2d_stress_3way(results, tunnel_radius, tunnel_center, output_path):
    """2D hoop stress field for each resolution side by side."""
    
    keys = [k for k in ['lores', 'hires', 'adaptive'] if k in results]
    n_res = len(keys)
    
    if n_res == 0:
        return
    
    fig, axes = plt.subplots(1, n_res, figsize=(5*n_res, 5))
    if n_res == 1:
        axes = [axes]
    
    # Find global color range
    all_stress = []
    for k in keys:
        all_stress.extend(results[k]['sim_sigma_tt'])
    vmin, vmax = np.percentile(all_stress, [2, 98])
    vmin, vmax = vmin/1e6, vmax/1e6
    
    for ax, key in zip(axes, keys):
        res = results[key]
        
        # Get particle positions from 'positions' array (shape N x 3)
        pos = res['positions']
        x_rel = pos[:, 0] - tunnel_center[0]
        z_rel = pos[:, 2] - tunnel_center[2]
        stress = res['sim_sigma_tt'] / 1e6
        
        sc = ax.scatter(x_rel, z_rel, c=stress, s=2, cmap='RdBu_r', 
                       vmin=vmin, vmax=vmax, alpha=0.8)
        
        # Draw tunnel
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(tunnel_radius * np.cos(theta), tunnel_radius * np.sin(theta), 
               'k-', linewidth=2)
        ax.fill(tunnel_radius * np.cos(theta), tunnel_radius * np.sin(theta), 
               'white', zorder=2)
        
        ax.set_xlabel('x - x_c [m]', fontsize=10)
        ax.set_ylabel('z - z_c [m]', fontsize=10)
        ax.set_title(res['name'], fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.grid(True, alpha=0.3)
    
    # Common colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.15, aspect=30)
    cbar.set_label('Hoop Stress σ_θθ [MPa]', fontsize=11)
    
    plt.suptitle('2D Hoop Stress Field Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_convergence(results, output_path):
    """Plot error vs particle count to show convergence."""
    
    keys = [k for k in ['lores', 'hires', 'adaptive'] if k in results]
    
    if len(keys) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_particles = []
    err_rr = []
    err_tt = []
    names = []
    colors = []
    
    for k in keys:
        n = len(results[k]['r'])
        n_particles.append(n)
        err_rr.append(results[k]['mean_rel_err_rr'] * 100)
        err_tt.append(results[k]['mean_rel_err_tt'] * 100)
        names.append(results[k]['name'])
        colors.append(results[k]['color'])
    
    # Plot with log scale on x
    ax.scatter(n_particles, err_rr, s=150, c=colors, marker='o', 
               label='Radial σ_rr', edgecolors='black', linewidth=1.5, zorder=5)
    ax.scatter(n_particles, err_tt, s=150, c=colors, marker='s', 
               label='Hoop σ_θθ', edgecolors='black', linewidth=1.5, zorder=5)
    
    # Connect points
    ax.plot(n_particles, err_rr, 'k--', alpha=0.5, zorder=1)
    ax.plot(n_particles, err_tt, 'k--', alpha=0.5, zorder=1)
    
    # Annotate
    for i, name in enumerate(names):
        ax.annotate(name, (n_particles[i], max(err_rr[i], err_tt[i])),
                   xytext=(5, 10), textcoords='offset points', fontsize=9)
    
    ax.set_xscale('log')
    ax.set_xlabel('Number of Particles (near tunnel)', fontsize=12)
    ax.set_ylabel('Mean Relative Error [%]', fontsize=12)
    ax.set_title('Convergence: Error vs Particle Count', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', label='Radial σ_rr'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', label='Hoop σ_θθ'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    args = parse_args()
    
    results = compare_three_resolutions(
        lores_dir=args.lores_dir,
        hires_dir=args.hires_dir,
        adaptive_dir=args.adaptive_dir,
        output_dir=args.output_dir,
        tunnel_radius=args.tunnel_radius,
        tunnel_center=tuple(args.tunnel_center),
        density=args.density,
        gravity=args.gravity,
        z_top=args.z_top,
        K0=args.K0
    )
    
    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Kirsch Analytical Solution Comparison for Tunnel Benchmark

This script loads simulation output VTP files and compares the stress state
around the circular tunnel to the Kirsch analytical solution.

Kirsch Solution (plane strain, circular tunnel in infinite elastic medium):
    σ_rr = (σ_v + σ_h)/2 * (1 - a²/r²) + (σ_v - σ_h)/2 * (1 - 4a²/r² + 3a⁴/r⁴) * cos(2θ)
    σ_θθ = (σ_v + σ_h)/2 * (1 + a²/r²) - (σ_v - σ_h)/2 * (1 + 3a⁴/r⁴) * cos(2θ)
    σ_rθ = -(σ_v - σ_h)/2 * (1 + 2a²/r² - 3a⁴/r⁴) * sin(2θ)

Where:
    a = tunnel radius
    r = radial distance from tunnel center
    θ = angle from horizontal (θ=0 is springline, θ=90° is crown)
    σ_v = vertical far-field stress (compression negative)
    σ_h = horizontal far-field stress

Author: MPMWarpYDW Benchmark Suite
"""

import numpy as np
import os
import sys
import glob
import argparse
from pathlib import Path

# Check for vtk availability
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("Warning: VTK not installed. Cannot read VTP files directly.")
    print("Install with: pip install vtk")

# Check for pyvista (nicer interface to VTK)
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


# =============================================================================
# Kirsch Analytical Solution Functions
# =============================================================================

def kirsch_stress_polar(r, theta, a, sigma_v, sigma_h):
    """
    Compute Kirsch analytical stress in polar coordinates.
    
    Parameters:
    -----------
    r : float or array
        Radial distance from tunnel center [m]
    theta : float or array
        Angle measured from VERTICAL (Z-axis, direction of sigma_v) [radians]
        θ=0 is crown/invert (top/bottom), θ=90° is springline (sides)
    a : float
        Tunnel radius [m]
    sigma_v : float
        Vertical far-field stress [Pa] (magnitude, positive value)
    sigma_h : float
        Horizontal far-field stress [Pa] (magnitude, positive value)
        
    Returns:
    --------
    sigma_rr : radial stress
    sigma_tt : tangential (hoop) stress  
    sigma_rt : shear stress
    
    Note: Returns compression as NEGATIVE (Cauchy convention)
    
    At wall (r=a):
        Crown (θ=0°):      σ_θθ = 3σ_h - σ_v  (lower concentration when σ_v > σ_h)
        Springline (θ=90°): σ_θθ = 3σ_v - σ_h (higher concentration when σ_v > σ_h)
    """
    # Ensure r > a to avoid division issues at the wall
    r = np.maximum(r, a * 1.001)
    
    # Ratios
    a2_r2 = (a / r) ** 2
    a4_r4 = (a / r) ** 4
    
    # Mean stress and deviatoric stress magnitudes
    sigma_mean = (sigma_v + sigma_h) / 2
    sigma_dev = (sigma_v - sigma_h) / 2
    
    # Kirsch solution with θ measured from VERTICAL (Z-axis)
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    
    # Radial stress
    sigma_rr = sigma_mean * (1 - a2_r2) + sigma_dev * (1 - 4*a2_r2 + 3*a4_r4) * cos2t
    
    # Tangential (hoop) stress
    sigma_tt = sigma_mean * (1 + a2_r2) - sigma_dev * (1 + 3*a4_r4) * cos2t
    
    # Shear stress
    sigma_rt = -sigma_dev * (1 + 2*a2_r2 - 3*a4_r4) * sin2t
    
    # Return with compression NEGATIVE
    return -sigma_rr, -sigma_tt, -sigma_rt


def kirsch_stress_cartesian(x, z, xc, zc, a, sigma_v, sigma_h):
    """
    Compute Kirsch stress and convert to Cartesian tensor components.
    
    Parameters:
    -----------
    x, z : float or array
        Cartesian coordinates [m]
    xc, zc : float
        Tunnel center coordinates [m]
    a : float
        Tunnel radius [m]
    sigma_v, sigma_h : float
        Far-field stresses [Pa] (magnitude, positive)
        
    Returns:
    --------
    sigma_xx, sigma_zz, sigma_xz : Cartesian stress components
    """
    # Convert to polar
    dx = x - xc
    dz = z - zc
    r = np.sqrt(dx**2 + dz**2)
    # θ measured from +Z axis (vertical, σ_v direction)
    # θ=0 is crown/invert, θ=90° is springline
    theta = np.arctan2(dx, dz)  # arctan2(x, z) measures from +Z axis
    
    # Get polar stresses
    sigma_rr, sigma_tt, sigma_rt = kirsch_stress_polar(r, theta, a, sigma_v, sigma_h)
    
    # Transform to Cartesian
    # Note: theta is now from +Z, so cos(theta) aligns with Z, sin(theta) with X
    cos_t = np.cos(theta)  # Z component of radial direction
    sin_t = np.sin(theta)  # X component of radial direction
    cos2_t = cos_t ** 2
    sin2_t = sin_t ** 2
    
    # σ_xx (horizontal): radial component in X direction
    sigma_xx = sigma_rr * sin2_t + sigma_tt * cos2_t + 2 * sigma_rt * cos_t * sin_t
    # σ_zz (vertical): radial component in Z direction  
    sigma_zz = sigma_rr * cos2_t + sigma_tt * sin2_t - 2 * sigma_rt * cos_t * sin_t
    # σ_xz (shear)
    sigma_xz = (sigma_rr - sigma_tt) * cos_t * sin_t + sigma_rt * (cos2_t - sin2_t)
    
    return sigma_xx, sigma_zz, sigma_xz, r, theta


# =============================================================================
# VTP File Reading
# =============================================================================

def read_vtp_file(filepath):
    """
    Read VTP file and extract particle positions and stress data.
    
    Returns:
    --------
    positions : ndarray (N, 3)
    stress : ndarray (N, 6) - Voigt notation [xx, yy, zz, xy, xz, yz]
    """
    if HAS_PYVISTA:
        mesh = pv.read(filepath)
        positions = np.array(mesh.points)
        
        # Look for stress data
        if 'stress' in mesh.point_data:
            stress = np.array(mesh.point_data['stress'])
        elif 'particle_stress' in mesh.point_data:
            stress = np.array(mesh.point_data['particle_stress'])
        else:
            # Try to find any array with 6 components
            stress = None
            for name in mesh.point_data.keys():
                arr = mesh.point_data[name]
                if hasattr(arr, 'shape') and len(arr.shape) == 2 and arr.shape[1] == 6:
                    stress = np.array(arr)
                    print(f"Found stress data in '{name}'")
                    break
            if stress is None:
                print(f"Available arrays: {list(mesh.point_data.keys())}")
                raise ValueError("Could not find stress data in VTP file")
        
        return positions, stress
        
    elif HAS_VTK:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        polydata = reader.GetOutput()
        points = polydata.GetPoints()
        positions = vtk_to_numpy(points.GetData())
        
        # Find stress array
        point_data = polydata.GetPointData()
        stress = None
        for i in range(point_data.GetNumberOfArrays()):
            name = point_data.GetArrayName(i)
            arr = point_data.GetArray(i)
            if arr.GetNumberOfComponents() == 6:
                stress = vtk_to_numpy(arr)
                print(f"Found stress data in '{name}'")
                break
        
        if stress is None:
            raise ValueError("Could not find stress data in VTP file")
        
        return positions, stress
    
    else:
        raise ImportError("Neither pyvista nor vtk available. Install with: pip install pyvista")


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_tunnel_stress(positions, stress, tunnel_center, tunnel_radius,
                         sigma_v, sigma_h, y_slice_tolerance=None, 
                         density=2600.0, gravity=9.81, z_top=200.0, K0=0.5):
    """
    Analyze stress around tunnel and compare to Kirsch solution.
    
    Parameters:
    -----------
    positions : ndarray (N, 3)
    stress : ndarray (N, 6) - Voigt [xx, yy, zz, xy, xz, yz]
    tunnel_center : tuple (xc, yc, zc)
    tunnel_radius : float
    sigma_v, sigma_h : float - far-field stresses at tunnel center (magnitude)
    y_slice_tolerance : float - thickness of Y slice to analyze (if None, use all)
    density : float - material density for depth-varying stress correction
    gravity : float - gravitational acceleration
    z_top : float - Z coordinate of top surface
    K0 : float - horizontal stress ratio
    
    Returns:
    --------
    results : dict with comparison data
    """
    xc, yc, zc = tunnel_center
    a = tunnel_radius
    
    # Filter particles by Y coordinate (for plane strain analysis)
    # Use tight tolerance to select only the central Y-slice
    if y_slice_tolerance is not None:
        y_mask = np.abs(positions[:, 1] - yc) < y_slice_tolerance
        # Report unique Y values in selection
        y_unique = np.unique(np.round(positions[y_mask, 1], 2))
        print(f"  Y-slice filter: tolerance={y_slice_tolerance:.2f}m, selected {np.sum(y_mask)} particles")
        print(f"    Unique Y values in slice: {y_unique}")
    else:
        y_mask = np.ones(len(positions), dtype=bool)
    
    pos_filtered = positions[y_mask]
    stress_filtered = stress[y_mask]
    
    # Compute radial distance and angle from tunnel center
    dx = pos_filtered[:, 0] - xc
    dz = pos_filtered[:, 2] - zc
    r = np.sqrt(dx**2 + dz**2)
    # θ measured from +Z axis (vertical, σ_v direction)
    # θ=0 is crown/invert (top/bottom), θ=90° is springline (sides)
    theta = np.arctan2(dx, dz)  # arctan2(x, z) measures from +Z axis
    
    # Filter particles near tunnel (1.2a to 5a radius)
    # Exclude very close particles (r < 1.2a) where boundary artifacts dominate
    r_min_factor = 1.2  # Skip the first ~2 particle layers from wall
    near_tunnel_mask = (r >= a * r_min_factor) & (r <= a * 5.0)
    print(f"  Radial filter: {r_min_factor}a to 5.0a, selected {np.sum(near_tunnel_mask)} particles for analysis")
    
    r_near = r[near_tunnel_mask]
    theta_near = theta[near_tunnel_mask]
    pos_near = pos_filtered[near_tunnel_mask]
    stress_near = stress_filtered[near_tunnel_mask]
    
    # Extract stress components (Voigt notation: xx, yy, zz, xy, xz, yz)
    # Note: We want xx (radial in springline dir), zz (vertical), xz (shear)
    sim_sigma_xx = stress_near[:, 0]  # σ_xx
    sim_sigma_zz = stress_near[:, 2]  # σ_zz
    sim_sigma_xz = stress_near[:, 4]  # σ_xz
    
    # DEBUG: Check far-field stress values in simulation
    # Particles far from tunnel (r > 4a) should have geostatic stress
    far_mask = r_near > 4 * a
    if np.any(far_mask):
        print(f"\n  DEBUG: Far-field stress check (r > 4a, {np.sum(far_mask)} particles):")
        print(f"    Mean σ_xx (horizontal): {np.mean(sim_sigma_xx[far_mask])/1e6:.3f} MPa")
        print(f"    Mean σ_zz (vertical):   {np.mean(sim_sigma_zz[far_mask])/1e6:.3f} MPa")
        print(f"    Expected σ_h ≈ {K0 * density * gravity * (z_top - np.mean(pos_near[far_mask, 2]))/1e6:.3f} MPa")
        print(f"    Expected σ_v ≈ {density * gravity * (z_top - np.mean(pos_near[far_mask, 2]))/1e6:.3f} MPa")
    
    # Compute LOCAL far-field stress for each particle based on its depth
    # This accounts for geostatic stress variation with depth
    z_particles = pos_near[:, 2]
    depth_particles = z_top - z_particles
    sigma_v_local = density * gravity * depth_particles  # magnitude (positive)
    sigma_h_local = K0 * sigma_v_local
    
    # Compute analytical Kirsch solution for each particle using LOCAL far-field stress
    # Note: We need to compute Kirsch per-particle since far-field stress varies
    ana_sigma_rr = np.zeros(len(pos_near))
    ana_sigma_tt = np.zeros(len(pos_near))
    ana_sigma_rt = np.zeros(len(pos_near))
    
    for i in range(len(pos_near)):
        ana_sigma_rr[i], ana_sigma_tt[i], ana_sigma_rt[i] = kirsch_stress_polar(
            r_near[i], theta_near[i], a, sigma_v_local[i], sigma_h_local[i]
        )
    
    # Transform simulated stress to polar coordinates
    # theta is measured from +Z axis, so:
    # cos(theta) = radial component in Z direction
    # sin(theta) = radial component in X direction
    cos_t = np.cos(theta_near)
    sin_t = np.sin(theta_near)
    
    # Radial stress: project Cartesian stress onto radial direction
    # σ_rr = σ_xx*sin²θ + σ_zz*cos²θ + 2*σ_xz*sinθ*cosθ
    sim_sigma_rr = (sim_sigma_xx * sin_t**2 + sim_sigma_zz * cos_t**2 + 
                   2 * sim_sigma_xz * sin_t * cos_t)
    # Tangential stress: perpendicular to radial
    # σ_θθ = σ_xx*cos²θ + σ_zz*sin²θ - 2*σ_xz*sinθ*cosθ  
    sim_sigma_tt = (sim_sigma_xx * cos_t**2 + sim_sigma_zz * sin_t**2 - 
                   2 * sim_sigma_xz * sin_t * cos_t)
    
    # Compute errors
    err_rr = sim_sigma_rr - ana_sigma_rr
    err_tt = sim_sigma_tt - ana_sigma_tt
    
    # Relative errors (avoid division by zero)
    rel_err_rr = np.abs(err_rr) / (np.abs(ana_sigma_rr) + 1e-10)
    rel_err_tt = np.abs(err_tt) / (np.abs(ana_sigma_tt) + 1e-10)
    
    results = {
        'r': r_near,
        'theta': theta_near,
        'positions': pos_near,
        'sim_sigma_rr': sim_sigma_rr,
        'sim_sigma_tt': sim_sigma_tt,
        'ana_sigma_rr': ana_sigma_rr,
        'ana_sigma_tt': ana_sigma_tt,
        'err_rr': err_rr,
        'err_tt': err_tt,
        'rel_err_rr': rel_err_rr,
        'rel_err_tt': rel_err_tt,
        'mean_rel_err_rr': np.mean(rel_err_rr),
        'mean_rel_err_tt': np.mean(rel_err_tt),
        'max_rel_err_rr': np.max(rel_err_rr),
        'max_rel_err_tt': np.max(rel_err_tt),
        'sigma_v': sigma_v,
        'sigma_h': sigma_h,
        'tunnel_radius': a,
        'tunnel_center': tunnel_center,
        # Parameters for depth-varying stress in plots
        'density': density,
        'gravity': gravity,
        'z_top': z_top,
        'K0': K0,
    }
    
    return results


def plot_radial_profile(results, output_path=None, angle_deg=0):
    """
    Plot radial stress profile at a given angle.
    
    Parameters:
    -----------
    results : dict from analyze_tunnel_stress
    output_path : str - save figure path
    angle_deg : float - angle to plot (0=crown, 90=springline)
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    a = results['tunnel_radius']
    sigma_v = results['sigma_v']
    sigma_h = results['sigma_h']
    
    # Get depth-varying parameters from results
    density = results.get('density', 2600.0)
    gravity = results.get('gravity', 9.81)
    z_top = results.get('z_top', 200.0)
    K0 = results.get('K0', 0.5)
    tunnel_center = results.get('tunnel_center', (100.0, 2.0, 100.0))
    zc = tunnel_center[2]
    
    # Analytical profile with DEPTH-VARYING far-field stress
    r_ana = np.linspace(a, 5*a, 100)
    theta_ana = np.radians(angle_deg)
    
    # Compute Z position along this radial direction
    # θ=0 is vertical (+Z from center), so z = zc + r*cos(θ)
    z_ana = zc + r_ana * np.cos(theta_ana)
    depth_ana = z_top - z_ana
    sigma_v_ana = density * gravity * depth_ana
    sigma_h_ana = K0 * sigma_v_ana
    
    # Compute Kirsch for each point with local far-field stress
    ana_rr = np.zeros_like(r_ana)
    ana_tt = np.zeros_like(r_ana)
    for i in range(len(r_ana)):
        ana_rr[i], ana_tt[i], _ = kirsch_stress_polar(r_ana[i], theta_ana, a, 
                                                       sigma_v_ana[i], sigma_h_ana[i])
    
    # Filter simulation data near this angle (±15°)
    angle_tol = np.radians(15)
    theta_target = np.radians(angle_deg)
    
    # Handle angle wrapping
    angle_diff = np.abs(np.mod(results['theta'] - theta_target + np.pi, 2*np.pi) - np.pi)
    mask = angle_diff < angle_tol
    
    r_sim = results['r'][mask]
    sim_rr = results['sim_sigma_rr'][mask]
    sim_tt = results['sim_sigma_tt'][mask]
    ana_rr_pts = results['ana_sigma_rr'][mask]
    ana_tt_pts = results['ana_sigma_tt'][mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Radial stress
    ax = axes[0]
    ax.plot(r_ana/a, ana_rr/1e6, 'b-', linewidth=2, label='Kirsch analytical')
    ax.scatter(r_sim/a, sim_rr/1e6, c='r', alpha=0.5, s=20, label='MPM simulation')
    ax.set_xlabel('r/a (normalized radius)')
    ax.set_ylabel('σ_rr [MPa]')
    ax.set_title(f'Radial Stress (θ = {angle_deg}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    
    # Tangential stress
    ax = axes[1]
    ax.plot(r_ana/a, ana_tt/1e6, 'b-', linewidth=2, label='Kirsch analytical')
    ax.scatter(r_sim/a, sim_tt/1e6, c='r', alpha=0.5, s=20, label='MPM simulation')
    ax.set_xlabel('r/a (normalized radius)')
    ax.set_ylabel('σ_θθ [MPa]')
    ax.set_title(f'Tangential (Hoop) Stress (θ = {angle_deg}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 5)
    
    plt.suptitle(f'Kirsch Solution Comparison at θ = {angle_deg}°\n'
                f'σ_v = {sigma_v/1e6:.1f} MPa, σ_h = {sigma_h/1e6:.1f} MPa, a = {a:.1f} m')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_angular_profile(results, output_path=None, r_ratio=1.1):
    """
    Plot stress around the tunnel wall at a given r/a ratio.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    a = results['tunnel_radius']
    sigma_v = results['sigma_v']
    sigma_h = results['sigma_h']
    r_target = r_ratio * a
    
    # Analytical profile
    theta_ana = np.linspace(-np.pi, np.pi, 180)
    ana_rr, ana_tt, _ = kirsch_stress_polar(r_target, theta_ana, a, sigma_v, sigma_h)
    
    # Filter simulation data near this radius (±10%)
    r_tol = 0.1 * a
    mask = np.abs(results['r'] - r_target) < r_tol
    
    theta_sim = results['theta'][mask]
    sim_rr = results['sim_sigma_rr'][mask]
    sim_tt = results['sim_sigma_tt'][mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})
    
    # Radial stress
    ax = axes[0]
    ax.plot(theta_ana, ana_rr/1e6, 'b-', linewidth=2, label='Kirsch')
    ax.scatter(theta_sim, sim_rr/1e6, c='r', alpha=0.5, s=20, label='MPM')
    ax.set_title(f'Radial Stress σ_rr [MPa]')
    ax.legend(loc='upper right')
    
    # Tangential stress
    ax = axes[1]
    ax.plot(theta_ana, ana_tt/1e6, 'b-', linewidth=2, label='Kirsch')
    ax.scatter(theta_sim, sim_tt/1e6, c='r', alpha=0.5, s=20, label='MPM')
    ax.set_title(f'Tangential Stress σ_θθ [MPa]')
    ax.legend(loc='upper right')
    
    plt.suptitle(f'Stress Distribution at r/a = {r_ratio:.2f}\n'
                f'σ_v = {sigma_v/1e6:.1f} MPa, σ_h = {sigma_h/1e6:.1f} MPa')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_2d_comparison(results, output_path=None):
    """
    Plot 2D XZ slice showing stress field comparison.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    a = results['tunnel_radius']
    xc, _, zc = results['tunnel_center']
    
    pos = results['positions']
    sim_tt = results['sim_sigma_tt']
    ana_tt = results['ana_sigma_tt']
    err_tt = results['err_tt']
    
    # Compute common colorbar range for simulated and analytical
    all_stress = np.concatenate([sim_tt, ana_tt]) / 1e6
    vmin_stress = np.min(all_stress)
    vmax_stress = np.max(all_stress)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Simulated hoop stress
    ax = axes[0]
    sc = ax.scatter(pos[:, 0], pos[:, 2], c=sim_tt/1e6, cmap='coolwarm', s=10,
                   vmin=vmin_stress, vmax=vmax_stress)
    circle = plt.Circle((xc, zc), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Simulated σ_θθ [MPa]')
    plt.colorbar(sc, ax=ax)
    
    # Analytical hoop stress
    ax = axes[1]
    sc = ax.scatter(pos[:, 0], pos[:, 2], c=ana_tt/1e6, cmap='coolwarm', s=10,
                   vmin=vmin_stress, vmax=vmax_stress)
    circle = plt.Circle((xc, zc), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Kirsch σ_θθ [MPa]')
    plt.colorbar(sc, ax=ax)
    
    # Error
    ax = axes[2]
    vmax = np.percentile(np.abs(err_tt/1e6), 95)
    sc = ax.scatter(pos[:, 0], pos[:, 2], c=err_tt/1e6, cmap='RdBu', s=10,
                   vmin=-vmax, vmax=vmax)
    circle = plt.Circle((xc, zc), a, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Error (Sim - Kirsch) [MPa]')
    plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("KIRSCH SOLUTION COMPARISON SUMMARY")
    print("="*60)
    print(f"Tunnel radius: {results['tunnel_radius']:.2f} m")
    print(f"Far-field stresses: σ_v = {results['sigma_v']/1e6:.2f} MPa, "
          f"σ_h = {results['sigma_h']/1e6:.2f} MPa")
    print(f"K₀ = σ_h/σ_v = {results['sigma_h']/results['sigma_v']:.2f}")
    print(f"\nNumber of particles analyzed: {len(results['r'])}")
    print(f"Radial range: {results['r'].min()/results['tunnel_radius']:.2f}a to "
          f"{results['r'].max()/results['tunnel_radius']:.2f}a")
    print(f"\nMean relative error (radial stress): {results['mean_rel_err_rr']*100:.2f}%")
    print(f"Mean relative error (hoop stress):   {results['mean_rel_err_tt']*100:.2f}%")
    print(f"Max relative error (radial stress):  {results['max_rel_err_rr']*100:.2f}%")
    print(f"Max relative error (hoop stress):    {results['max_rel_err_tt']*100:.2f}%")
    
    # Wall stress concentration
    a = results['tunnel_radius']
    sigma_v = results['sigma_v']
    sigma_h = results['sigma_h']
    
    # Analytical wall stresses
    # θ=0 is crown (top/bottom), θ=90° is springline (sides)
    _, wall_tt_crown, _ = kirsch_stress_polar(a, 0, a, sigma_v, sigma_h)
    _, wall_tt_springline, _ = kirsch_stress_polar(a, np.pi/2, a, sigma_v, sigma_h)
    
    print(f"\nAnalytical wall stress (crown, θ=0°):      σ_θθ = {wall_tt_crown/1e6:.2f} MPa")
    print(f"Analytical wall stress (springline, θ=90°): σ_θθ = {wall_tt_springline/1e6:.2f} MPa")
    
    # Find simulated wall stress (particles closest to wall)
    wall_mask = results['r'] < 1.2 * a
    if np.any(wall_mask):
        # Crown (θ near 0, i.e., cos(θ) near 1, meaning Z direction)
        crown_mask = wall_mask & (np.abs(np.cos(results['theta'])) > 0.9)
        if np.any(crown_mask):
            sim_wall_crown = np.mean(results['sim_sigma_tt'][crown_mask])
            print(f"Simulated wall stress (crown):             σ_θθ ≈ {sim_wall_crown/1e6:.2f} MPa")
        
        # Springline (θ near π/2, i.e., sin(θ) near 1, meaning X direction)
        springline_mask = wall_mask & (np.abs(np.sin(results['theta'])) > 0.9)
        if np.any(springline_mask):
            sim_wall_springline = np.mean(results['sim_sigma_tt'][springline_mask])
            print(f"Simulated wall stress (springline):        σ_θθ ≈ {sim_wall_springline/1e6:.2f} MPa")
    
    print("="*60 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare MPM simulation to Kirsch analytical solution')
    parser.add_argument('--vtp', type=str, help='Path to VTP file (or use --output_dir for latest)')
    parser.add_argument('--output_dir', type=str, 
                       default='./benchmarks/tunnelBenchmark/output_elastic',
                       help='Directory containing VTP output files')
    parser.add_argument('--tunnel_radius', type=float, default=10.0,
                       help='Tunnel radius [m]')
    parser.add_argument('--tunnel_center', type=float, nargs=3, default=[100.0, 2.0, 100.0],
                       help='Tunnel center [xc, yc, zc]')
    parser.add_argument('--density', type=float, default=2600.0,
                       help='Material density [kg/m³]')
    parser.add_argument('--K0', type=float, default=0.5,
                       help='Horizontal stress ratio K₀')
    parser.add_argument('--z_top', type=float, default=200.0,
                       help='Top surface Z coordinate [m]')
    parser.add_argument('--gravity', type=float, default=9.81,
                       help='Gravitational acceleration [m/s²]')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='Directory to save plots (if not specified, plots shown interactively)')
    
    args = parser.parse_args()
    
    # Find VTP file
    if args.vtp:
        vtp_file = args.vtp
    else:
        vtp_files = sorted(glob.glob(os.path.join(args.output_dir, '*_particles.vtp')))
        if not vtp_files:
            print(f"No VTP files found in {args.output_dir}")
            print("Run simulation first: python runMPMYDW.py --config ./benchmarks/tunnelBenchmark/config_tunnel_elastic.json")
            sys.exit(1)
        vtp_file = vtp_files[-1]  # Use latest file
        print(f"Using latest VTP file: {vtp_file}")
    
    # Compute far-field stresses
    # At tunnel center depth
    tunnel_center = tuple(args.tunnel_center)
    depth = args.z_top - tunnel_center[2]  # depth below surface
    sigma_v = args.density * args.gravity * depth  # magnitude (positive)
    sigma_h = args.K0 * sigma_v
    
    print(f"\nFar-field stresses at tunnel depth (z = {tunnel_center[2]:.1f} m):")
    print(f"  Depth below surface: {depth:.1f} m")
    print(f"  σ_v = ρgh = {sigma_v/1e6:.3f} MPa")
    print(f"  σ_h = K₀σ_v = {sigma_h/1e6:.3f} MPa")
    
    # Read VTP file
    print(f"\nReading {vtp_file}...")
    positions, stress = read_vtp_file(vtp_file)
    print(f"Loaded {len(positions)} particles")
    
    # Analyze
    print("\nAnalyzing stress around tunnel...")
    print("  (Using depth-varying far-field stress correction)")
    results = analyze_tunnel_stress(
        positions, stress, tunnel_center, args.tunnel_radius,
        sigma_v, sigma_h, y_slice_tolerance=2.5,
        density=args.density, gravity=args.gravity, z_top=args.z_top, K0=args.K0
    )
    
    # Print summary
    print_summary(results)
    
    # Generate plots
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
        plot_dir = args.save_plots
    else:
        plot_dir = None
    
    # Radial profiles at different angles
    for angle in [0, 45, 90]:
        if plot_dir:
            path = os.path.join(plot_dir, f'radial_profile_{angle}deg.png')
        else:
            path = None
        plot_radial_profile(results, path, angle_deg=angle)
    
    # Angular profile near wall
    if plot_dir:
        path = os.path.join(plot_dir, 'angular_profile_wall.png')
    else:
        path = None
    plot_angular_profile(results, path, r_ratio=1.2)
    
    # 2D comparison
    if plot_dir:
        path = os.path.join(plot_dir, '2d_comparison.png')
    else:
        path = None
    plot_2d_comparison(results, path)
    
    # Save numerical results
    if plot_dir:
        np.savez(os.path.join(plot_dir, 'kirsch_comparison_results.npz'),
                r=results['r'],
                theta=results['theta'],
                sim_sigma_rr=results['sim_sigma_rr'],
                sim_sigma_tt=results['sim_sigma_tt'],
                ana_sigma_rr=results['ana_sigma_rr'],
                ana_sigma_tt=results['ana_sigma_tt'],
                mean_rel_err_rr=results['mean_rel_err_rr'],
                mean_rel_err_tt=results['mean_rel_err_tt'])
        print(f"Saved results to {os.path.join(plot_dir, 'kirsch_comparison_results.npz')}")
    
    return results


if __name__ == '__main__':
    main()

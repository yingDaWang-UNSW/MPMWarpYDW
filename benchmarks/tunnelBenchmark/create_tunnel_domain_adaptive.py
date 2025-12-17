"""
Create an ADAPTIVE tunnel domain with improved voxelization for Kirsch validation.

This version uses body-fitted sampling near the tunnel with:
1. Polar coordinate sampling near the tunnel wall (conforming to circular boundary)
2. Gradual transition to Cartesian grid in far-field
3. Variable particle density (finer near tunnel, coarser far away)

This improves accuracy compared to simple grid voxelization which creates
a jagged tunnel boundary.

Usage:
    python create_tunnel_domain_adaptive.py [--base_spacing 0.5] [--wall_spacing 0.25]
"""

import numpy as np
import h5py
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Create adaptive tunnel domain')
    parser.add_argument('--base_spacing', type=float, default=1.0,
                        help='Base particle spacing in far-field (m)')
    parser.add_argument('--wall_spacing', type=float, default=0.25,
                        help='Particle spacing at tunnel wall (m)')
    parser.add_argument('--suffix', type=str, default='_adaptive',
                        help='Suffix for output files')
    parser.add_argument('--refinement_radius', type=float, default=3.0,
                        help='Refinement radius as multiple of tunnel radius')
    return parser.parse_args()

def create_adaptive_tunnel_domain(base_spacing=1.0, wall_spacing=0.25, 
                                   refinement_radius_factor=3.0, suffix='_adaptive'):
    """
    Create tunnel domain with body-fitted adaptive sampling.
    
    Strategy:
    1. Near-wall zone (r < 1.5a): Polar sampling with fine angular/radial resolution
    2. Transition zone (1.5a < r < 3a): Polar sampling with coarsening
    3. Far-field zone (r > 3a): Regular Cartesian grid with base spacing
    
    The key improvement is that the tunnel boundary is represented by particles
    placed exactly at radial distance a + wall_spacing/2, giving a smooth
    circular boundary rather than a jagged voxelized one.
    """
    
    # ============================================================
    # DOMAIN PARAMETERS
    # ============================================================
    
    # Tunnel geometry
    tunnel_radius = 10.0  # meters (a in Kirsch equations)
    tunnel_center = np.array([100.0, 2.0, 100.0])  # Center of domain
    
    # Domain size
    Lx = 200.0  # meters
    Ly = 4.0    # meters (thin for plane strain)
    Lz = 200.0  # meters
    
    # Material properties
    E = 10e9        # 10 GPa
    nu = 0.25       # Poisson's ratio
    density = 2600  # kg/m³
    ys = 1e20       # Very high for elastic-only
    
    # Geostatic parameters
    g = 9.81
    K0 = 0.5
    z_surface = Lz
    
    # Zone definitions
    a = tunnel_radius
    r_inner = a + wall_spacing / 2  # First particle layer
    r_transition_start = 1.5 * a    # Start of transition zone
    r_refinement = refinement_radius_factor * a  # End of refinement zone
    
    print(f"Adaptive Tunnel Domain Generator")
    print(f"=" * 60)
    print(f"Tunnel radius: {a}m")
    print(f"Domain: {Lx}m × {Ly}m × {Lz}m")
    print(f"Wall spacing: {wall_spacing}m")
    print(f"Base spacing: {base_spacing}m")
    print(f"Refinement zones:")
    print(f"  Near-wall (r < {r_transition_start:.1f}m): spacing ~{wall_spacing}m")
    print(f"  Transition ({r_transition_start:.1f}m < r < {r_refinement:.1f}m): graded")
    print(f"  Far-field (r > {r_refinement:.1f}m): spacing {base_spacing}m")
    
    # ============================================================
    # Y-COORDINATE LAYERS (same for all zones)
    # ============================================================
    
    # Use wall_spacing in Y for consistency
    ny = max(2, int(Ly / wall_spacing) + 1)
    y_coords = np.linspace(0, Ly, ny)
    print(f"\nY layers: {ny} at y = {y_coords}")
    
    all_positions = []
    all_volumes = []
    
    # ============================================================
    # ZONE 1: NEAR-WALL POLAR SAMPLING (r_inner to r_transition_start)
    # ============================================================
    
    print(f"\nGenerating near-wall zone (polar sampling)...")
    
    # Radial layers with wall_spacing
    r_values = []
    r = r_inner
    while r < r_transition_start:
        r_values.append(r)
        r += wall_spacing
    r_values = np.array(r_values)
    
    # Angular resolution based on arc length = wall_spacing
    # At each radius, n_theta = 2*pi*r / wall_spacing
    
    for r in r_values:
        circumference = 2 * np.pi * r
        n_theta = max(16, int(circumference / wall_spacing))
        theta_values = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        
        # Volume based on annular sector
        dr = wall_spacing
        dtheta = 2 * np.pi / n_theta
        vol = r * dr * dtheta  # Area in XZ plane (per unit Y)
        
        for theta in theta_values:
            x = tunnel_center[0] + r * np.cos(theta)
            z = tunnel_center[2] + r * np.sin(theta)
            
            # Check if within domain
            if 0 <= x <= Lx and 0 <= z <= Lz:
                for y in y_coords:
                    all_positions.append([x, y, z])
                    all_volumes.append(vol * (Ly / ny))  # Scale by Y layer thickness
    
    n_near_wall = len(all_positions)
    print(f"  Near-wall particles: {n_near_wall}")
    
    # ============================================================
    # ZONE 2: TRANSITION ZONE (r_transition_start to r_refinement)
    # ============================================================
    
    print(f"Generating transition zone (graded polar sampling)...")
    
    # Radial spacing increases linearly from wall_spacing to base_spacing
    r = r_transition_start
    dr = wall_spacing
    
    while r < r_refinement:
        # Linear interpolation of spacing
        t = (r - r_transition_start) / (r_refinement - r_transition_start)
        current_spacing = wall_spacing + t * (base_spacing - wall_spacing)
        
        circumference = 2 * np.pi * r
        n_theta = max(16, int(circumference / current_spacing))
        theta_values = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        dtheta = 2 * np.pi / n_theta
        
        vol = r * current_spacing * dtheta
        
        for theta in theta_values:
            x = tunnel_center[0] + r * np.cos(theta)
            z = tunnel_center[2] + r * np.sin(theta)
            
            if 0 <= x <= Lx and 0 <= z <= Lz:
                for y in y_coords:
                    all_positions.append([x, y, z])
                    all_volumes.append(vol * (Ly / ny))
        
        r += current_spacing
    
    n_transition = len(all_positions) - n_near_wall
    print(f"  Transition particles: {n_transition}")
    
    # ============================================================
    # ZONE 3: FAR-FIELD CARTESIAN GRID
    # ============================================================
    
    print(f"Generating far-field zone (Cartesian grid)...")
    
    # Regular grid with base_spacing
    nx = int(Lx / base_spacing) + 1
    nz = int(Lz / base_spacing) + 1
    
    x_grid = np.linspace(0, Lx, nx)
    z_grid = np.linspace(0, Lz, nz)
    
    far_field_vol = base_spacing**2 * (Ly / ny)
    
    for x in x_grid:
        for z in z_grid:
            # Distance from tunnel center
            dx = x - tunnel_center[0]
            dz = z - tunnel_center[2]
            r = np.sqrt(dx**2 + dz**2)
            
            # Only add if outside refinement zone AND outside tunnel
            if r >= r_refinement and r > a + base_spacing/2:
                for y in y_coords:
                    all_positions.append([x, y, z])
                    all_volumes.append(far_field_vol)
    
    n_far_field = len(all_positions) - n_near_wall - n_transition
    print(f"  Far-field particles: {n_far_field}")
    
    # ============================================================
    # CONVERT TO ARRAYS
    # ============================================================
    
    positions = np.array(all_positions, dtype=np.float32)
    volumes = np.array(all_volumes, dtype=np.float32)
    n_particles = len(positions)
    
    print(f"\nTotal particles: {n_particles}")
    print(f"  Near-wall: {n_near_wall} ({100*n_near_wall/n_particles:.1f}%)")
    print(f"  Transition: {n_transition} ({100*n_transition/n_particles:.1f}%)")
    print(f"  Far-field: {n_far_field} ({100*n_far_field/n_particles:.1f}%)")
    
    # ============================================================
    # MATERIAL PROPERTIES
    # ============================================================
    
    E_array = np.full(n_particles, E, dtype=np.float32)
    nu_array = np.full(n_particles, nu, dtype=np.float32)
    density_array = np.full(n_particles, density, dtype=np.float32)
    ys_array = np.full(n_particles, ys, dtype=np.float32)
    
    # ============================================================
    # KIRSCH REFERENCE VALUES
    # ============================================================
    
    depth = z_surface - tunnel_center[2]
    sigma_v = -density * g * depth
    sigma_h = K0 * sigma_v
    
    sigma_theta_crown = 3 * sigma_h - sigma_v
    sigma_theta_springline = 3 * sigma_v - sigma_h
    
    print(f"\n" + "=" * 60)
    print("KIRSCH ANALYTICAL SOLUTION (Reference)")
    print("=" * 60)
    print(f"Far-field stresses at tunnel center (depth={depth:.1f}m):")
    print(f"  σ_v = {sigma_v/1e6:.3f} MPa")
    print(f"  σ_h = {sigma_h/1e6:.3f} MPa")
    print(f"Hoop stress at tunnel wall:")
    print(f"  Crown (θ=0°):      σ_θθ = {sigma_theta_crown/1e6:.3f} MPa")
    print(f"  Springline (θ=90°): σ_θθ = {sigma_theta_springline/1e6:.3f} MPa")
    
    # ============================================================
    # SAVE TO HDF5
    # ============================================================
    
    output_file = f"tunnel_domain_elastic{suffix}.h5"
    print(f"\n" + "=" * 60)
    print(f"Saving to: {output_file}")
    
    with h5py.File(output_file, "w") as f:
        f.create_dataset("x", data=positions.T.astype(np.float32))
        f.create_dataset("particle_volume", data=volumes)
        f.create_dataset("E", data=E_array)
        f.create_dataset("nu", data=nu_array)
        f.create_dataset("density", data=density_array)
        f.create_dataset("ys", data=ys_array)
        
        f.attrs["tunnel_radius"] = tunnel_radius
        f.attrs["tunnel_center_x"] = tunnel_center[0]
        f.attrs["tunnel_center_y"] = tunnel_center[1]
        f.attrs["tunnel_center_z"] = tunnel_center[2]
        f.attrs["domain_Lx"] = Lx
        f.attrs["domain_Ly"] = Ly
        f.attrs["domain_Lz"] = Lz
        f.attrs["wall_spacing"] = wall_spacing
        f.attrs["base_spacing"] = base_spacing
        f.attrs["adaptive"] = True
        f.attrs["E"] = E
        f.attrs["nu"] = nu
        f.attrs["density"] = density
        f.attrs["K0"] = K0
        f.attrs["sigma_v_at_center"] = sigma_v
        f.attrs["sigma_h_at_center"] = sigma_h
    
    print(f"  Positions: shape {positions.T.shape}")
    print(f"  Volume range: {volumes.min():.4f} to {volumes.max():.4f} m³")
    print(f"  Total particles: {n_particles}")
    
    # Save reference data
    ref_file = f"kirsch_reference_data{suffix}.npz"
    angles_deg = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180])
    r_ratios = np.array([1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0])
    
    np.savez(ref_file,
             tunnel_radius=tunnel_radius,
             tunnel_center=tunnel_center,
             angles_deg=angles_deg,
             r_ratios=r_ratios,
             sigma_v=sigma_v,
             sigma_h=sigma_h,
             K0=K0,
             E=E,
             nu=nu,
             density=density,
             z_surface=z_surface,
             wall_spacing=wall_spacing,
             base_spacing=base_spacing)
    
    print(f"Saved reference data to: {ref_file}")
    print(f"\n" + "=" * 60)
    print("Domain creation complete!")
    print("=" * 60)
    
    return output_file, n_particles

if __name__ == "__main__":
    args = parse_args()
    create_adaptive_tunnel_domain(
        base_spacing=args.base_spacing,
        wall_spacing=args.wall_spacing,
        refinement_radius_factor=args.refinement_radius,
        suffix=args.suffix
    )

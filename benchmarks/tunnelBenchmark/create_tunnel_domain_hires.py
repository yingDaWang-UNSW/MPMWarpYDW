"""
Create high-resolution tunnel domain for Kirsch analytical solution validation.

This creates a finer mesh (0.5m spacing) compared to the standard 1.0m spacing,
giving ~20 particles across the tunnel radius instead of ~10.

Higher resolution should:
1. Better capture stress gradients near the tunnel wall
2. Reduce discretization errors
3. Provide smoother stress fields
"""

import numpy as np
import h5py
import os
import argparse

def create_tunnel_domain(spacing=0.5, output_suffix="_hires"):
    """
    Create tunnel domain with specified particle spacing.
    
    Parameters:
    -----------
    spacing : float
        Particle spacing in meters (default 0.5m for high-res)
    output_suffix : str
        Suffix for output filename
    """
    
    # ============================================================
    # DOMAIN PARAMETERS
    # ============================================================
    
    # Tunnel geometry
    tunnel_radius = 10.0  # meters (a in Kirsch equations)
    tunnel_center = np.array([100.0, 2.0, 100.0])  # Center of domain
    
    # Domain size (should be >> tunnel radius for Kirsch validity)
    Lx = 200.0  # meters (20× tunnel radius total width)
    Ly = 4.0    # meters (thin for plane strain)
    Lz = 200.0  # meters (20× tunnel radius total height)
    
    # Material properties (elastic validation - very high yield stress)
    E = 10e9        # 10 GPa - typical rock
    nu = 0.25       # Poisson's ratio
    density = 2600  # kg/m³
    ys = 1e20       # Very high yield stress for elastic-only simulation
    
    # Geostatic parameters
    g = 9.81        # gravity m/s²
    K0 = 0.5        # Lateral earth pressure coefficient
    z_surface = Lz  # Surface at top of domain
    
    # ============================================================
    # GENERATE PARTICLE POSITIONS
    # ============================================================
    
    nx = int(Lx / spacing) + 1
    ny = int(Ly / spacing) + 1
    nz = int(Lz / spacing) + 1
    
    print(f"Tunnel Domain Generator (High Resolution)")
    print(f"=" * 60)
    print(f"Particle spacing: {spacing}m")
    print(f"Particles across tunnel radius: {tunnel_radius / spacing:.0f}")
    print(f"Tunnel radius: {tunnel_radius}m")
    print(f"Domain: {Lx}m × {Ly}m × {Lz}m")
    print(f"Initial grid: {nx} × {ny} × {nz} = {nx*ny*nz:,} particles")
    
    # Generate regular grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # ============================================================
    # REMOVE TUNNEL (CIRCULAR HOLE) - WITH REFINED BOUNDARY
    # ============================================================
    
    # Distance from tunnel axis in XZ plane
    dx_from_center = positions[:, 0] - tunnel_center[0]
    dz_from_center = positions[:, 2] - tunnel_center[2]
    
    # Radial distance from tunnel axis
    r_from_tunnel = np.sqrt(dx_from_center**2 + dz_from_center**2)
    
    # Use refined sub-grid test for particles near tunnel boundary
    # This prevents staircase artifacts on the circular wall
    refine_factor = 4
    fine_spacing = spacing / refine_factor
    
    tunnel_influence_radius = tunnel_radius + spacing * 2
    near_tunnel = r_from_tunnel < tunnel_influence_radius
    
    keep_mask = np.ones(len(positions), dtype=bool)
    near_indices = np.where(near_tunnel)[0]
    
    # Create sub-grid offsets
    sub_offsets_1d = np.linspace(-spacing/2 + fine_spacing/2, 
                                   spacing/2 - fine_spacing/2, 
                                   refine_factor)
    sub_dx, sub_dz = np.meshgrid(sub_offsets_1d, sub_offsets_1d, indexing='ij')
    sub_dx = sub_dx.flatten()
    sub_dz = sub_dz.flatten()
    
    print(f"\nRefined tunnel exclusion (sub-grid factor: {refine_factor}x)...")
    
    for idx in near_indices:
        px, py, pz = positions[idx]
        sub_x = px + sub_dx
        sub_z = pz + sub_dz
        sub_r = np.sqrt((sub_x - tunnel_center[0])**2 + (sub_z - tunnel_center[2])**2)
        if np.any(sub_r < tunnel_radius):
            keep_mask[idx] = False
    
    # Also mark particles clearly inside tunnel
    clearly_inside = r_from_tunnel < (tunnel_radius - spacing)
    keep_mask[clearly_inside] = False
    
    positions_filtered = positions[keep_mask]
    n_particles = len(positions_filtered)
    n_removed = len(positions) - n_particles
    
    print(f"\nTunnel removal:")
    print(f"  Removed {n_removed:,} particles inside/intersecting tunnel")
    print(f"  Final particle count: {n_particles:,}")
    
    # ============================================================
    # COMPUTE PARTICLE PROPERTIES
    # ============================================================
    
    # Particle volume (cubic for uniform spacing)
    particle_volume = np.full(n_particles, spacing**3, dtype=np.float32)
    
    # Material properties (uniform for elastic validation)
    E_array = np.full(n_particles, E, dtype=np.float32)
    nu_array = np.full(n_particles, nu, dtype=np.float32)
    density_array = np.full(n_particles, density, dtype=np.float32)
    ys_array = np.full(n_particles, ys, dtype=np.float32)
    
    # ============================================================
    # COMPUTE ANALYTICAL KIRSCH SOLUTION FOR REFERENCE
    # ============================================================
    
    print(f"\n" + "=" * 60)
    print("KIRSCH ANALYTICAL SOLUTION (Reference)")
    print("=" * 60)
    
    # Far-field stresses at tunnel depth
    depth = z_surface - tunnel_center[2]
    sigma_v = -density * g * depth  # Vertical stress (negative = compression)
    sigma_h = K0 * sigma_v
    
    print(f"\nFar-field stresses at tunnel center (depth={depth:.1f}m):")
    print(f"  σ_v = {sigma_v/1e6:.3f} MPa (vertical)")
    print(f"  σ_h = {sigma_h/1e6:.3f} MPa (horizontal, K₀={K0})")
    
    # Kirsch solution at tunnel wall
    sigma_theta_crown = 3 * sigma_h - sigma_v
    sigma_theta_springline = 3 * sigma_v - sigma_h
    
    SCF_crown = sigma_theta_crown / sigma_v if sigma_v != 0 else 0
    SCF_springline = sigma_theta_springline / sigma_v if sigma_v != 0 else 0
    
    print(f"\nHoop stress at tunnel wall (r = a = {tunnel_radius}m):")
    print(f"  At crown/invert (θ=0°):   σ_θθ = {sigma_theta_crown/1e6:.3f} MPa (SCF = {SCF_crown:.2f})")
    print(f"  At springline (θ=90°):    σ_θθ = {sigma_theta_springline/1e6:.3f} MPa (SCF = {SCF_springline:.2f})")
    
    # ============================================================
    # SAVE TO HDF5
    # ============================================================
    
    output_file = f"tunnel_domain_elastic{output_suffix}.h5"
    print(f"\n" + "=" * 60)
    print(f"Saving to: {output_file}")
    
    with h5py.File(output_file, "w") as f:
        # Required fields (note: x is transposed to shape (3, N))
        f.create_dataset("x", data=positions_filtered.T.astype(np.float32))
        f.create_dataset("particle_volume", data=particle_volume)
        
        # Material properties
        f.create_dataset("E", data=E_array)
        f.create_dataset("nu", data=nu_array)
        f.create_dataset("density", data=density_array)
        f.create_dataset("ys", data=ys_array)
        
        # Metadata attributes
        f.attrs["tunnel_radius"] = tunnel_radius
        f.attrs["tunnel_center_x"] = tunnel_center[0]
        f.attrs["tunnel_center_y"] = tunnel_center[1]
        f.attrs["tunnel_center_z"] = tunnel_center[2]
        f.attrs["domain_Lx"] = Lx
        f.attrs["domain_Ly"] = Ly
        f.attrs["domain_Lz"] = Lz
        f.attrs["spacing"] = spacing
        f.attrs["E"] = E
        f.attrs["nu"] = nu
        f.attrs["density"] = density
        f.attrs["K0"] = K0
        f.attrs["sigma_v_at_center"] = sigma_v
        f.attrs["sigma_h_at_center"] = sigma_h
    
    print(f"  Positions: shape {positions_filtered.T.shape}")
    print(f"  Particle volume: {particle_volume[0]:.6f} m³")
    print(f"  Total particles: {n_particles:,}")
    
    # Save reference data
    ref_file = f"kirsch_reference_data{output_suffix}.npz"
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
             spacing=spacing)
    
    print(f"Saved reference data to: {ref_file}")
    print(f"\n" + "=" * 60)
    print("Domain creation complete!")
    print("=" * 60)
    
    return output_file, n_particles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create tunnel domain with specified resolution')
    parser.add_argument('--spacing', type=float, default=0.5,
                       help='Particle spacing in meters (default: 0.5)')
    parser.add_argument('--suffix', type=str, default='_hires',
                       help='Output filename suffix (default: _hires)')
    
    args = parser.parse_args()
    create_tunnel_domain(spacing=args.spacing, output_suffix=args.suffix)

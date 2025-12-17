"""
Create a VERY HIGH RESOLUTION tunnel domain with refined tunnel geometry.

The key improvement: use a finer sub-grid for the tunnel boundary test,
so the circular tunnel wall is not staircased at the particle spacing scale.

Spacing comparison:
- Low-res:  1.0m  (~200k particles)
- High-res: 0.5m  (~1.4M particles)  
- VHires:   0.4m  (~2.7M particles) with refined tunnel geometry
"""

import numpy as np
import h5py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create very high-res tunnel domain')
    parser.add_argument('--spacing', type=float, default=0.4,
                        help='Particle spacing (m)')
    parser.add_argument('--suffix', type=str, default='_vhires',
                        help='Suffix for output files')
    parser.add_argument('--refine_factor', type=int, default=4,
                        help='Sub-grid refinement factor for tunnel boundary')
    return parser.parse_args()

def create_vhires_tunnel_domain(spacing=0.4, suffix='_vhires', refine_factor=4):
    """
    Create tunnel domain with refined tunnel geometry.
    
    The tunnel boundary is sampled at (spacing / refine_factor) resolution
    to avoid staircase artifacts on the circular wall.
    """
    
    # ============================================================
    # DOMAIN PARAMETERS (same as other versions)
    # ============================================================
    
    tunnel_radius = 10.0  # meters
    tunnel_center = np.array([100.0, 2.0, 100.0])
    
    Lx = 200.0
    Ly = 4.0
    Lz = 200.0
    
    E = 10e9
    nu = 0.25
    density = 2600
    ys = 1e20
    
    g = 9.81
    K0 = 0.5
    z_surface = Lz
    
    # ============================================================
    # GENERATE PARTICLE POSITIONS WITH REFINED TUNNEL BOUNDARY
    # ============================================================
    
    # Use a much finer grid near the tunnel to get smooth circular boundary
    fine_spacing = spacing / refine_factor
    
    nx = int(Lx / spacing) + 1
    ny = int(Ly / spacing) + 1
    nz = int(Lz / spacing) + 1
    
    print(f"Very High-Resolution Tunnel Domain Generator")
    print(f"=" * 60)
    print(f"Tunnel radius: {tunnel_radius}m")
    print(f"Domain: {Lx}m × {Ly}m × {Lz}m")
    print(f"Particle spacing: {spacing}m")
    print(f"Tunnel boundary refinement: {refine_factor}x (effective {fine_spacing}m)")
    print(f"Initial grid: {nx} × {ny} × {nz} = {nx*ny*nz:,} particles")
    
    # Generate regular grid for particle centers
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    positions = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # ============================================================
    # REFINED TUNNEL EXCLUSION TEST
    # ============================================================
    # For each particle, check if ANY of the sub-grid points within
    # its cell are inside the tunnel. This gives a smooth boundary.
    
    print(f"\nPerforming refined tunnel exclusion test...")
    
    # Distance from tunnel axis (in X-Z plane)
    dx_from_center = positions[:, 0] - tunnel_center[0]
    dz_from_center = positions[:, 2] - tunnel_center[2]
    r_from_tunnel = np.sqrt(dx_from_center**2 + dz_from_center**2)
    
    # Quick pre-filter: particles far from tunnel don't need refined test
    tunnel_influence_radius = tunnel_radius + spacing * 2
    near_tunnel = r_from_tunnel < tunnel_influence_radius
    far_from_tunnel = r_from_tunnel > tunnel_influence_radius
    
    # For particles near tunnel, do refined sub-grid test
    # A particle is OUTSIDE tunnel only if its cell center is > tunnel_radius + half_spacing
    # This ensures no particle volume overlaps with the tunnel
    
    # More accurate: check the closest point of the particle's volume to tunnel axis
    # The particle occupies a cube of side `spacing` centered at its position
    # We want to exclude particles whose cube intersects the tunnel cylinder
    
    # For a cube centered at (px, pz), the closest point to tunnel axis is:
    #   min distance = r - half_spacing (approximately, for axis-aligned cube)
    # But for circular boundary, we need to be more careful
    
    # Use sub-sampling within each particle's cell
    keep_mask = np.ones(len(positions), dtype=bool)
    
    # Process particles near the tunnel with refined test
    near_indices = np.where(near_tunnel)[0]
    
    # Create sub-grid offsets for refined sampling within each cell
    sub_offsets_1d = np.linspace(-spacing/2 + fine_spacing/2, 
                                   spacing/2 - fine_spacing/2, 
                                   refine_factor)
    sub_dx, sub_dz = np.meshgrid(sub_offsets_1d, sub_offsets_1d, indexing='ij')
    sub_dx = sub_dx.flatten()
    sub_dz = sub_dz.flatten()
    n_subsamples = len(sub_dx)
    
    print(f"  Testing {len(near_indices):,} particles near tunnel with {n_subsamples} sub-samples each")
    
    # For each particle near tunnel, check if majority of sub-samples are outside
    for idx in near_indices:
        px, py, pz = positions[idx]
        
        # Check all sub-sample points within this particle's cell
        sub_x = px + sub_dx
        sub_z = pz + sub_dz
        
        sub_r = np.sqrt((sub_x - tunnel_center[0])**2 + (sub_z - tunnel_center[2])**2)
        
        # Particle is inside tunnel if ANY sub-sample is inside
        # (conservative - ensures no particle volume overlaps tunnel)
        if np.any(sub_r < tunnel_radius):
            keep_mask[idx] = False
    
    # Also mark particles clearly inside tunnel (optimization already done above)
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
    
    particle_volume = np.full(n_particles, spacing**3, dtype=np.float32)
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
        f.create_dataset("x", data=positions_filtered.T.astype(np.float32))
        f.create_dataset("particle_volume", data=particle_volume)
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
             spacing=spacing)
    
    print(f"Saved reference data to: {ref_file}")
    print(f"\n" + "=" * 60)
    print("Domain creation complete!")
    print("=" * 60)
    
    return output_file, n_particles

if __name__ == "__main__":
    args = parse_args()
    create_vhires_tunnel_domain(spacing=args.spacing, suffix=args.suffix, 
                                 refine_factor=args.refine_factor)

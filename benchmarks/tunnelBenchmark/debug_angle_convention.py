"""
Debug script to verify angle conventions and stress transformations.
"""

import numpy as np
import pyvista as pv
from pathlib import Path

def debug_analysis():
    output_dir = Path("./output_elastic_long")
    tunnel_center = np.array([100.0, 20.0, 100.0])
    tunnel_radius = 10.0
    
    rho = 2600
    g = 9.81
    depth = 100.0
    sigma_v = -rho * g * depth  # -2.55 MPa (vertical, in Z direction)
    K0 = 0.5
    sigma_h = K0 * sigma_v  # -1.275 MPa (horizontal, in X direction)
    
    vtp_files = sorted(output_dir.glob("*_particles.vtp"))
    latest_vtp = vtp_files[-1]
    
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
    
    print("="*60)
    print("COORDINATE SYSTEM CHECK")
    print("="*60)
    print(f"Tunnel center: x={tunnel_center[0]}, y={tunnel_center[1]}, z={tunnel_center[2]}")
    print(f"σ_v (vertical, z): {sigma_v/1e6:.3f} MPa")
    print(f"σ_h (horizontal, x): {sigma_h/1e6:.3f} MPa")
    print(f"\nKirsch angle convention:")
    print(f"  θ = 0: +x direction (horizontal, springline)")
    print(f"  θ = 90°: +z direction (vertical, crown)")
    
    # Sample specific points near the wall
    print("\n" + "="*60)
    print("STRESS AT SPECIFIC LOCATIONS (r ≈ 1.3a)")
    print("="*60)
    
    x_rel = pos_slice[:, 0] - tunnel_center[0]
    z_rel = pos_slice[:, 2] - tunnel_center[2]
    r = np.sqrt(x_rel**2 + z_rel**2)
    
    # Wall region
    wall_mask = (r >= 1.2*tunnel_radius) & (r <= 1.5*tunnel_radius)
    
    # Springline (+x direction, θ=0)
    springline_mask = wall_mask & (np.abs(z_rel) < 2.0) & (x_rel > 5.0)
    # Crown (+z direction, θ=90)
    crown_mask = wall_mask & (np.abs(x_rel) < 2.0) & (z_rel > 5.0)
    # Invert (-z direction, θ=-90 or 270)
    invert_mask = wall_mask & (np.abs(x_rel) < 2.0) & (z_rel < -5.0)
    
    for name, mask in [("Springline (θ=0°, +x)", springline_mask), 
                       ("Crown (θ=90°, +z)", crown_mask),
                       ("Invert (θ=-90°, -z)", invert_mask)]:
        pass  # Handled below
    
    # Let me redo this more carefully
    stress_xx_slice = sigma_xx[y_mask]
    stress_zz_slice = sigma_zz[y_mask]
    stress_xz_slice = sigma_xz[y_mask]
    
    for name, x_cond, z_cond in [
        ("Springline (+x, θ=0°)", (x_rel > 5.0) & (np.abs(z_rel) < 3.0), None),
        ("Crown (+z, θ=90°)", (np.abs(x_rel) < 3.0) & (z_rel > 5.0), None),
        ("Invert (-z, θ=-90°)", (np.abs(x_rel) < 3.0) & (z_rel < -5.0), None),
    ]:
        combined = wall_mask & x_cond
        n = np.sum(combined)
        if n > 0:
            xx = np.mean(stress_xx_slice[combined])
            zz = np.mean(stress_zz_slice[combined])
            xz = np.mean(stress_xz_slice[combined])
            r_mean = np.mean(r[combined])
            
            # At springline (θ=0): radial = x, tangential = z
            # At crown (θ=90): radial = z, tangential = -x
            if "Springline" in name:
                sigma_rr = xx
                sigma_tt = zz
            elif "Crown" in name:
                sigma_rr = zz
                sigma_tt = xx
            else:  # Invert
                sigma_rr = zz
                sigma_tt = xx
            
            print(f"\n{name} (n={n}, r_mean={r_mean/tunnel_radius:.2f}a):")
            print(f"  Raw stresses: σ_xx={xx/1e6:.3f}, σ_zz={zz/1e6:.3f}, σ_xz={xz/1e6:.3f} MPa")
            print(f"  Polar stresses: σ_rr={sigma_rr/1e6:.3f}, σ_θθ={sigma_tt/1e6:.3f} MPa")
            
            # Kirsch prediction at this radius
            a = tunnel_radius
            p0 = (sigma_v + sigma_h) / 2
            q0 = (sigma_v - sigma_h) / 2  # negative since sigma_v more negative
            
            ratio = a / r_mean
            ratio2 = ratio**2
            ratio4 = ratio**4
            
            if "Springline" in name:
                theta = 0
            elif "Crown" in name:
                theta = np.pi/2
            else:
                theta = -np.pi/2
            
            kirsch_rr = p0 * (1 - ratio2) + q0 * (1 - 4*ratio2 + 3*ratio4) * np.cos(2*theta)
            kirsch_tt = p0 * (1 + ratio2) - q0 * (1 + 3*ratio4) * np.cos(2*theta)
            
            print(f"  Kirsch prediction: σ_rr={kirsch_rr/1e6:.3f}, σ_θθ={kirsch_tt/1e6:.3f} MPa")
    
    print("\n" + "="*60)
    print("KIRSCH THEORETICAL VALUES AT WALL (r=a)")
    print("="*60)
    p0 = (sigma_v + sigma_h) / 2
    q0 = (sigma_v - sigma_h) / 2
    
    print(f"p0 = (σ_v + σ_h)/2 = {p0/1e6:.3f} MPa")
    print(f"q0 = (σ_v - σ_h)/2 = {q0/1e6:.3f} MPa")
    
    # At wall (r=a), Kirsch gives:
    # σ_rr = 0 (free surface)
    # σ_θθ = 2*p0 - 2*q0*cos(2θ) = (σ_v + σ_h) - (σ_v - σ_h)*cos(2θ)
    
    print(f"\nAt wall (r=a):")
    for theta_deg in [0, 45, 90]:
        theta = np.radians(theta_deg)
        sigma_tt_wall = 2*p0 - 2*q0*np.cos(2*theta)
        print(f"  θ={theta_deg:3d}°: σ_rr=0, σ_θθ = {sigma_tt_wall/1e6:.3f} MPa")
    
    print(f"\nNote: At θ=0° (springline), σ_θθ = σ_v + σ_h - (σ_v - σ_h) = 2σ_h = {2*sigma_h/1e6:.3f} MPa")
    print(f"      At θ=90° (crown), σ_θθ = σ_v + σ_h + (σ_v - σ_h) = 2σ_v = {2*sigma_v/1e6:.3f} MPa")
    print(f"      But wait... let me recalculate:")
    print(f"      σ_θθ(θ=0) = 2p0 - 2q0*cos(0) = 2p0 - 2q0 = (σ_v+σ_h) - (σ_v-σ_h) = 2σ_h")
    print(f"      σ_θθ(θ=90) = 2p0 - 2q0*cos(180°) = 2p0 + 2q0 = (σ_v+σ_h) + (σ_v-σ_h) = 2σ_v")
    print(f"      Hmm, that doesn't match standard Kirsch. Let me check the formula...")
    
    # Standard Kirsch (Jaeger & Cook convention):
    # σ_θθ(wall) = σ_v + σ_h - 2(σ_v - σ_h)cos(2θ)  where θ from vertical
    # So at θ=0 (springline, horizontal from vertical): σ_θθ = σ_v + σ_h - 2(σ_v - σ_h) = 3σ_h - σ_v
    # At θ=90° (crown/invert): σ_θθ = σ_v + σ_h + 2(σ_v - σ_h) = 3σ_v - σ_h
    
    print(f"\n*** CORRECTED Kirsch (θ from vertical axis) ***")
    print(f"At θ=0° from vertical (i.e., at springline):")
    sigma_tt_springline = 3*sigma_h - sigma_v
    print(f"  σ_θθ = 3σ_h - σ_v = {sigma_tt_springline/1e6:.3f} MPa")
    
    print(f"At θ=90° from vertical (i.e., at crown/invert):")
    sigma_tt_crown = 3*sigma_v - sigma_h
    print(f"  σ_θθ = 3σ_v - σ_h = {sigma_tt_crown/1e6:.3f} MPa")


if __name__ == "__main__":
    debug_analysis()

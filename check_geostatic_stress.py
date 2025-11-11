"""
Diagnostic script to check geostatic stress initialization.
Run this after initializing your simulation to verify stress distribution.
"""
import numpy as np
import matplotlib.pyplot as plt

def check_geostatic_stress(particle_x, particle_stress, z_top, rho, g):
    """
    Check if geostatic stress field is realistic.
    
    Args:
        particle_x: (N,3) array of particle positions
        particle_stress: (N,3,3) array of particle stress tensors
        z_top: Top surface elevation
        rho: Density (kg/m³)
        g: Gravity magnitude (m/s²)
    """
    
    # Extract data
    z_coords = particle_x[:, 2]
    depth = z_top - z_coords
    
    sigma_zz = particle_stress[:, 2, 2]  # Vertical stress
    sigma_xx = particle_stress[:, 0, 0]  # Horizontal stress (x)
    sigma_yy = particle_stress[:, 1, 1]  # Horizontal stress (y)
    
    # Mean stress (pressure)
    mean_stress = (sigma_xx + sigma_yy + sigma_zz) / 3.0
    
    # Von Mises stress
    s = particle_stress - mean_stress[:, None, None] * np.eye(3)
    von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1,2)))
    
    # Expected stress
    expected_sigma_v = -rho * g * depth  # Negative = compression
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Vertical stress vs depth
    ax = axes[0, 0]
    ax.scatter(depth, sigma_zz, alpha=0.5, s=1, label='Measured σ_zz')
    ax.plot(depth, expected_sigma_v, 'r-', linewidth=2, label='Expected (-ρgh)')
    ax.set_xlabel('Depth from surface (m)')
    ax.set_ylabel('Vertical stress σ_zz (Pa)')
    ax.set_title('Vertical Stress vs Depth')
    ax.legend()
    ax.grid(True)
    
    # 2. Horizontal stresses vs depth
    ax = axes[0, 1]
    ax.scatter(depth, sigma_xx, alpha=0.5, s=1, label='σ_xx')
    ax.scatter(depth, sigma_yy, alpha=0.5, s=1, label='σ_yy')
    ax.set_xlabel('Depth from surface (m)')
    ax.set_ylabel('Horizontal stress (Pa)')
    ax.set_title('Horizontal Stress vs Depth')
    ax.legend()
    ax.grid(True)
    
    # 3. Mean stress vs depth
    ax = axes[0, 2]
    ax.scatter(depth, mean_stress, alpha=0.5, s=1, label='Mean stress')
    ax.set_xlabel('Depth from surface (m)')
    ax.set_ylabel('Mean stress (Pa)')
    ax.set_title('Mean Stress vs Depth')
    ax.legend()
    ax.grid(True)
    
    # 4. Von Mises stress vs depth
    ax = axes[1, 0]
    ax.scatter(depth, von_mises, alpha=0.5, s=1, label='Von Mises')
    ax.set_xlabel('Depth from surface (m)')
    ax.set_ylabel('Von Mises stress (Pa)')
    ax.set_title('Von Mises Stress vs Depth\n(Nearly uniform under geostatic loading!)')
    ax.legend()
    ax.grid(True)
    
    # 5. K0 ratio vs depth
    ax = axes[1, 1]
    K0_measured = sigma_xx / (sigma_zz + 1e-10)  # Avoid division by zero
    ax.scatter(depth, K0_measured, alpha=0.5, s=1, label='σ_xx/σ_zz')
    ax.set_xlabel('Depth from surface (m)')
    ax.set_ylabel('K0 ratio')
    ax.set_title('Lateral Earth Pressure Coefficient')
    ax.legend()
    ax.grid(True)
    
    # 6. Stress error
    ax = axes[1, 2]
    error_percent = 100 * (sigma_zz - expected_sigma_v) / (np.abs(expected_sigma_v) + 1e-10)
    ax.scatter(depth, error_percent, alpha=0.5, s=1)
    ax.set_xlabel('Depth from surface (m)')
    ax.set_ylabel('Error (%)')
    ax.set_title('Vertical Stress Error')
    ax.axhline(0, color='r', linestyle='--')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output/geostatic_stress_check.png', dpi=150)
    print(f"Saved diagnostic plot to ./output/geostatic_stress_check.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("GEOSTATIC STRESS DIAGNOSTICS")
    print("="*60)
    print(f"Domain height: {z_top - z_coords.min():.2f} m")
    print(f"Expected gradient: dσ_v/dz = {rho*g:.2f} Pa/m")
    print(f"\nAt surface (z={z_top:.2f}):")
    print(f"  Expected σ_zz: 0 Pa")
    print(f"  Measured σ_zz: {sigma_zz[np.argmax(z_coords)]:.2e} Pa")
    print(f"\nAt bottom (z={z_coords.min():.2f}):")
    print(f"  Expected σ_zz: {expected_sigma_v[np.argmin(z_coords)]:.2e} Pa")
    print(f"  Measured σ_zz: {sigma_zz[np.argmin(z_coords)]:.2e} Pa")
    print(f"\nAt mid-depth:")
    print(f"  Mean σ_zz: {np.mean(sigma_zz):.2e} Pa")
    print(f"  Mean σ_xx: {np.mean(sigma_xx):.2e} Pa")
    print(f"  Mean K0 ratio: {np.mean(K0_measured[depth>0.1]):.3f}")
    print(f"  Mean Von Mises: {np.mean(von_mises):.2e} Pa")
    print(f"\nWhy Von Mises looks uniform:")
    print(f"  Under geostatic loading, the stress RATIO (σ_h/σ_v) is constant.")
    print(f"  Von Mises measures deviatoric stress (shape change), which depends on this ratio.")
    print(f"  Since the ratio is constant, Von Mises is nearly uniform!")
    print(f"  Use 'sigma_zz' color mode to visualize the actual stress gradient.")
    print("="*60)

if __name__ == "__main__":
    print("Import this module and call check_geostatic_stress() with your particle data.")
    print("Example:")
    print("  from check_geostatic_stress import check_geostatic_stress")
    print("  check_geostatic_stress(particle_x.numpy(), particle_stress.numpy(), z_top, rho, g)")

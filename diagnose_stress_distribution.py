"""
Diagnostic script to check coordinate system and stress distribution.
Run this after a few simulation steps to verify the stress field.
"""
import numpy as np
import matplotlib.pyplot as plt

def diagnose_stress_distribution(particle_x, particle_stress, z_top, rho, g):
    """
    Check if stress distribution is correct or inverted.
    """
    
    # Extract coordinates and stress components
    z_coords = particle_x[:, 2]
    sigma_zz = particle_stress[:, 2, 2]  # Vertical stress
    sigma_xx = particle_stress[:, 0, 0]  # Horizontal x
    sigma_yy = particle_stress[:, 1, 1]  # Horizontal y
    
    # Mean stress (pressure)
    mean_stress = (sigma_xx + sigma_yy + sigma_zz) / 3.0
    
    print("="*70)
    print("COORDINATE SYSTEM CHECK")
    print("="*70)
    print(f"z_top (reference): {z_top:.3f}")
    print(f"min(z_coords): {np.min(z_coords):.3f}")
    print(f"max(z_coords): {np.max(z_coords):.3f}")
    print(f"Domain height: {np.max(z_coords) - np.min(z_coords):.3f}")
    
    if np.max(z_coords) > z_top:
        print("WARNING: Some particles are above z_top!")
        print("  → Check z_top calculation")
    
    print("\n" + "="*70)
    print("STRESS STATISTICS")
    print("="*70)
    
    # Find particles at top and bottom
    idx_top = np.argmax(z_coords)
    idx_bottom = np.argmin(z_coords)
    
    print(f"\nAt TOP (z={z_coords[idx_top]:.3f}):")
    print(f"  sigma_zz: {sigma_zz[idx_top]:.2e} Pa")
    print(f"  sigma_xx: {sigma_xx[idx_top]:.2e} Pa")
    print(f"  Mean stress: {mean_stress[idx_top]:.2e} Pa")
    
    print(f"\nAt BOTTOM (z={z_coords[idx_bottom]:.3f}):")
    print(f"  sigma_zz: {sigma_zz[idx_bottom]:.2e} Pa")
    print(f"  sigma_xx: {sigma_xx[idx_bottom]:.2e} Pa")
    print(f"  Mean stress: {mean_stress[idx_bottom]:.2e} Pa")
    
    # Expected values
    depth_bottom = z_top - z_coords[idx_bottom]
    expected_sigma_v_bottom = -rho * g * depth_bottom
    
    print(f"\nEXPECTED at bottom (depth={depth_bottom:.3f} m):")
    print(f"  sigma_zz: {expected_sigma_v_bottom:.2e} Pa (compression = negative)")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    # Check if stress increases with depth
    top_mean = np.mean(mean_stress[z_coords > np.percentile(z_coords, 90)])
    bottom_mean = np.mean(mean_stress[z_coords < np.percentile(z_coords, 10)])
    
    print(f"\nMean stress in top 10%: {top_mean:.2e} Pa")
    print(f"Mean stress in bottom 10%: {bottom_mean:.2e} Pa")
    
    if np.abs(bottom_mean) > np.abs(top_mean):
        print("✓ CORRECT: Stress magnitude increases with depth")
    else:
        print("✗ INVERTED: Stress is higher at top than bottom!")
        print("\nPossible causes:")
        print("  1. Coordinate system: Check if z increases upward or downward")
        print("  2. depth calculation: depth = z_top - z assumes z increases upward")
        print("  3. Stress sign: Check kirchoff_stress function")
        print("  4. Visualization: Check if colormap is inverted")
    
    # Correlation test
    corr = np.corrcoef(z_coords, mean_stress)[0, 1]
    print(f"\nCorrelation (z vs mean_stress): {corr:.3f}")
    if corr > 0:
        print("  → Stress increases with z (expected if z increases downward)")
    else:
        print("  → Stress decreases with z (expected if z increases upward)")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Vertical stress vs z
    axes[0].scatter(sigma_zz, z_coords, alpha=0.5, s=1)
    axes[0].set_xlabel('Vertical stress sigma_zz (Pa)')
    axes[0].set_ylabel('z coordinate')
    axes[0].set_title('Vertical Stress vs Height')
    axes[0].grid(True)
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Mean stress vs z
    axes[1].scatter(mean_stress, z_coords, alpha=0.5, s=1)
    axes[1].set_xlabel('Mean stress (Pa)')
    axes[1].set_ylabel('z coordinate')
    axes[1].set_title('Mean Stress vs Height')
    axes[1].grid(True)
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Depth vs stress
    depth_from_top = z_top - z_coords
    axes[2].scatter(mean_stress, depth_from_top, alpha=0.5, s=1)
    axes[2].set_xlabel('Mean stress (Pa)')
    axes[2].set_ylabel('Depth from z_top (m)')
    axes[2].set_title('Mean Stress vs Depth')
    axes[2].grid(True)
    axes[2].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[2].invert_yaxis()  # Top at top, bottom at bottom
    
    plt.tight_layout()
    plt.savefig('./output/stress_diagnosis.png', dpi=150)
    print(f"\n✓ Saved diagnostic plot to: ./output/stress_diagnosis.png")
    plt.close()
    
    return corr, top_mean, bottom_mean

if __name__ == "__main__":
    print("Usage:")
    print("  from diagnose_stress_distribution import diagnose_stress_distribution")
    print("  diagnose_stress_distribution(particle_x.numpy(), particle_stress.numpy(), z_top, rho, g)")

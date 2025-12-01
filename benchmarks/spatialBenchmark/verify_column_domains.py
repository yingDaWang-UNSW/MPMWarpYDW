"""
Quick verification of column coupling test domains.
Shows particle counts and property distributions.
"""

import h5py
import numpy as np

def verify_domain(filename, case_name):
    print(f"\n{'='*70}")
    print(f"VERIFYING: {case_name}")
    print(f"File: {filename}")
    print(f"{'='*70}")
    
    with h5py.File(filename, 'r') as f:
        # Read data
        x = np.array(f['x'])  # Shape: (3, nPoints)
        positions = x.T  # Shape: (nPoints, 3)
        
        density = np.array(f['density'])
        ys = np.array(f['ys'])
        E = np.array(f['E'])
        strainCriteria = np.array(f['strainCriteria'])
        volume = np.array(f['particle_volume'])
        
        nPoints = positions.shape[0]
        z = positions[:, 2]
        
        # Identify bottom and top
        interface_z = 100.0
        bottom_mask = z < interface_z
        top_mask = z >= interface_z
        
        n_bottom = np.sum(bottom_mask)
        n_top = np.sum(top_mask)
        
        print(f"\nGeometry:")
        print(f"  Total particles: {nPoints}")
        print(f"  Bottom half (z < {interface_z}m): {n_bottom} particles")
        print(f"  Top half (z ≥ {interface_z}m): {n_top} particles")
        print(f"  Z range: [{z.min():.2f}, {z.max():.2f}] m")
        
        print(f"\nBottom properties:")
        print(f"  Density: {density[bottom_mask].min():.0f} - {density[bottom_mask].max():.0f} kg/m³")
        print(f"  Yield stress: {ys[bottom_mask].min():.2e} - {ys[bottom_mask].max():.2e} Pa")
        print(f"  Strain criteria: {strainCriteria[bottom_mask].min():.3f} - {strainCriteria[bottom_mask].max():.3f}")
        print(f"  Total mass: {(density[bottom_mask] * volume[bottom_mask]).sum():.2e} kg")
        
        print(f"\nTop properties:")
        print(f"  Density: {density[top_mask].min():.0f} - {density[top_mask].max():.0f} kg/m³")
        print(f"  Yield stress: {ys[top_mask].min():.2e} - {ys[top_mask].max():.2e} Pa")
        print(f"  Strain criteria: {strainCriteria[top_mask].min():.3f} - {strainCriteria[top_mask].max():.3f}")
        print(f"  Total mass: {(density[top_mask] * volume[top_mask]).sum():.2e} kg")
        
        # Expected values
        contact_area = 50 * 50
        top_weight = (density[top_mask] * volume[top_mask]).sum() * 9.81
        expected_stress = top_weight / contact_area
        
        print(f"\nExpected stress at interface:")
        print(f"  Top weight: {top_weight:.2e} N")
        print(f"  Contact area: {contact_area:.1f} m²")
        print(f"  Expected σ_zz: {expected_stress/1e3:.2f} kPa")
        
        return {
            'case': case_name,
            'n_total': nPoints,
            'n_bottom': n_bottom,
            'n_top': n_top,
            'top_density': density[top_mask].mean(),
            'top_ys': ys[top_mask].mean(),
            'top_mass': (density[top_mask] * volume[top_mask]).sum(),
            'expected_stress': expected_stress,
        }

# Verify all three domains
results = []
results.append(verify_domain("coupling_column_same.h5", "SAME"))
results.append(verify_domain("coupling_column_weak_same_density.h5", "WEAK_SAME_DENSITY"))
results.append(verify_domain("coupling_column_weak_light.h5", "WEAK_LIGHT"))

# Summary comparison
print(f"\n{'='*70}")
print("SUMMARY COMPARISON")
print(f"{'='*70}")
print(f"\n{'Case':<22} {'Top ρ (kg/m³)':<15} {'Top ys (Pa)':<15} {'Top mass (kg)':<18} {'Exp. σ_zz (kPa)':<15}")
print("-"*70)

for r in results:
    print(f"{r['case']:<22} {r['top_density']:<15.0f} {r['top_ys']:<15.2e} {r['top_mass']:<18.2e} {r['expected_stress']/1e3:<15.2f}")

print("-"*70)

# Key ratios
ratio_mass = results[2]['top_mass'] / results[1]['top_mass']
ratio_stress = results[2]['expected_stress'] / results[1]['expected_stress']

print(f"\nExpected ratios (WEAK_LIGHT / WEAK_SAME_DENSITY):")
print(f"  Mass ratio:   {ratio_mass:.3f} (should be 0.100)")
print(f"  Stress ratio: {ratio_stress:.3f} (should be 0.100)")

if abs(ratio_mass - 0.1) < 0.001 and abs(ratio_stress - 0.1) < 0.001:
    print(f"  ✓✓ PERFECT: Ratios match expectation!")
else:
    print(f"  ⚠ WARNING: Ratios don't match expectation")

print(f"\n{'='*70}\n")

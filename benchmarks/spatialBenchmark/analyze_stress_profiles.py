"""
Comprehensive Stress Profile Analysis for MPM-XPBD Coupling

Compares three cases:
1. SAME: 100m MPM bottom + 100m MPM top (both 3000 kg/m³, high strength)
2. WEAK_SAME_DENSITY: 100m MPM bottom + 100m XPBD top (both 3000 kg/m³)
3. WEAK_LIGHT: 100m MPM bottom + 100m XPBD top (3000 + 300 kg/m³)

Purpose: Determine if XPBD particles contribute their weight to MPM stress field
"""

import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import glob

def read_vtp(filepath):
    """Read VTP file and extract all fields."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    positions = vtk_to_numpy(points.GetData())
    
    point_data = polydata.GetPointData()
    data = {'positions': positions}
    
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        array = vtk_to_numpy(point_data.GetArray(i))
        data[name] = array
    
    return data

def find_latest_vtp(folder, step=None):
    """Find a VTP file in a folder. If step specified, use that step; otherwise use latest."""
    if step is not None:
        pattern = os.path.join(folder, f"*{step:010d}_particles.vtp")
        files = glob.glob(pattern)
        if files:
            return files[0]
        # Try old filename format without underscores
        pattern = os.path.join(folder, f"sim_step_{step:010d}_particles.vtp")
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    # Fall back to latest
    pattern = os.path.join(folder, "*_particles.vtp")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No VTP files found in {folder}")
    return sorted(files)[-1]

def compute_stress_components(data):
    """Extract stress components from stress tensor."""
    # stress_tensor has 6 components: xx, yy, zz, xy, xz, yz
    sigma = data['stress_tensor']
    result = {
        'sigma_xx': sigma[:, 0],
        'sigma_yy': sigma[:, 1],
        'sigma_zz': sigma[:, 2],
        'sigma_xy': sigma[:, 3],
        'sigma_xz': sigma[:, 4],
        'sigma_yz': sigma[:, 5],
    }
    # Mean stress (pressure, negative = compression)
    result['mean_stress'] = (sigma[:, 0] + sigma[:, 1] + sigma[:, 2]) / 3.0
    return result

def bin_by_height(z, values, n_bins=50, z_min=None, z_max=None):
    """Bin values by height and compute mean/std."""
    if z_min is None:
        z_min = z.min()
    if z_max is None:
        z_max = z.max()
    
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    means = np.zeros(n_bins)
    stds = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (z >= bin_edges[i]) & (z < bin_edges[i+1])
        if mask.sum() > 0:
            means[i] = values[mask].mean()
            stds[i] = values[mask].std()
            counts[i] = mask.sum()
        else:
            means[i] = np.nan
            stds[i] = np.nan
    
    return bin_centers, means, stds, counts

# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("COMPREHENSIVE STRESS PROFILE ANALYSIS")
print("=" * 70)

# Find and load files - use step 50000 for consistency with check_xpbd_completely_weightless.py
TARGET_STEP = 50000
folders = {
    'SAME': 'output_coupling_column_same',
    'WEAK_SAME_DENSITY': 'output_coupling_column_weak_same_density', 
    'WEAK_LIGHT': 'output_coupling_column_weak_light'
}

data = {}
for name, folder in folders.items():
    try:
        filepath = find_latest_vtp(folder, step=TARGET_STEP)
        print(f"\nLoading {name}: {os.path.basename(filepath)}")
        data[name] = read_vtp(filepath)
        print(f"  Particles: {len(data[name]['positions'])}")
        print(f"  Z range: {data[name]['positions'][:, 2].min():.1f} - {data[name]['positions'][:, 2].max():.1f} m")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

if len(data) == 0:
    print("\nNo data files found! Run the simulations first.")
    exit(1)

# ============================================================================
# THEORETICAL EXPECTATIONS
# ============================================================================
print("\n" + "=" * 70)
print("THEORETICAL EXPECTATIONS")
print("=" * 70)

# Constants
rho_heavy = 3000.0  # kg/m³
rho_light = 300.0   # kg/m³
g = 9.81  # m/s²
H_total = 200.0  # m (total height)
H_bottom = 100.0  # m (bottom MPM block)

# Theoretical stress profiles (σ_zz = -ρ*g*h where h = depth from top)
z_theory = np.linspace(0, H_total, 200)

# Case 1: SAME - Full 200m column at 3000 kg/m³
# At depth d from top (z = H_total - d): σ_zz = -ρ*g*d
depth_same = H_total - z_theory
sigma_zz_theory_same = -rho_heavy * g * depth_same

# Case 2: WEAK_SAME_DENSITY - If XPBD has full weight (3000 kg/m³)
# Same as SAME case
sigma_zz_theory_weak_same_full = sigma_zz_theory_same.copy()

# Case 3: WEAK_SAME_DENSITY - If XPBD is weightless
# Only bottom 100m contributes: at z, depth from z=100m interface
depth_from_interface = np.clip(H_bottom - z_theory, 0, H_bottom)
sigma_zz_theory_weightless = -rho_heavy * g * depth_from_interface

# Case 4: WEAK_LIGHT - If XPBD has light weight (300 kg/m³)
# Top 100m at 300 kg/m³, bottom 100m at 3000 kg/m³
depth_in_top = np.clip(H_total - z_theory, 0, H_bottom)  # Depth into top block (0-100m)
depth_in_bottom = np.clip(H_bottom - z_theory, 0, H_bottom)  # Depth into bottom block (0-100m)
sigma_zz_theory_weak_light = -(rho_light * g * depth_in_top + rho_heavy * g * depth_in_bottom)

print(f"\nMaterial properties:")
print(f"  Heavy density (MPM): {rho_heavy} kg/m³")
print(f"  Light density (XPBD in WEAK_LIGHT): {rho_light} kg/m³")
print(f"  Gravity: {g} m/s²")

print(f"\nExpected stress at z=0 (bottom):")
print(f"  SAME (200m @ 3000 kg/m³):       {-rho_heavy * g * H_total / 1e6:.3f} MPa")
print(f"  WEAK_LIGHT (100m@3000 + 100m@300): {-(rho_heavy * g * H_bottom + rho_light * g * H_bottom) / 1e6:.3f} MPa")
print(f"  If XPBD weightless (100m only):  {-rho_heavy * g * H_bottom / 1e6:.3f} MPa")

# ============================================================================
# FIGURE 1: STRESS PROFILES (σ_zz vs z)
# ============================================================================
fig1, axes = plt.subplots(1, 3, figsize=(15, 6))

# Color scheme
colors = {
    'SAME': 'blue',
    'WEAK_SAME_DENSITY': 'green',
    'WEAK_LIGHT': 'orange'
}

# Plot 1a: All cases on same axes
ax = axes[0]
for name, d in data.items():
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    z_bins, sigma_mean, sigma_std, counts = bin_by_height(z, sigma['sigma_zz'], n_bins=40)
    ax.plot(sigma_mean / 1e6, z_bins, 'o-', label=name, color=colors[name], markersize=4)
    ax.fill_betweenx(z_bins, (sigma_mean - sigma_std) / 1e6, (sigma_mean + sigma_std) / 1e6, 
                     alpha=0.2, color=colors[name])

# Add theoretical lines
ax.plot(sigma_zz_theory_same / 1e6, z_theory, 'b--', label='Theory: Full 200m', linewidth=2)
ax.plot(sigma_zz_theory_weightless / 1e6, z_theory, 'r--', label='Theory: XPBD weightless', linewidth=2)
ax.plot(sigma_zz_theory_weak_light / 1e6, z_theory, 'orange', linestyle=':', label='Theory: WEAK_LIGHT', linewidth=2)

ax.set_xlabel('Vertical Stress σ_zz (MPa)', fontsize=12)
ax.set_ylabel('Height z (m)', fontsize=12)
ax.set_title('Vertical Stress Profiles', fontsize=14)
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='MPM/XPBD interface')
ax.set_xlim([-7, 0.5])

# Plot 1b: Deviation from theory
ax = axes[1]
for name, d in data.items():
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    z_bins, sigma_mean, sigma_std, counts = bin_by_height(z, sigma['sigma_zz'], n_bins=40)
    
    # Interpolate theory to bin centers
    if name == 'SAME':
        theory_interp = np.interp(z_bins, z_theory, sigma_zz_theory_same)
    elif name == 'WEAK_LIGHT':
        theory_interp = np.interp(z_bins, z_theory, sigma_zz_theory_weak_light)
    else:  # WEAK_SAME_DENSITY - compare to full weight theory
        theory_interp = np.interp(z_bins, z_theory, sigma_zz_theory_weak_same_full)
    
    deviation = (sigma_mean - theory_interp) / 1e6
    ax.plot(deviation, z_bins, 'o-', label=name, color=colors[name], markersize=4)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Deviation from Theory (MPa)', fontsize=12)
ax.set_ylabel('Height z (m)', fontsize=12)
ax.set_title('Deviation from Expected Stress', fontsize=14)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

# Plot 1c: Mean stress (pressure)
ax = axes[2]
for name, d in data.items():
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    z_bins, p_mean, p_std, counts = bin_by_height(z, sigma['mean_stress'], n_bins=40)
    ax.plot(p_mean / 1e6, z_bins, 'o-', label=name, color=colors[name], markersize=4)

ax.set_xlabel('Mean Stress (MPa)', fontsize=12)
ax.set_ylabel('Height z (m)', fontsize=12)
ax.set_title('Mean Stress (Pressure) Profiles', fontsize=14)
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('stress_profiles_comparison.png', dpi=150)
print("\nSaved: stress_profiles_comparison.png")

# ============================================================================
# FIGURE 2: STRESS COMPONENTS COMPARISON
# ============================================================================
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

components = ['sigma_xx', 'sigma_yy', 'sigma_zz', 'sigma_xy', 'sigma_xz', 'sigma_yz']
titles = ['σ_xx (Horizontal)', 'σ_yy (Horizontal)', 'σ_zz (Vertical)', 
          'τ_xy (Shear)', 'τ_xz (Shear)', 'τ_yz (Shear)']

for idx, (comp, title) in enumerate(zip(components, titles)):
    ax = axes.flat[idx]
    for name, d in data.items():
        z = d['positions'][:, 2]
        sigma = compute_stress_components(d)
        z_bins, s_mean, s_std, counts = bin_by_height(z, sigma[comp], n_bins=40)
        ax.plot(s_mean / 1e6, z_bins, 'o-', label=name, color=colors[name], markersize=3)
    
    ax.set_xlabel(f'{title} (MPa)', fontsize=10)
    ax.set_ylabel('Height z (m)', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('stress_components_comparison.png', dpi=150)
print("Saved: stress_components_comparison.png")

# ============================================================================
# FIGURE 3: MATERIAL STATE AND DAMAGE
# ============================================================================
fig3, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Material label distribution (if available)
for idx, (name, d) in enumerate(data.items()):
    ax = axes[0, idx]
    z = d['positions'][:, 2]
    
    if 'material_label' in d:
        mat_label = d['material_label']
        
        # Scatter plot colored by material label
        scatter = ax.scatter(np.zeros_like(z) + np.random.uniform(-0.3, 0.3, len(z)), 
                            z, c=mat_label, cmap='viridis', s=1, alpha=0.5)
        
        # Count by material type
        mpm_locked = (mat_label == 0).sum()
        mpm_free = (mat_label == 1).sum()
        xpbd = (mat_label == 2).sum()
        
        ax.set_title(f'{name}\nMPM(0):{mpm_locked}, MPM(1):{mpm_free}, XPBD:{xpbd}', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Material Label')
    else:
        # Use damage as proxy for material state
        damage = d.get('damage', np.zeros(len(z)))
        scatter = ax.scatter(np.zeros_like(z) + np.random.uniform(-0.3, 0.3, len(z)), 
                            z, c=damage, cmap='Reds', s=1, alpha=0.5)
        ax.set_title(f'{name}\n(material_label not available)', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Damage')
    
    ax.set_ylabel('Height z (m)', fontsize=10)
    ax.set_xlim([-1, 1])
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Interface')

# Row 2: Damage distribution
for idx, (name, d) in enumerate(data.items()):
    ax = axes[1, idx]
    z = d['positions'][:, 2]
    damage = d.get('damage', np.zeros(len(z)))
    
    z_bins, d_mean, d_std, counts = bin_by_height(z, damage, n_bins=40)
    ax.plot(d_mean, z_bins, 'o-', color=colors[name], markersize=4)
    ax.fill_betweenx(z_bins, d_mean - d_std, d_mean + d_std, alpha=0.3, color=colors[name])
    
    ax.set_xlabel('Damage', fontsize=10)
    ax.set_ylabel('Height z (m)', fontsize=10)
    ax.set_title(f'{name} - Damage Profile', fontsize=10)
    ax.set_xlim([-0.1, 1.1])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('material_state_comparison.png', dpi=150)
print("Saved: material_state_comparison.png")

# ============================================================================
# FIGURE 4: QUANTITATIVE ANALYSIS - WEIGHT CONTRIBUTION
# ============================================================================
fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 4a: Stress at different heights
ax = axes[0]
heights = [10, 30, 50, 70, 90]  # Heights to sample in bottom block
bar_width = 0.25

for i, name in enumerate(data.keys()):
    d = data[name]
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    
    stresses = []
    for h in heights:
        mask = (z >= h - 5) & (z < h + 5)
        if mask.sum() > 0:
            stresses.append(sigma['sigma_zz'][mask].mean() / 1e6)
        else:
            stresses.append(np.nan)
    
    x = np.arange(len(heights)) + i * bar_width
    ax.bar(x, [-s for s in stresses], bar_width, label=name, color=colors[name])

# Add theoretical values
theory_same = [rho_heavy * g * (H_total - h) / 1e6 for h in heights]
theory_weightless = [rho_heavy * g * max(H_bottom - h, 0) / 1e6 for h in heights]

ax.plot(np.arange(len(heights)) + bar_width, theory_same, 'k^--', label='Theory: Full weight', markersize=8)
ax.plot(np.arange(len(heights)) + bar_width, theory_weightless, 'rv--', label='Theory: Weightless', markersize=8)

ax.set_xlabel('Height z (m)', fontsize=12)
ax.set_ylabel('|σ_zz| (MPa)', fontsize=12)
ax.set_title('Vertical Stress Magnitude at Different Heights', fontsize=14)
ax.set_xticks(np.arange(len(heights)) + bar_width)
ax.set_xticklabels([f'z={h}m' for h in heights])
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4b: Effective weight fraction
ax = axes[1]

# Calculate effective weight fraction for WEAK cases
# σ_obs = -ρ*g*(h_mpm_above + α*h_xpbd_above)
# At z=50m (middle of bottom): h_mpm_above = 50m, h_xpbd_above = 100m
# σ_obs = -ρ*g*(50 + α*100)
# α = (σ_obs/(-ρ*g) - 50) / 100

z_sample = 50
for name in ['WEAK_SAME_DENSITY', 'WEAK_LIGHT']:
    if name not in data:
        continue
    d = data[name]
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    
    # Sample at different heights in bottom block
    heights_sample = np.arange(10, 100, 10)
    alphas = []
    
    for h in heights_sample:
        mask = (z >= h - 5) & (z < h + 5)
        if mask.sum() > 0:
            sigma_obs = sigma['sigma_zz'][mask].mean()
            h_mpm_above = H_bottom - h  # MPM material above this point
            h_xpbd_above = H_bottom  # All XPBD is above z=100
            
            # σ_obs = -ρ*g*(h_mpm + α*h_xpbd)
            # α = (σ_obs/(-ρ*g) - h_mpm) / h_xpbd
            alpha = (sigma_obs / (-rho_heavy * g) - h_mpm_above) / h_xpbd_above
            alphas.append(alpha)
        else:
            alphas.append(np.nan)
    
    ax.plot(heights_sample, alphas, 'o-', label=name, color=colors[name], markersize=6)

# Reference lines
ax.axhline(y=1.0, color='blue', linestyle='--', label='Full weight (α=1)', alpha=0.7)
ax.axhline(y=0.0, color='red', linestyle='--', label='Weightless (α=0)', alpha=0.7)
ax.axhline(y=0.1, color='orange', linestyle=':', label='10% weight (α=0.1)', alpha=0.7)

ax.set_xlabel('Height z in bottom block (m)', fontsize=12)
ax.set_ylabel('Effective Weight Fraction α', fontsize=12)
ax.set_title('XPBD Effective Weight Contribution', fontsize=14)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.2, 1.3])

plt.tight_layout()
plt.savefig('weight_contribution_analysis.png', dpi=150)
print("Saved: weight_contribution_analysis.png")

# ============================================================================
# NUMERICAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

print(f"\n{'Case':<25} {'σ_zz at z=50m (MPa)':<20} {'Expected (MPa)':<20} {'Error (%)':<15}")
print("-" * 80)

for name, d in data.items():
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    mask = (z >= 40) & (z < 60)  # Match original script: z=40-60m
    
    if mask.sum() > 0:
        sigma_obs = sigma['sigma_zz'][mask].mean() / 1e6
        
        # Expected value
        if name == 'SAME':
            expected = -rho_heavy * g * (H_total - 50) / 1e6
        elif name == 'WEAK_LIGHT':
            # Top at 300 kg/m³, bottom at 3000 kg/m³
            expected = -(rho_light * g * H_bottom + rho_heavy * g * (H_bottom - 50)) / 1e6
        else:  # WEAK_SAME_DENSITY with full weight
            expected = -rho_heavy * g * (H_total - 50) / 1e6
        
        error = (sigma_obs - expected) / expected * 100
        print(f"{name:<25} {sigma_obs:<20.3f} {expected:<20.3f} {error:<15.1f}")

# Calculate effective weight fractions
print(f"\n{'Case':<25} {'Effective Weight Fraction α':<30}")
print("-" * 55)

for name in ['WEAK_SAME_DENSITY', 'WEAK_LIGHT']:
    if name not in data:
        continue
    d = data[name]
    z = d['positions'][:, 2]
    sigma = compute_stress_components(d)
    
    mask = (z >= 40) & (z < 60)  # Match original script
    if mask.sum() > 0:
        sigma_obs = sigma['sigma_zz'][mask].mean()
        h_mpm_above = 50  # 50m of MPM above z=50
        h_xpbd_above = 100  # 100m of XPBD above
        alpha = (sigma_obs / (-rho_heavy * g) - h_mpm_above) / h_xpbd_above
        
        interpretation = ""
        if alpha < 0.05:
            interpretation = "(WEIGHTLESS)"
        elif alpha > 0.95:
            interpretation = "(FULL WEIGHT)"
        elif 0.05 <= alpha <= 0.15:
            interpretation = "(~10% weight as expected for LIGHT)"
        else:
            interpretation = f"({alpha*100:.0f}% weight)"
        
        print(f"{name:<25} {alpha:<10.3f} {interpretation}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Final interpretation based on results
if 'WEAK_SAME_DENSITY' in data and 'WEAK_LIGHT' in data:
    d_same = data['WEAK_SAME_DENSITY']
    d_light = data['WEAK_LIGHT']
    
    z_same = d_same['positions'][:, 2]
    z_light = d_light['positions'][:, 2]
    
    sigma_same = compute_stress_components(d_same)
    sigma_light = compute_stress_components(d_light)
    
    mask_same = (z_same >= 45) & (z_same < 55)
    mask_light = (z_light >= 45) & (z_light < 55)
    
    if mask_same.sum() > 0 and mask_light.sum() > 0:
        s_same = sigma_same['sigma_zz'][mask_same].mean()
        s_light = sigma_light['sigma_zz'][mask_light].mean()
        
        alpha_same = (s_same / (-rho_heavy * g) - 50) / 100
        alpha_light = (s_light / (-rho_heavy * g) - 50) / 100
        
        print(f"\nXPBD Weight Analysis at z=50m:")
        print(f"  WEAK_SAME_DENSITY effective weight: {alpha_same*100:.1f}%")
        print(f"  WEAK_LIGHT effective weight: {alpha_light*100:.1f}%")
        
        if alpha_same < 0.1:
            print(f"\n  ✗ XPBD particles appear WEIGHTLESS")
            print(f"    They do not contribute mass to the MPM stress field")
        elif alpha_same > 0.9:
            print(f"\n  ✓ XPBD particles have FULL WEIGHT")
            ratio = alpha_light / alpha_same
            if 0.08 < ratio < 0.12:
                print(f"    ✓ Density ratio ({ratio:.2f}) matches expected (0.10)")
            else:
                print(f"    ✗ Density ratio ({ratio:.2f}) does NOT match expected (0.10)")
        else:
            print(f"\n  ? XPBD particles have PARTIAL WEIGHT ({alpha_same*100:.0f}%)")

print("\n" + "=" * 70)
plt.show()

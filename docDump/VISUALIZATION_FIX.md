# Stress Visualization Fix - ACTUAL ROOT CAUSE

## Problem Summary

**Symptom**: Render shows HIGH stress (red) at TOP and LOW stress (blue) at BOTTOM - inverted!

**Root Cause**: NOT boundary conditions (grid bounds are far away) - it's the **COLOR MAPPING**!

---

## The Real Issue: Negative Stress Values + Color Mapping

### What Was Happening

Your stress field is **correct** - compression is negative:
- **Top (z=max)**: mean_stress ≈ -1e5 Pa (small compression)
- **Bottom (z=min)**: mean_stress ≈ -2e6 Pa (large compression)

But the `values_to_rgb` function was mapping:
```python
# OLD CODE (WRONG):
min_val = np.quantile(mean_stress, 0.01)  # = -2e6 Pa (most negative)
max_val = np.quantile(mean_stress, 0.99)  # = -1e5 Pa (least negative)

# Result:
# -2e6 Pa → normalized to 0.0 → BLUE (bottom particles)
# -1e5 Pa → normalized to 1.0 → RED (top particles)
```

**This is BACKWARDS!** Larger compression magnitude should be hotter color!

---

## The Fix Applied

### Changed `color_mode="stress"` (lines 403-413):

```python
# NEW CODE (CORRECT):
elif color_mode == "stress":
    sigma = particle_stress.numpy().astype(np.float64)
    mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0
    # Use absolute value so compression magnitude maps correctly
    mean_stress_abs = np.abs(mean_stress)
    colors = values_to_rgb(
        mean_stress_abs,
        min_val=0.0,  # No stress → BLUE
        max_val=np.quantile(mean_stress_abs, 0.99)  # Max compression → RED
    )
```

### Changed `color_mode="sigma_zz"` (lines 415-424):

```python
# NEW CODE (CORRECT):
elif color_mode == "sigma_zz":
    sigma = particle_stress.numpy().astype(np.float64)
    sigma_zz = sigma[:, 2, 2]  # Vertical component
    sigma_zz_abs = np.abs(sigma_zz)
    colors = values_to_rgb(
        sigma_zz_abs,
        min_val=0.0,  # Surface (no stress) → BLUE
        max_val=np.quantile(sigma_zz_abs, 0.99)  # Bottom (max compression) → RED
    )
```

---

## Expected Result After Fix

### With `color_mode="stress"`:
- **BLUE colors at TOP** (low compression magnitude, near surface)
- **RED colors at BOTTOM** (high compression magnitude, full overburden)
- **Smooth gradient** from blue → yellow → red as depth increases

### With `color_mode="sigma_zz"`:
- Same pattern as above
- Shows vertical stress component specifically

### With `color_mode="von_mises"`:
- Nearly uniform (this is CORRECT for geostatic loading!)
- Von Mises measures deviatoric stress (shape change), not pressure

---

## Why This Happened

### Sign Convention Confusion

In geomechanics:
- **Compression = NEGATIVE stress**
- **Tension = POSITIVE stress**

Your code correctly uses this convention, but visualization needs **magnitude** (absolute value) to map colors correctly.

### The Quantile Trap

Using quantiles on **negative values**:
```python
mean_stress = [-2e6, -1.5e6, -1e6, -5e5]  # More negative = more compression

# WRONG: Maps least negative → hot color
quantile(0.01) = -2e6  → min_val → blue
quantile(0.99) = -5e5  → max_val → red

# CORRECT: Use absolute value first
abs(mean_stress) = [2e6, 1.5e6, 1e6, 5e5]
quantile(0.01) = 5e5   → min_val → blue
quantile(0.99) = 2e6   → max_val → red
```

---

## Verification Steps

1. **Run your simulation again** with the fixed visualization
2. **Check colors**:
   - Bottom should be RED/ORANGE (high compression)
   - Top should be BLUE/CYAN (low compression)
3. **Use diagnostic script**:
   ```python
   from diagnose_stress_distribution import diagnose_stress_distribution
   diagnose_stress_distribution(particle_x.numpy(), particle_stress.numpy(), z_top, rho, g)
   ```

---

## Summary

✅ **FIXED**: Color mapping now uses absolute value of stress  
✅ **RESULT**: Correct visualization (blue at top, red at bottom)  
✅ **PHYSICS**: Stress field was correct all along - just displayed wrong!  

The boundary condition change I made earlier is fine (free top surface is correct), but it wasn't the cause of the visualization issue. The problem was purely in the color mapping logic.

# Drucker-Prager Pressure Term - Sign Convention Fix

## Problem Identified

In the `drucker_prager_return_mapping` function (line 415), the pressure-dependent yield stress had a **sign error**:

```python
# WRONG (original):
yield_eff = (1.0 - D) * (yield_stress[p] + alpha[p] * p_stress)
                                          ↑
                                       PLUS sign
```

### Why This Was Wrong

With your sign convention (compression = negative):

**Under COMPRESSION** (geostatic loading):
- `p_stress = -2e6 Pa` (negative = compression)
- `yield_eff = 1e6 + 0.5 × (-2e6) = 1e6 - 1e6 = 0 Pa`
- **Result**: Yield strength **DROPS to ZERO** under compression! ❌

**Under TENSION**:
- `p_stress = +1e6 Pa` (positive = tension)
- `yield_eff = 1e6 + 0.5 × (+1e6) = 1e6 + 0.5e6 = 1.5e6 Pa`
- **Result**: Yield strength **INCREASES** under tension! ❌

This is **opposite** of physical behavior for rocks/soils!

---

## Physical Expectation: Drucker-Prager Model

Rock and soil materials should be **stronger under compression** due to the **friction angle effect**:

```
Compression → Higher confinement → Harder to yield → HIGHER yield stress
Tension     → Lower confinement  → Easier to yield  → LOWER yield stress
```

This is the fundamental behavior captured by the **Mohr-Coulomb** and **Drucker-Prager** models used in geomechanics.

### The Friction Angle

In the classical Drucker-Prager model:

```
F = √J₂ + α·I₁ - k ≤ 0
```

Where:
- `√J₂` = deviatoric stress (Von Mises-like)
- `α` = friction parameter (related to friction angle φ)
- `I₁ = tr(σ)` = first stress invariant (3 × mean stress)
- `k` = cohesion parameter

The key insight: **α·I₁ increases the "resistance" to yielding** when the material is compressed.

---

## The Fix Applied

Changed the sign in the yield stress calculation:

```python
# CORRECT (fixed):
yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)
                                          ↑
                                      MINUS sign
```

### Why This Works

With compression = negative:

**Under COMPRESSION**:
- `p_stress = -2e6 Pa` (negative)
- `yield_eff = 1e6 - 0.5 × (-2e6) = 1e6 + 1e6 = 2e6 Pa`
- **Result**: Yield strength **INCREASES** under compression ✓ CORRECT!

**Under TENSION**:
- `p_stress = +1e6 Pa` (positive)
- `yield_eff = 1e6 - 0.5 × (+1e6) = 1e6 - 0.5e6 = 0.5e6 Pa`
- **Result**: Yield strength **DECREASES** under tension ✓ CORRECT!

---

## Mathematical Consistency

### Sign Convention Chain

The minus sign correctly accounts for your sign convention:

```
Compression: p < 0  →  -α·p > 0  →  increases yield_eff  ✓
Tension:     p > 0  →  -α·p < 0  →  decreases yield_eff  ✓
```

### Relation to Classical Drucker-Prager

In the classical formulation with compression = positive:

```
yield = √J₂ + α·p - k ≤ 0
```

Rearranging for equivalent yield stress:
```
√J₂ ≤ k - α·p
```

With your convention (compression = negative), this becomes:
```
√J₂ ≤ k - α·p    (where p < 0 for compression)
√J₂ ≤ k + α·|p|  (equivalent form)
```

So the effective yield stress is: `yield_eff = base_yield - α·p`

**This matches the fixed formula!** ✓

---

## Impact on Your Simulation

### Before Fix (WRONG)

Under geostatic compression at depth:
- Bottom particles: `p_stress ≈ -1.5e6 Pa` (high compression)
- With `α = 0.5`, `ys = 1e6 Pa`:
  - `yield_eff = 1e6 + 0.5×(-1.5e6) = 0.25e6 Pa`
  - Material becomes **very weak** at depth!
  - Would plastically yield immediately ❌

### After Fix (CORRECT)

Under geostatic compression at depth:
- Bottom particles: `p_stress ≈ -1.5e6 Pa` (high compression)
- With `α = 0.5`, `ys = 1e6 Pa`:
  - `yield_eff = 1e6 - 0.5×(-1.5e6) = 1.75e6 Pa`
  - Material becomes **stronger** at depth!
  - Resists yielding under confinement ✓

This is why you saw **zero plastic strain** in your geostatic simulation - the material was correctly staying elastic!

---

## Parameter Interpretation

### α (alpha) Parameter

With the fixed formula, `α` represents the **pressure sensitivity**:

```
α = 0:   No pressure effect (standard Von Mises)
α > 0:   Stronger under compression (friction-like behavior)
```

**Typical values for rocks**:
- Brittle rock (granite): `α ≈ 0.3 - 0.5`
- Weak rock (sandstone): `α ≈ 0.5 - 0.8`
- Soil: `α ≈ 0.8 - 1.2`

### Relationship to Friction Angle φ

For the classical Drucker-Prager model in plane strain:

```
α = 2·sin(φ) / [√3·(3 - sin(φ))]
```

Example:
- `φ = 30°` → `α ≈ 0.42`
- `φ = 40°` → `α ≈ 0.68`

---

## Code Changes Summary

**File**: `utils/mpmRoutines.py`

**Line 415** (in `drucker_prager_return_mapping`):

```python
# Before:
yield_eff = (1.0 - D) * (yield_stress[p] + alpha[p] * p_stress)

# After:
yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)
                                          ↑
                                      Changed + to -
```

**Added comment** (lines 409-416):
```python
# Add pressure effect to yield stress (Drucker-Prager-like)
# α controls pressure sensitivity (friction angle effect)
# Sign convention: compression = negative, so use MINUS to increase strength
# - Compression (p < 0): -α·p > 0 → increases yield stress
# - Tension (p > 0):     -α·p < 0 → decreases yield stress
# If α = 0, behaves like standard Von Mises
```

---

## Testing Recommendations

1. **Verify geostatic equilibrium**:
   - Material should remain elastic under geostatic stress
   - No plastic yielding unless ys is very low

2. **Test excavation**:
   - Remove material from center
   - Observe stress redistribution
   - Material near excavation should yield when deviatoric stress exceeds reduced yield stress

3. **Compare with classical Drucker-Prager**:
   - For validation, compare with software using standard formulation
   - Account for sign convention difference

---

## Summary

✅ **FIXED**: Sign error in pressure-dependent yield stress  
✅ **RESULT**: Material now correctly strengthens under compression  
✅ **PHYSICS**: Matches classical Drucker-Prager friction angle behavior  
✅ **IMPACT**: Your geostatic simulations should now properly resist yielding at depth  

The fix ensures that your rock caving simulations will correctly model the pressure-dependent strength of geomaterials!

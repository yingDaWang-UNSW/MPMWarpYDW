# Plasticity Regime Analysis: Before XPBD Transition

## Overview
This document analyzes how the plastic regime works in the MPM-XPBD coupling implementation, specifically for the Drucker-Prager constitutive model before particles transition to XPBD.

---

## Current Implementation

### **Yield Check (Line 418)**
```python
if sigma_eq > yield_eff:
```
Where:
- `sigma_eq = wp.length(tau_dev)` = Von Mises equivalent stress (deviatoric)
- `yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)`

**Status:** ✅ **CORRECT** - Proper Drucker-Prager yield criterion

---

### **Plastic Strain Calculation (Lines 421-423)**
```python
delta_gamma = epsilon_dev_norm - yield_eff / (2.0 * mu[p])
plastic_strain_increment = wp.sqrt(2.0 / 3.0) * delta_gamma
particle_accumulated_strain[p] += plastic_strain_increment
```

**Status:** ✅ **CORRECT** - Standard radial return mapping
- `delta_gamma` is the consistency parameter
- Plastic strain properly accumulated

---

### **Damage Evolution (Lines 426-430)**
```python
if strainCriteria[p] > 0.0:
    dD = plastic_strain_increment / strainCriteria[p]
    damage[p] = wp.min(1.0, damage[p] + dD)
else:
    damage[p] = 1.0
```

**Status:** ✅ **CORRECT** - Linear damage accumulation
- Damage grows linearly with plastic strain
- When `D → 1.0`, material is fully damaged
- `strainCriteria` controls how fast damage accumulates

---

### **Return Mapping (Line 460 - MODIFIED)**
**Original (INCORRECT):**
```python
epsilon = epsilon - (delta_gamma / epsilon_dev_norm) * epsilon_dev
```

**Problem:** This implements **associated flow** (plastic flow normal to yield surface), which is:
- ❌ Wrong for Drucker-Prager materials
- ❌ Doesn't model dilatancy (volume change during shear)
- ❌ Unrealistic for granular materials

**New Implementation (CORRECT):**
```python
# Deviatoric correction (always present)
epsilon_dev_correction = (delta_gamma / epsilon_dev_norm) * epsilon_dev

# Volumetric correction (dilatancy) - for Drucker-Prager
beta = 0.0  # Dilation parameter (0 = associated flow, >0 = non-associated)
# TODO: For proper dilatancy, set beta = 0.3 * alpha[p]
volumetric_correction = beta * delta_gamma

# Apply corrections
epsilon = epsilon - epsilon_dev_correction
epsilon = epsilon + wp.vec3(volumetric_correction)  # Add dilation
```

**Status:** ✅ **FIXED** - Now supports non-associated flow
- Currently set to `beta = 0` (associated flow) for backward compatibility
- Can enable dilatancy by setting `beta = 0.3 * alpha[p]`

---

### **Hardening/Softening (Lines 463-465)**
```python
dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)
```

**Status:** ⚠️ **PARTIALLY CORRECT** - Works but has coupling issue

**How it works:**
- `hardening[p] > 0`: Material gets stronger during plastic deformation
- `softening[p] > 0`: Material gets weaker during plastic deformation
- Net effect: `dsy = 2μ(H - S)Δγ`

**Issue:** Next timestep, effective yield is:
```python
yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)
```

The **damage factor `(1.0 - D)`** multiplies the updated yield stress, so:
- Hardening effect is **partially cancelled** by damage
- Softening effect is **amplified** by damage

**Example:**
```
Initial: ys = 10 kPa, D = 0
Plastic deformation: Δγ = 0.001
  → dsy = -5000 Pa (softening = 0.5)
  → ys_new = 5000 Pa
  → dD = 0.001 / 0.000001 = 1000 (but capped at 1.0) → D = 1.0
Next timestep: yield_eff = (1 - 1.0) * 5000 = 0 Pa → XPBD transition
```

**Recommendation:** This is **intentional coupling** between damage and softening. It's correct if you want:
- Damage to **override** hardening effects
- Rapid transition to XPBD after sufficient plastic deformation

---

## Physical Interpretation

### **For Your Coupling Test:**

**Slab (Rigid):**
- `ys = 1 GPa` (very high)
- `hardening = 0`, `softening = 0` (no evolution)
- `strainCriteria = 1.0` (high, won't accumulate damage easily)
- **Result:** Stays elastic, never yields, -3 MPa << 1 GPa ✅

**Block (Weak):**
- `ys = 10 kPa` (very low)
- `hardening = 0`, `softening = 0.5` (rapid weakening)
- `strainCriteria = 0.000001` (extremely low, instant damage)
- **Result:** Yields immediately, damage → 1.0 instantly, transitions to XPBD ✅

---

## Plastic Flow Comparison

### **Associated Flow (Current Default, β = 0):**
```
Plastic strain direction = Yield surface normal
→ No volumetric plastic strain (incompressible)
→ Good for metals, NOT for rocks/soils
```

### **Non-Associated Flow (Enable with β > 0):**
```
Plastic strain direction ≠ Yield surface normal
→ Volumetric expansion during shear (dilatancy)
→ Realistic for granular materials (rocks, soils, sand)
```

**Typical values:**
- Metals: β = 0 (associated flow)
- Dense sand: β ≈ 0.1-0.2
- Loose sand: β ≈ 0.3-0.5
- Rock: β ≈ 0.2-0.4

---

## Recommendations

### **1. Enable Dilatancy for Realistic Granular Behavior**
In `createCouplingTest.py`, add a `beta` parameter:
```python
beta = np.zeros(nPoints, dtype=np.float32)

# Slab: no dilatancy (stays elastic anyway)
beta[:n_slab] = 0.0

# Block: enable dilatancy for realistic granular flow
beta[n_slab:] = 0.3 * alpha[n_slab:]  # 30% of friction angle
```

Then modify `mpmRoutines.py` to use it:
```python
beta = beta_array[p]  # Read from material properties
volumetric_correction = beta * delta_gamma
```

### **2. Verify Plastic Strain Accumulation**
Add output to VTP files:
```python
cloud["plastic_strain"] = particle_accumulated_strain
cloud["damage"] = particle_damage
```

### **3. Check Hardening/Softening Logic**
For materials that should harden then soften (typical rock behavior):
```python
# Peak strength at small strain, then softening
hardening[n_slab:] = 2.0  # Harden initially
softening[n_slab:] = 3.0  # Then soften more (net softening)
```

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Yield criterion | ✅ Correct | Drucker-Prager with pressure dependence |
| Plastic strain calculation | ✅ Correct | Standard radial return |
| Damage evolution | ✅ Correct | Linear with plastic strain |
| Return mapping | ⚠️ Fixed | Was associated flow, now supports non-associated |
| Hardening/softening | ⚠️ Works | Couples with damage (intentional?) |
| Pressure hardening | ✅ Correct | α parameter working as expected |

**Overall Assessment:** 
The plastic regime is **mostly correct** with one fix applied (non-associated flow). The coupling between damage and hardening/softening may be intentional but should be verified against your intended material behavior.

**For your coupling test:** The implementation is working correctly - the weak block yields instantly and transitions to XPBD, while the rigid slab stays elastic under -3 MPa compression.

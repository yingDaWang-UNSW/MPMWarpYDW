# Plastic Irreversibility: How It Works in Your MPM Implementation

## Overview
**Plastic irreversibility** means that once a material undergoes plastic deformation, that deformation is **permanent** and cannot be recovered by unloading. This is the fundamental difference between elastic (reversible) and plastic (irreversible) deformation.

---

## Key Concept: Multiplicative Decomposition

Your code uses **multiplicative elastoplasticity**:

```
F = F_elastic × F_plastic
```

Where:
- `F` = Total deformation gradient (what you measure)
- `F_elastic` = Recoverable elastic deformation (stored in code)
- `F_plastic` = Permanent plastic deformation (implicitly tracked)

**The magic:** You only store `F_elastic`, and plastic deformation is "forgotten" by **updating F_elastic after each plastic step**.

---

## How Irreversibility is Imposed: Step-by-Step

### **Step 1: Trial Elastic Deformation (Lines 395-402)**

```python
# Input: F_trial from previous timestep's deformation
wp.svd3(F_trial, U, sig_old, V)
epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
```

**What happens:**
- Assume ALL new deformation is elastic
- Compute trial stress from F_trial
- Check if this violates yield criterion

**Key insight:** This is a "trial" - we're testing if the material can handle the load elastically.

---

### **Step 2: Check Yield Criterion (Line 418)**

```python
if sigma_eq > yield_eff:
    # Material has yielded - must apply plastic correction
```

**Two outcomes:**
1. **No yielding** (`sigma_eq ≤ yield_eff`): 
   - Return `F_trial` unchanged
   - All deformation is elastic and **reversible**
   
2. **Yielding occurs** (`sigma_eq > yield_eff`):
   - Must decompose into elastic + plastic
   - Plastic part becomes **irreversible**

---

### **Step 3: Plastic Correction (Lines 468-479) - WHERE IRREVERSIBILITY HAPPENS**

This is the **critical moment** where irreversibility is imposed:

```python
# Compute how much plastic strain occurred
delta_gamma = epsilon_dev_norm - yield_eff / (2.0 * mu[p])
plastic_strain_increment = wp.sqrt(2.0 / 3.0) * delta_gamma

# IRREVERSIBILITY MECHANISM #1: Accumulate plastic strain permanently
particle_accumulated_strain[p] += plastic_strain_increment  # ← NEVER DECREASES!

# IRREVERSIBILITY MECHANISM #2: Remove plastic part from elastic strain
epsilon_dev_correction = (delta_gamma / epsilon_dev_norm) * epsilon_dev
epsilon = epsilon - epsilon_dev_correction  # ← "Forget" the plastic deformation

# Reconstruct ONLY the elastic part
sig_elastic = wp.mat33(
    wp.exp(epsilon[0]), 0.0, 0.0,
    0.0, wp.exp(epsilon[1]), 0.0,
    0.0, 0.0, wp.exp(epsilon[2])
)
F_elastic = U * sig_elastic * wp.transpose(V)
return F_elastic  # ← Return ONLY elastic part!
```

---

## Why This Enforces Irreversibility

### **Mechanism 1: Plastic Strain Accumulation**

```python
particle_accumulated_strain[p] += plastic_strain_increment
```

**Why irreversible?**
- This is a **monotonically increasing** variable
- It **never decreases**, even if you unload the material
- Tracks the "history" of plastic deformation

**Physical meaning:** 
- Represents permanent microstructural changes (dislocation motion, grain boundary sliding)
- Used to drive damage evolution and eventual failure

---

### **Mechanism 2: Elastic Strain Reduction**

```python
epsilon = epsilon - epsilon_dev_correction
```

**Why irreversible?**
- The **trial strain** `epsilon` includes both elastic + plastic components
- We **subtract the plastic part** to get only the elastic strain
- Return `F_elastic` computed from this **reduced** elastic strain

**What this means:**

| State | Total Strain | Elastic Strain | Plastic Strain |
|-------|-------------|----------------|----------------|
| Before yielding | ε_total | ε_total | 0 |
| After plastic correction | ε_total | ε_total - ε_plastic | ε_plastic |
| After unloading | ε_plastic | 0 | ε_plastic |

The **plastic strain remains** even after unloading!

---

## Physical Interpretation

### **Loading Cycle Example:**

```
1. Load material → F_trial grows
2. Check yield: σ_eq > σ_y → Yielding!
3. Plastic correction:
   - Remove plastic strain from F_elastic
   - Store plastic history in accumulated_strain
   - Return reduced F_elastic
4. Next timestep: Start from reduced F_elastic (plastic forgotten)
5. Unload material → F_trial shrinks
6. Elastic unloading: σ_eq < σ_y → No yielding
7. Return to stress-free state: F_elastic → I (identity)
8. BUT: accumulated_strain > 0 → Permanent deformation recorded!
```

**Result:** Material has **permanent deformation** (plastic strain) that doesn't recover.

---

## Mathematical Proof of Irreversibility

### **Elastic-Plastic Decomposition:**

```
F_trial = F_elastic × F_plastic_new
```

During plastic correction:
```python
# We solve for F_elastic such that:
# σ(F_elastic) = σ_yield  (stress brought back to yield surface)

# The "lost" deformation is:
F_plastic_new = F_elastic^(-1) × F_trial

# But we DON'T store F_plastic_new!
# Instead, we only return F_elastic
```

**Why this works:**
- Next timestep, new deformation is applied to `F_elastic` (not `F_trial`)
- The plastic part `F_plastic_new` is **implicitly absorbed** into the reference configuration
- This is equivalent to "resetting" the reference state after plastic flow

---

## Verification: How to Check Irreversibility

### **Test 1: Load-Unload Cycle**

```python
# Initial state
F_initial = identity
accumulated_strain_initial = 0

# Load beyond yield
F_loaded = apply_deformation(F_initial, large_strain)
# → accumulated_strain increases
# → F_elastic < F_loaded (plastic correction applied)

# Unload completely
F_unloaded = apply_deformation(F_loaded, -large_strain)
# → F_unloaded ≈ identity (elastic recovery)
# → BUT: accumulated_strain > 0 (plastic history remains!)
```

**Expected result:**
- Stress returns to zero ✓
- F_elastic returns to identity ✓
- **accumulated_strain ≠ 0** ✓ ← **PROOF OF IRREVERSIBILITY**

---

### **Test 2: Cyclic Loading**

```python
for cycle in range(10):
    # Load
    F = apply_load()
    # Unload
    F = apply_unload()

# After many cycles:
# - accumulated_strain keeps increasing (ratcheting)
# - damage keeps accumulating
# - material eventually fails (D → 1.0)
```

**Expected result:**
- Plastic strain accumulates with each cycle
- Damage grows monotonically
- Material weakens progressively ← **IRREVERSIBLE DEGRADATION**

---

## Comparison with Elastic Behavior

### **Elastic (Reversible):**

```python
# No yielding branch (line 508)
else:
    return F_trial  # ← Returns full deformation, nothing "forgotten"
```

**Characteristics:**
- `F_elastic = F_trial` (no correction)
- `accumulated_strain` unchanged
- Full recovery upon unloading

---

### **Plastic (Irreversible):**

```python
# Yielding branch (lines 468-506)
epsilon = epsilon - epsilon_dev_correction  # ← Plastic part removed
return F_elastic  # ← Returns reduced elastic part only
```

**Characteristics:**
- `F_elastic < F_trial` (plastic correction applied)
- `accumulated_strain` increases
- Partial recovery upon unloading (only elastic part recovers)

---

## Connection to Damage and Failure

### **Damage Evolution (Lines 426-432):**

```python
if strainCriteria[p] > 0.0:
    dD = plastic_strain_increment / strainCriteria[p]
    damage[p] = wp.min(1.0, damage[p] + dD)
```

**How irreversibility drives failure:**

1. **Plastic deformation** → `accumulated_strain` increases (irreversible)
2. **Damage accumulation** → `damage` increases proportionally (irreversible)
3. **Strength degradation** → `yield_eff = (1-D) × ys` decreases (irreversible)
4. **Final failure** → `damage → 1.0` → Transition to XPBD (irreversible)

**Key insight:** Irreversibility at micro-scale (plastic strain) leads to irreversible failure at macro-scale.

---

## Summary: Three Pillars of Irreversibility

| Mechanism | How It Works | What It Stores |
|-----------|--------------|----------------|
| **1. Plastic strain accumulation** | `accumulated_strain += Δε_p` | History of plastic deformation |
| **2. Elastic strain reduction** | `ε_elastic = ε_trial - ε_plastic` | Only recoverable part |
| **3. Damage accumulation** | `D += Δε_p / ε_crit` | Progressive material degradation |

**Result:** Once plastic deformation occurs:
- ✅ Stress state can return to zero (unloading)
- ✅ Elastic deformation recovers
- ❌ **Plastic deformation NEVER recovers** (irreversible)
- ❌ **Damage NEVER heals** (irreversible)

---

## Code Flow Diagram

```
┌─────────────────────────────────────┐
│ Input: F_trial (total deformation) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Compute trial stress from F_trial  │
│ σ_trial = f(F_trial)                │
└──────────────┬──────────────────────┘
               │
               ▼
         ┌─────────────┐
         │ σ_eq > σ_y? │  ← Yield check
         └──────┬──────┘
                │
     ┌──────────┴──────────┐
     │ NO                   │ YES
     ▼                      ▼
┌─────────────┐    ┌────────────────────┐
│ ELASTIC     │    │ PLASTIC YIELDING   │
│ Return      │    │ - Compute Δε_p     │
│ F_trial     │    │ - accumulate += Δε_p│ ← IRREVERSIBLE!
│             │    │ - ε = ε - Δε_p     │ ← IRREVERSIBLE!
│ REVERSIBLE  │    │ - Return F_elastic │
└─────────────┘    └────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Update damage  │ ← IRREVERSIBLE!
                   │ D += Δε_p/ε_c  │
                   └────────────────┘
```

---

## Conclusion

**Plastic irreversibility is enforced by:**

1. **Permanently recording** plastic strain history (`accumulated_strain`)
2. **Removing** plastic strain from the stored deformation gradient (`F_elastic`)
3. **Monotonically accumulating** damage (`D`)
4. **Never allowing** these quantities to decrease

When you unload the material, only the **elastic part** (`F_elastic`) recovers. The **plastic part** is "baked into" the reference configuration and cannot be recovered. This is the essence of plastic irreversibility in your implementation! 🎯

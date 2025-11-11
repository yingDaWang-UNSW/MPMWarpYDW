# Hardening-Softening: Net Effect vs. Strain-Dependent Behavior

## Your Question: Should They Apply at Different Strains?

**Short Answer: YES!** The current implementation `(H - S)` is overly simplistic for realistic materials.

## Current Implementation (Lines 466-467)

```python
dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)
```

**Problem:** This gives a constant net effect throughout all plastic deformation:
- If H > S: Material hardens forever
- If H < S: Material softens forever
- No transition from hardening to softening behavior

## Realistic Material Behavior

Real materials (especially rocks) show **strain-dependent transitions**:

```
      Yield Stress
           │
           │     ╱╲  Peak (hardening phase ends)
           │    ╱  ╲
           │   ╱    ╲___  Residual (softening stabilizes)
     σ_y0  │──╱
           │ ╱
           └─────────────── Plastic Strain
             ↑      ↑
          Hardening Softening
           Phase    Phase
```

### Three Stages:

1. **Early Stage (0-3% strain)**: 
   - H dominates → net hardening
   - Pore collapse, grain interlocking
   
2. **Peak Stage (3-5% strain)**:
   - H ≈ S → plateau
   - Maximum strength
   
3. **Post-Peak (>5% strain)**:
   - S dominates → net softening
   - Microcrack coalescence, strain localization

## Why Current Model Doesn't Capture This

Your code has damage that multiplies yield stress:
```python
yield_eff = (1.0 - D) * yield_stress[p]
```

And H/S that adds to yield stress:
```python
yield_stress[p] += 2μ(H - S)Δγ
```

**Issue:** If you set H=0.3, S=0.1:
- Net effect is **always** +0.2 (hardening)
- Material will harden indefinitely
- Doesn't capture peak → softening transition

**The damage D handles softening, but:**
- Damage is independent of H/S
- You can't control WHEN softening starts relative to hardening

## Better Approaches

### Option 1: Strain-Dependent Hardening/Softening

Make H and S functions of accumulated plastic strain:

```python
@wp.func
def get_hardening_rate(plastic_strain: float, 
                       H_initial: float, 
                       strain_peak: float) -> float:
    """
    Hardening decreases with plastic strain, goes to zero at peak.
    """
    if plastic_strain < strain_peak:
        # Linear decay: H_max at zero strain, zero at peak
        return H_initial * (1.0 - plastic_strain / strain_peak)
    else:
        return 0.0

@wp.func
def get_softening_rate(plastic_strain: float, 
                      S_max: float, 
                      strain_peak: float,
                      strain_residual: float) -> float:
    """
    Softening activates after peak, saturates at residual.
    """
    if plastic_strain < strain_peak:
        return 0.0  # No softening before peak
    elif plastic_strain < strain_residual:
        # Ramp up from 0 to S_max
        progress = (plastic_strain - strain_peak) / (strain_residual - strain_peak)
        return S_max * progress
    else:
        return S_max  # Full softening after residual

# Then in return mapping:
H_current = get_hardening_rate(particle_accumulated_strain[p], 
                               hardening[p], 
                               strain_peak[p])
S_current = get_softening_rate(particle_accumulated_strain[p], 
                               softening[p], 
                               strain_peak[p], 
                               strain_residual[p])

dsy = 2.0 * mu[p] * (H_current - S_current) * delta_gamma
```

**Benefits:**
- Early stage: H_current > 0, S_current = 0 → net hardening
- Peak stage: H_current = 0, S_current = 0 → constant yield
- Post-peak: H_current = 0, S_current > 0 → net softening

### Option 2: Damage-Modulated Softening

Let damage control the transition:

```python
@wp.func
def get_damage_modulated_rates(D: float, H_base: float, S_base: float):
    """
    Hardening decreases with damage, softening increases.
    """
    H_current = H_base * (1.0 - D)  # Hardening fades as damage grows
    S_current = S_base * D           # Softening grows with damage
    return H_current, S_current

# In return mapping:
D = damage[p]
H_current, S_current = get_damage_modulated_rates(D, hardening[p], softening[p])
dsy = 2.0 * mu[p] * (H_current - S_current) * delta_gamma
```

**Effect:**
- D=0 (intact): H=full, S=0 → net hardening
- D=0.5 (damaged): H=0.5, S=0.5S_base → transition
- D=1 (failed): H=0, S=full → net softening

### Option 3: Separate Hardening and Softening Variables

Track them independently instead of net effect:

```python
# Add new particle arrays:
particle_hardening_state: wp.array(dtype=float)  # Accumulated hardening
particle_softening_state: wp.array(dtype=float)  # Accumulated softening

# In return mapping:
if plastic_strain_increment > 0:
    # Hardening: saturates at some maximum
    if particle_hardening_state[p] < hardening_max[p]:
        dH = hardening_rate[p] * delta_gamma
        particle_hardening_state[p] += dH
    
    # Softening: activates after threshold
    if particle_accumulated_strain[p] > strain_threshold[p]:
        dS = softening_rate[p] * delta_gamma
        particle_softening_state[p] += dS
    
    # Net effect
    yield_stress[p] = yield_base[p] + particle_hardening_state[p] - particle_softening_state[p]
```

### Option 4: Peak Strength Model (Hoek-Brown Style)

Classic rock mechanics approach:

```python
@wp.func
def compute_yield_with_peak(
    plastic_strain: float,
    yield_initial: float,
    yield_peak: float,
    yield_residual: float,
    strain_peak: float,
    strain_residual: float
) -> float:
    """
    Piecewise yield stress evolution:
    - Hardens from initial to peak
    - Softens from peak to residual
    - Stays at residual
    """
    if plastic_strain < strain_peak:
        # Hardening phase
        ratio = plastic_strain / strain_peak
        return yield_initial + (yield_peak - yield_initial) * ratio
    elif plastic_strain < strain_residual:
        # Softening phase
        ratio = (plastic_strain - strain_peak) / (strain_residual - strain_peak)
        return yield_peak - (yield_peak - yield_residual) * ratio
    else:
        # Residual phase
        return yield_residual

# Usage:
yield_eff_no_damage = compute_yield_with_peak(
    particle_accumulated_strain[p],
    yield_stress_initial[p],
    yield_stress_peak[p],
    yield_stress_residual[p],
    strain_peak[p],
    strain_residual[p]
)

# Then apply damage
yield_eff = (1.0 - damage[p]) * yield_eff_no_damage
```

## Comparison: Current vs. Strain-Dependent

### Current Model (Net Effect):
```
Config: H=0.3, S=0.1
Result: Net hardening of 0.2 forever

0% strain:  ys = 10.0 + 0.0 = 10.0 MPa
5% strain:  ys = 10.0 + 0.4 = 10.4 MPa  ← Still hardening!
10% strain: ys = 10.0 + 0.8 = 10.8 MPa  ← Still hardening!
20% strain: ys = 10.0 + 1.7 = 11.7 MPa  ← Still hardening!

Never reaches peak and softens!
```

### With Strain-Dependent (Option 1):
```
Config: H_initial=0.5, S_max=0.3, strain_peak=0.03, strain_residual=0.10

0% strain:   H=0.50, S=0.00 → net +0.50 → ys = 10.0 MPa
1.5% strain: H=0.25, S=0.00 → net +0.25 → ys = 10.3 MPa (hardening)
3% strain:   H=0.00, S=0.00 → net  0.00 → ys = 10.4 MPa (peak)
6.5% strain: H=0.00, S=0.15 → net -0.15 → ys = 10.1 MPa (softening)
10% strain:  H=0.00, S=0.30 → net -0.30 → ys = 9.8 MPa (residual)
15% strain:  H=0.00, S=0.30 → net -0.30 → ys = 9.5 MPa (stable)

Realistic peak → softening → residual behavior!
```

## Recommended Implementation for Your Model

Given you already have **damage**, I recommend **Option 2** (Damage-Modulated):

### Why Option 2?
1. **Simple**: Just modify existing H and S based on damage
2. **Physical**: Damage naturally causes transition from hardening to softening
3. **No new parameters**: Uses existing damage variable
4. **Compatible**: Works with your current damage evolution

### Implementation:

```python
# In von_mises_return_mapping_with_damage_YDW and drucker_prager_return_mapping:

# Before line 466, add:
# Modulate H and S based on damage state
D = damage[p]
H_effective = hardening[p] * (1.0 - D)      # Hardening fades with damage
S_effective = softening[p] * D              # Softening grows with damage

# Then replace line 466:
dsy = 2.0 * mu[p] * (H_effective - S_effective) * delta_gamma
```

### Example with Damage-Modulated:
```
Config: H=0.4, S=0.3, strainCriteria=0.2

0% strain:  D=0.0  → H_eff=0.40, S_eff=0.00 → net +0.40 (hardening)
5% strain:  D=0.25 → H_eff=0.30, S_eff=0.08 → net +0.22 (still hardening)
10% strain: D=0.50 → H_eff=0.20, S_eff=0.15 → net +0.05 (transition)
15% strain: D=0.75 → H_eff=0.10, S_eff=0.23 → net -0.13 (softening)
20% strain: D=1.00 → H_eff=0.00, S_eff=0.30 → net -0.30 (full softening)

Smooth transition controlled by damage!
```

## Configuration Recommendations

### For Current Model (Net Effect):
```json
"hardening": 0.3,    // Always net hardening
"softening": 0.0,    // Let damage handle softening
```
- Simple, but unrealistic
- No peak behavior

### For Damage-Modulated (Recommended):
```json
"hardening": 0.4,    // Full hardening when intact
"softening": 0.3,    // Full softening when fully damaged
"strainCriteria": 0.2
```
- Realistic transition
- Damage controls the shift
- Early: net hardening (+0.4)
- Late: net softening (-0.3)

### For Strain-Dependent (Most Realistic):
```json
"hardening_initial": 0.5,
"softening_max": 0.3,
"strain_peak": 0.03,      // Peak at 3% plastic strain
"strain_residual": 0.10   // Residual at 10% plastic strain
```
- Full control over behavior
- Requires code modification
- Best for rock mechanics

## Summary

**Your insight is correct!** The current `(H - S)` formulation is too simplistic:

1. **Problem**: Net effect is constant throughout deformation
2. **Reality**: Materials transition from hardening → peak → softening
3. **Solution**: Make H and S strain-dependent or damage-dependent

**Quick Fix (Damage-Modulated):**
```python
# Add before line 466 in both return mapping functions:
D = damage[p]
H_eff = hardening[p] * (1.0 - D)
S_eff = softening[p] * D
dsy = 2.0 * mu[p] * (H_eff - S_eff) * delta_gamma
```

This gives you:
- Early stage (D≈0): Hardening dominates
- Transition (D≈0.5): Balanced
- Late stage (D→1): Softening dominates

Much more realistic than constant net effect!

# Summary: Hardening & Softening Parameters

## Quick Answer

**Units:** Both `hardening` and `softening` are **DIMENSIONLESS** [-]

**Meaning:** They control how fast yield stress evolves with plastic deformation:
```python
dsy = 2.0 * mu * (hardening - softening) * delta_gamma
yield_stress += dsy
```

**Can both be on?** **YES!** Very common in geomechanics—they represent competing physical mechanisms.

---

## The Math

### Dimensional Analysis:
```
dsy [Pa] = mu [Pa] × (H - S) [-] × Δγ [-]

Therefore: H and S must be dimensionless
```

### Evolution Rate:
```
Rate of yield stress change = 2μ(H - S)

Example with μ = 42 MPa:
  H=0.3, S=0.0 → +25 MPa per 100% plastic strain
  H=0.3, S=0.1 → +17 MPa per 100% plastic strain
  H=0.0, S=0.0 → 0 MPa (no evolution)
```

---

## Physical Interpretation

### Hardening (H > 0)
**Material strengthens with plastic deformation**

Mechanisms:
- Grain interlocking under compression
- Pore collapse (porosity reduction)
- Dislocation pile-up (metals)
- Friction increase at contacts

### Softening (S > 0)
**Material weakens with plastic deformation**

Mechanisms:
- Microcrack growth and coalescence
- Bond breakage
- Fabric damage
- Strain localization

---

## Your Model Has TWO Softening Mechanisms!

### 1. Damage (Multiplicative):
```python
yield_eff = (1 - D) * yield_stress
```
- Goes from D=0 (intact) to D=1 (failed)
- Multiplies yield stress by (1-D)
- At D=1, effective yield = 0

### 2. Softening Parameter (Additive):
```python
dsy = 2μ(H - S)Δγ
```
- Accumulates over plastic strain
- Can be positive or negative
- Independent of damage

### Combined Effect:
```
Effective yield = (1 - D) × [ys_base + ∫2μ(H-S)dγ - α·p]
                  \_____/   \___________________________/
                  Damage         H/S evolution
```

**Critical Insight:** Damage is **multiplicative** and dominates. Even with hardening, once D→1, effective yield→0.

---

## From Your Analysis

### Current Config (Problem):
```json
"hardening": 0.0,
"softening": 0.0,
"strainCriteria": 0.2  ← You updated this (was 0.05) ✓
```

**Issue:** No mechanism to counteract damage weakening!

### Numerical Results (From Plot):

| Case | At 2.5% strain | At 5% strain |
|------|---------------|-------------|
| H=0, S=0 | 4.97 MPa (D=50%) | 0.05 MPa (D=99%) |
| H=0.3, S=0 | 5.29 MPa (D=50%) | 0.06 MPa (D=99%) |

**Key Finding:** Even with H=0.3, damage STILL drives yield to zero!

---

## Recommended Fixes

### Option 1: Moderate Hardening (Start Here)
```json
"hardening": 0.3,
"softening": 0.0,
"strainCriteria": 0.2
```
- Clean and simple
- Hardening fights damage
- Let damage be the only softening mechanism

### Option 2: Competing Mechanisms (More Realistic)
```json
"hardening": 0.4,
"softening": 0.1,
"strainCriteria": 0.2
```
- Net hardening = 0.3
- Small softening for fabric damage
- More realistic multi-stage behavior

### Option 3: Strong Hardening (Very Confined Rock)
```json
"hardening": 0.5,
"softening": 0.0,
"strainCriteria": 0.3
```
- High confinement scenario
- Very ductile response
- Good for deep underground

---

## Why You Need Both Hardening AND Higher Strain Criteria

With `strainCriteria = 0.05` (5%):
- Damage reaches 100% at only 5% plastic strain
- Material has ZERO strength after this
- Hardening can't help—multiplicative damage wins

With `strainCriteria = 0.2` (20%) AND `hardening = 0.3`:
- Damage accumulates more slowly
- Hardening has time to increase base yield stress
- More controlled failure progression

### Example Evolution:

**Old (H=0, strain_crit=0.05):**
```
0% strain: ys=10 MPa, D=0%   → eff_ys = 10.0 MPa
2.5%:      ys=10 MPa, D=50%  → eff_ys = 5.0 MPa
5%:        ys=10 MPa, D=100% → eff_ys = 0.0 MPa ← COMPLETE FAILURE
```

**New (H=0.3, strain_crit=0.2):**
```
0% strain:  ys=10.0 MPa, D=0%   → eff_ys = 10.0 MPa
5%:         ys=10.6 MPa, D=25%  → eff_ys = 8.0 MPa
10%:        ys=11.3 MPa, D=50%  → eff_ys = 5.6 MPa
15%:        ys=11.9 MPa, D=75%  → eff_ys = 3.0 MPa
20%:        ys=12.5 MPa, D=100% → eff_ys = 0.0 MPa
```

Material stays strong 4× longer!

---

## Does It Make Sense to Have Both On?

**YES!** Here's why:

### Real Rock Behavior:
1. **Early loading (0-3% strain):**
   - Pores close, grains compact
   - Hardening dominates (H > S)
   - Stress increases

2. **Peak strength (3-5% strain):**
   - Balanced (H ≈ S)
   - Plateau region
   
3. **Post-peak (5-10% strain):**
   - Microcracks coalesce
   - Softening dominates (handled by DAMAGE in your model)
   - Stress decreases

### In Your Model:
Since you already have DAMAGE for post-peak softening:
- Use **H > 0** for early-stage hardening
- Use **S ≈ 0** (let damage handle softening)
- This prevents double-counting softening mechanisms

---

## Implementation Notes

### Where It's Used:
Lines 466-467 in `mpmRoutines.py`:
```python
dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)
```

Also in Von Mises model (lines 344-345):
```python
dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)
```

### Order of Operations:
1. Check yielding: `if sigma_eq > yield_eff`
2. Compute plastic increment: `delta_gamma`
3. **Update damage:** `D += dD` (weakening)
4. **Update yield stress:** `ys += dsy` (hardening or softening)
5. Net effect next step: `yield_eff = (1-D) × ys`

---

## Final Recommendation

**Update your config to:**
```json
"hardening": 0.3,
"softening": 0.0,
"strainCriteria": 0.2
```

This will:
✅ Add competing mechanism to damage
✅ Slow down damage propagation
✅ Create more realistic failure patterns
✅ Prevent immediate runaway weakening

**Why this works:**
- Damage: (1-D) factor weakens material
- Hardening: +2μH·Δγ term strengthens material
- They compete, creating controlled progression
- Higher strain_criteria gives more time for hardening to act

---

## Validation After Change

After running with new parameters, check:

1. **Damage field:** Should see localized bands, not uniform
2. **Yield stress:** Should increase before damage takes over
3. **Plastic strain:** Should be heterogeneous (0-20% range)
4. **Phase transition:** Particles transition progressively, not all at once

**Expected:** Controlled damage growth instead of runaway propagation!

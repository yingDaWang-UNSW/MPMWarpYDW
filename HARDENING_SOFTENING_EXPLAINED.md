# Hardening and Softening Parameters: Units and Meaning

## The Equation (Lines 466-467)

```python
dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)
```

Where:
- `dsy`: Yield stress increment [Pa]
- `mu`: Shear modulus [Pa]
- `hardening`, `softening`: Dimensionless parameters [-]
- `delta_gamma`: Plastic multiplier (dimensionless) [-]

## Dimensional Analysis

```
[dsy] = [mu] · [H - S] · [delta_gamma]
[Pa]  = [Pa] · [-]     · [-]

Therefore: hardening and softening are DIMENSIONLESS
```

## Physical Meaning

### Hardening (H)
- **Type**: Dimensionless parameter
- **Effect**: Material STRENGTHENS with plastic deformation
- **When H > 0**: `dsy > 0` → yield stress increases
- **Physical mechanisms**:
  - **Dislocation hardening**: Dislocations pile up, making further deformation harder
  - **Grain interlocking**: Under compression, grains wedge together
  - **Pore collapse**: Porosity decreases, increasing contact area
  - **Friction increase**: More contact → more resistance
  
### Softening (S)
- **Type**: Dimensionless parameter
- **Effect**: Material WEAKENS with plastic deformation
- **When S > 0**: `dsy < 0` → yield stress decreases
- **Physical mechanisms**:
  - **Microcracking**: Cracks grow and coalesce
  - **Bond breakage**: Material cohesion degrades
  - **Fabric damage**: Internal structure breaks down
  - **Strain localization**: Shear bands form

## Net Effect: (H - S)

The yield stress evolution depends on the **net** effect:

| Condition | Net Effect | Result |
|-----------|-----------|---------|
| H > S | Positive | Material hardens overall |
| H < S | Negative | Material softens overall |
| H = S | Zero | No yield stress evolution |

### Example Evolution:
```
At each plastic increment Δγ:
  dsy = 2μ(H - S)Δγ
  
If H=0.3, S=0.1, μ=40 MPa, Δγ=0.001:
  dsy = 2 × 40 × (0.3 - 0.1) × 0.001
      = 2 × 40 × 0.2 × 0.001
      = 0.016 MPa per increment
      
Over 100 increments (10% plastic strain):
  Total increase = 1.6 MPa
```

## Does It Make Sense to Have Both On?

**YES!** This is very common and physically realistic, especially in geomechanics.

### Real Materials Have COMPETING Mechanisms:

#### Early Stage Deformation (Low Plastic Strain):
- **Hardening dominates**: Pores close, grains compact
- Rock gets stronger initially
- H > S, net hardening

#### Late Stage Deformation (High Plastic Strain):
- **Softening dominates**: Microcracks connect, bonds break
- Rock weakens toward failure
- This should be handled by DAMAGE, not S parameter

### Rock Mechanics Example:

```
Compression Test on Sandstone:

0-2% strain:   H dominates → stress increases (work hardening)
2-5% strain:   H ≈ S       → peak stress plateau
5-10% strain:  S dominates → stress decreases (strain softening)
>10% strain:   Damage = 1  → complete failure
```

## Your Model Has TWO Softening Mechanisms!

This is important to understand:

### 1. Damage (Multiplicative Softening)
```python
yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)
```
- Damage `D` goes from 0 → 1
- Multiplies the effective yield stress by `(1-D)`
- At D=1, yield_eff = 0 (complete failure)

### 2. Softening Parameter (Additive Softening)
```python
dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
```
- Softening `S` is a rate parameter
- Reduces yield stress incrementally with plastic strain
- Accumulates over time

### Combined Effect:

For rock with both mechanisms:
```
Total yield evolution = Damage effect × (Base + Hardening/Softening)

yield_eff = (1 - D) · [ys_base + ∫(2μ(H-S)dγ) - α·p]
            \_____/   \___________________/
            Damage      H/S evolution
          (weakening)   (competing)
```

## Recommended Values for Rock Caving

Given that you already have damage mechanics:

### Option 1: Hardening to Counteract Damage (Recommended)
```json
"hardening": 0.3,    // Moderate hardening from compaction
"softening": 0.0     // Let damage handle all softening
```
**Why**: Simplifies model—damage is the only softening mechanism

### Option 2: Net Hardening with Small Softening
```json
"hardening": 0.3,    // Hardening from grain interlocking
"softening": 0.1     // Small softening from fabric damage
```
**Net effect**: H - S = 0.2 (net hardening)
**Why**: More realistic for multi-stage behavior

### Option 3: Pure Softening (Not Recommended)
```json
"hardening": 0.0,
"softening": 0.2
```
**Why not**: You already have damage for softening—this creates runaway weakening

## Physical Interpretation for Your Parameters

From your config: `E = 1e8 Pa`, `nu = 0.2`
- Shear modulus: `mu = E/(2(1+nu)) = 4.17e7 Pa ≈ 42 MPa`

### Effect of Different H-S Values:

| H | S | Net | Effect per 1% Plastic Strain |
|---|---|-----|------------------------------|
| 0.0 | 0.0 | 0.0 | 0 MPa (no evolution) |
| 0.3 | 0.0 | 0.3 | +2.5 MPa (hardening) |
| 0.0 | 0.1 | -0.1 | -0.8 MPa (softening) |
| 0.3 | 0.1 | 0.2 | +1.7 MPa (net hardening) |

Calculation: `Δσ_y = 2 × 42 MPa × (H-S) × 0.01`

## Interaction with Damage

Critical to understand the **order of operations**:

```python
# Step 1: Check if yielding occurs
yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)
if sigma_eq > yield_eff:  # Yielding!
    
    # Step 2: Compute plastic increment
    delta_gamma = ...
    
    # Step 3: Update damage
    dD = delta_gamma / strainCriteria[p]
    D = min(1.0, D + dD)  # Damage increases
    
    # Step 4: Update yield stress (hardening/softening)
    dsy = 2.0 * mu * (H - S) * delta_gamma
    yield_stress[p] = max(0.0, yield_stress[p] + dsy)
```

### The Feedback Loop:

1. Yielding occurs → `delta_gamma > 0`
2. Damage increases: `D ↑` → `yield_eff ↓` (weakening)
3. Yield stress updates: `ys ↑` if H>S, `ys ↓` if H<S
4. Net effect in next step: `yield_eff = (1-D)·ys`

### Example (H=0.3, S=0, strainCriteria=0.05):

```
Step    Plastic Strain    Damage    Base ys    Effective ys
-----   --------------    ------    --------   -------------
0       0%                0%        10 MPa     10.0 MPa
1       1%                20%       10.3 MPa   8.2 MPa  ← Damage wins!
2       2%                40%       10.6 MPa   6.4 MPa
3       3%                60%       10.9 MPa   4.4 MPa
4       4%                80%       11.2 MPa   2.2 MPa
5       5%                100%      11.5 MPa   0.0 MPa  ← Complete failure
```

**Key insight**: Even with hardening (H=0.3), damage (D→1) dominates and drives material to failure!

## Recommendations for Your Model

### Current Issue:
With `H=0, S=0, strainCriteria=0.05`:
- Damage is the ONLY mechanism changing strength
- No competing mechanism to slow down failure
- Result: Runaway damage propagation

### Fix Option 1 (Preferred):
```json
"hardening": 0.3,
"softening": 0.0,
"strainCriteria": 0.15
```
- Hardening counteracts some damage weakening
- Higher strain criteria = more ductile
- Controlled damage propagation

### Fix Option 2 (More Realistic):
```json
"hardening": 0.4,
"softening": 0.1,
"strainCriteria": 0.2
```
- Net hardening of 0.3
- Small softening for realism
- Even more ductile response

### Fix Option 3 (Very Ductile):
```json
"hardening": 0.5,
"softening": 0.0,
"strainCriteria": 0.3
```
- Strong hardening to really fight damage
- High strain criteria
- Good for very confined rock masses

## Summary

**Units:**
- `hardening`: Dimensionless [-]
- `softening`: Dimensionless [-]

**Meaning:**
- Control the rate of yield stress evolution with plastic strain
- Net effect: `dσ_y/dγ = 2μ(H-S)`

**Can both be on?**
- Yes! Very common in geomechanics
- Represents competing physical mechanisms
- BUT: You already have damage for softening
- Recommendation: Use H>0, S≈0 to counteract damage

**Key Insight:**
Your model has TWO softening mechanisms (damage + softening parameter). For rock caving, use hardening to provide resistance against damage-driven failure, creating more controlled and realistic fracture propagation.

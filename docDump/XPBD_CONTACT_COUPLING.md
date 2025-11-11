# XPBD Contact-Aware Coupling

## Problem

In the MPM-XPBD coupling, XPBD particles (materialLabel=2, phase-changed debris) contribute their momentum to the velocity grid during P2G. This is correct when particles are colliding/compressing against the MPM continuum. However, when XPBD particles move away from MPM, they can create an artificial "pull" effect during G2P, because:

1. **P2G**: XPBD particles transfer their momentum to nearby grid nodes
2. **G2P**: MPM particles interpolate velocities from those grid nodes
3. **Problem**: If XPBD is moving away, MPM gets "dragged" along even though there's no physical contact

This violates the principle that granular debris (XPBD) should only push on the continuum (MPM), not pull.

## Solution: Contact-Aware Coupling

Added a velocity-based contact detection in P2G that prevents XPBD particles from contributing momentum when they're moving away from grid nodes.

### Implementation

**Modified Files:**
1. `utils/mpmRoutines.py` - `p2g_apic_with_stress` kernel
2. `utils/simulationRoutines.py` - `mpmSimulationStep` function
3. `utils/getArgs.py` - Added config parameter
4. `runMPMYDW.py` - Pass parameter to simulation

**Algorithm:**
```python
# For each XPBD particle → grid node pair
direction = (grid_pos - particle_pos).normalized()
v_normal = dot(particle_velocity, direction)

# Only contribute if approaching (negative v_normal)
if v_normal < threshold:
    add_to_grid(momentum)
else:
    # Moving away, don't contribute (prevents pulling)
    skip
```

### Configuration Parameter

**`xpbd_contact_threshold`** (units: m/s)

- **Default: `-1e20`** (disabled, original behavior - always couple)
- **Recommended: `0.0`** (compression only - physical contact)
- **Positive values**: Allow small separation velocity (useful for numerical stability)

#### Example Values:
```json
{
  "xpbd_contact_threshold": -1e20   // Disabled (original)
  "xpbd_contact_threshold": 0.0     // Compression only (recommended)
  "xpbd_contact_threshold": 0.1     // Allow 0.1 m/s separation
  "xpbd_contact_threshold": 1.0     // Soft coupling, more gradual
}
```

### Usage

**In JSON config:**
```json
{
  "xpbd_contact_threshold": 0.0
}
```

**Command line override:**
```bash
python runMPMYDW.py --config config.json --xpbd_contact_threshold 0.0
```

### Physical Interpretation

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| `-1e20` (default) | Always couple | Testing, comparison with old results |
| `0.0` | Compression only | **Recommended**: Physical contact-only |
| `0.1` to `1.0` | Soft contact | Numerical stability, gradual separation |
| `>1.0` | Very soft | Highly permissive, may allow unphysical pulling |

### Expected Effects

**With `threshold = 0.0` (compression only):**

✅ **Improvements:**
- XPBD debris only pushes on MPM, doesn't pull
- More physically realistic granular-continuum interaction
- Prevents artificial "vacuum" effect when debris falls away
- Better energy conservation (no tensile work at interface)

⚠️ **Potential Issues:**
- Slightly more aggressive separation at interface
- May need to tune other damping parameters
- Could see faster debris detachment from MPM

### Testing Recommendations

1. **Start with threshold = 0.0** (compression only)
2. **Monitor:**
   - Debris separation behavior
   - Energy conservation
   - MPM deformation patterns near phase-changed zones
3. **Tune if needed:**
   - If separation too aggressive: increase to 0.1-1.0
   - If still seeing "pulling": check for other coupling issues

### Technical Details

**Location in Code:**
```python
# utils/mpmRoutines.py, line ~730
if is_xpbd and xpbd_contact_threshold > -1e10:
    direction = dpos / (wp.length(dpos) + 1e-12)
    v_normal = wp.dot(particle_v[p], direction)
    
    if v_normal < xpbd_contact_threshold:
        # Approaching: add momentum
        wp.atomic_add(grid_v_in, ix, iy, iz, v_in_add)
        wp.atomic_add(grid_m, ix, iy, iz, weight * particle_mass[p])
    # else: moving away, skip (no pulling)
```

**Sign Convention:**
- `dpos`: vector from particle to grid node
- `v_normal < 0`: particle approaching grid node (compression) → **couple**
- `v_normal > 0`: particle receding from grid node (separation) → **don't couple**

### Validation

To verify the fix is working:

1. **Visual inspection**: XPBD debris should freely fall away without dragging MPM
2. **Energy check**: No tensile work should occur at XPBD-MPM interface
3. **Compare runs**: 
   - `threshold = -1e20` (old behavior)
   - `threshold = 0.0` (new behavior)
   - Debris should separate more cleanly with new behavior

### Future Enhancements

Possible improvements:
1. **Two-way contact forces**: Compute proper contact stress tensor instead of momentum filtering
2. **Distance-based**: Add proximity check (only couple if distance < threshold)
3. **Stress-based**: Weight coupling by stress state (compression vs tension)
4. **Adaptive threshold**: Make threshold depend on local deformation rate

### Related Parameters

This parameter works with:
- `eff` (phase change efficiency): Controls energy release when MPM→XPBD
- `xpbd_relaxation`: XPBD constraint relaxation
- `rpic_damping`: P2G damping (affects grid velocity smoothness)
- `grid_v_damping_scale`: Grid-level damping

Recommended to keep defaults for these unless coupling instabilities observed.

---

**Author:** GitHub Copilot  
**Date:** 2025-11-11  
**Version:** 1.0

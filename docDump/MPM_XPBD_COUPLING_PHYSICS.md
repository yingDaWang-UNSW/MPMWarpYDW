# MPM-XPBD Coupling Physics

## Current Implementation: Mass-Only Coupling

### What XPBD Particles Transfer to Grid (P2G)
- **Mass**: ✓ Yes
- **Momentum**: ✗ No
- **Stress**: ✗ No (XPBD are discrete particles)
- **Gravitational Force**: ✗ No

### Why Not Add XPBD Gravitational Momentum?

**Question**: "Should XPBD particles contribute `mass × gravity × dt` to `grid_v_in`?"

**Answer**: **NO**, for three critical reasons:

#### 1. **Double-Counting of Gravity**
```python
# In P2G (if we added gravity):
grid_v_in += weight * mass_XPBD * gravity * dt  # ← Would apply gravity here

# Later in grid_normalization_and_gravity:
v_out = grid_v_in / grid_m
v_out += dt * gravity  # ← Gravity applied again uniformly!
```
- XPBD particles would experience gravity TWICE
- This would artificially accelerate them downward
- Violates conservation of energy

#### 2. **XPBD Has Its Own Gravity**
- XPBD particles handle their own gravity in `xpbdSimulationStep`
- They integrate their motion independently
- Adding gravity in P2G would interfere with their dynamics

#### 3. **Falling XPBD Would "Pull" on MPM**
When roof failure occurs:
- XPBD particles (fragmented rock) fall downward
- If we add their gravitational momentum to grid:
  - Grid acceleration increases downward
  - MPM below would be "dragged" along with falling XPBD
  - This is physically incorrect (no contact force during free fall)

### What About Static Load Transfer?

**Problem**: An XPBD pile sitting on top of MPM block should create compressive stress, but currently doesn't.

**Why**: Mass-only coupling provides only **kinematic resistance** (inertial), not **static force transfer**.

**Solution**: Need proper **contact mechanics**:
1. Detect when XPBD is in **compressive contact** with MPM (not separating)
2. Transfer **contact force** only in compression
3. Zero force transfer during free fall or tension

This is what the `xpbd_contact_threshold` parameter was intended for (currently unused).

## Physics Summary

### What Works Currently
✅ **Kinematic Coupling**: XPBD mass dilutes grid velocity
- MPM particles feel resistance when trying to accelerate through XPBD mass
- Provides "virtual inertia" from XPBD presence

✅ **No Pulling**: XPBD particles don't drag MPM when separating
- Velocity coupling disabled for XPBD
- Only mass contributes to grid

✅ **Gravity Handled Separately**: 
- MPM: Gravity in `grid_normalization_and_gravity` + geostatic initialization
- XPBD: Gravity in `xpbdSimulationStep`

### What's Missing
❌ **Static Load Transfer**: XPBD pile weight doesn't compress MPM below
- Need contact detection to identify compression state
- Only transfer force when `relative_velocity · contact_normal < 0`

❌ **Contact Forces**: No normal/tangential force transfer at MPM-XPBD interface
- Current: Pure kinematic coupling
- Needed: Penalty-based or constraint-based contact

## Implementation Options

### Option 1: Velocity-Based Contact Detection (Current Framework)
Use `xpbd_contact_threshold` to detect compression:
```python
if is_xpbd:
    # Compute relative velocity at contact
    v_rel = particle_v[p] - grid_v_in[ix,iy,iz] / grid_m[ix,iy,iz]
    v_normal = dot(v_rel, contact_normal)  # Assuming vertical: contact_normal = (0,0,-1)
    
    if v_normal < xpbd_contact_threshold:
        # Compression: transfer momentum
        wp.atomic_add(grid_v_in, ix, iy, iz, v_in_add)
    # Always add mass
    wp.atomic_add(grid_m, ix, iy, iz, weight * mass)
```

**Pros**: Simple, uses existing parameter
**Cons**: Doesn't capture static equilibrium (needs ongoing compression velocity)

### Option 2: Penetration-Based Contact (Better for Static)
Track XPBD-MPM overlap and apply penalty forces:
- Compute signed distance between XPBD particle and MPM material
- If penetration detected: apply repulsion force proportional to overlap
- This naturally handles both static and dynamic contact

**Pros**: Physically correct for static loads
**Cons**: More complex, requires spatial queries

### Option 3: Two-Way Coupling via Grid (Hybrid)
- XPBD contributes mass + velocity when compressing
- Use grid stress state to determine if compression
- Asymmetric: Full coupling in compression, mass-only in tension

**Pros**: Balanced approach
**Cons**: Needs careful tuning to avoid instabilities

## Recommendations for Rock Caving

For block caving simulation, the key scenarios are:
1. **Static pile**: XPBD rubble sitting on MPM rock → Need load transfer
2. **Roof collapse**: MPM fails → XPBD particles fall → No load transfer during fall
3. **Impact**: Falling XPBD hits MPM below → Brief compression spike

**Best approach**: **Option 2** (Penetration-based contact)
- Captures static loading correctly
- Handles dynamic impact naturally
- Prevents pulling during free fall

## Current Status
- **Implementation**: Mass-only coupling (no force transfer)
- **Parameter**: `xpbd_contact_threshold` defined but unused
- **Next Step**: Implement proper contact detection and force transfer

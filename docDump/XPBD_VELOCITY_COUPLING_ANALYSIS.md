# XPBD Velocity Coupling Analysis

## Current Implementation

You've changed from **mass-only coupling** to **velocity coupling** where:

1. **MPM particles** (materialLabel=1): Updated every timestep via MPM solver
2. **XPBD particles** (materialLabel=2): Updated periodically via XPBD solver
3. **Coupling**: Both particle types now contribute velocity/momentum to the grid in P2G

### Flow:
```
MPM Step (every dt):
├── P2G: All particles → grid (including XPBD with current velocity)
├── Grid normalization + gravity
├── Grid boundary conditions
└── G2P: Grid → MPM particles only (XPBD velocities unchanged here)

XPBD Step (every dtxpbd, typically ~10 MPM steps):
├── Integrate with gravity
├── Solve contacts (particle-particle, particle-boundary)
└── Commit new positions & velocities (particle_v.assign(particle_v_integrated))
```

## The Key Change

**Before (mass-only):**
```python
if is_xpbd:
    wp.atomic_add(grid_m, ix, iy, iz, weight * particle_mass[p])
    # No momentum contribution
else:
    wp.atomic_add(grid_v_in, ix, iy, iz, v_in_add)
    wp.atomic_add(grid_m, ix, iy, iz, weight * particle_mass[p])
```

**After (velocity coupling):**
```python
# Both MPM and XPBD contribute momentum
wp.atomic_add(grid_v_in, ix, iy, iz, v_in_add)
wp.atomic_add(grid_m, ix, iy, iz, weight * particle_mass[p])
```

## Problems with This Approach

### 1. **Temporal Coupling Mismatch** ⚠️
- **MPM updates**: Every `dt` (e.g., 0.001s)
- **XPBD updates**: Every `dtxpbd` (e.g., 0.01s, 10× slower)

Between XPBD updates:
- XPBD velocity is **stale** (not updated for 9 MPM steps)
- XPBD position is **stale** 
- But XPBD still contributes this outdated velocity to the grid

**Effect**: 
- XPBD particles act like they're moving with old velocity
- If XPBD collided with wall/particle, velocity should change, but MPM doesn't know for 10 steps
- Creates artificial momentum transfer

### 2. **Double Gravity Application** ❌❌
This is **critical**:

```python
# In P2G (for XPBD):
v_in_add = weight * mass * (particle_v[p] + C*dpos) + dt * elastic_force
# particle_v[p] already includes gravity from XPBD integration

# Later in grid_normalization_and_gravity:
v_out = grid_v_in / grid_m
v_out = v_out + dt * gravity  # ← Gravity added AGAIN!
```

**XPBD particles experience gravity twice**:
1. In `integrateParticlesXPBD`: `v += gravity * dt`
2. In grid normalization: `v += gravity * dt`

**Result**: XPBD particles accelerate downward at **2g** instead of **g**!

### 3. **Pulling Problem Returns** ⚠️
Remember the original issue? When XPBD moves away from MPM, it pulls the grid along.

**Now you have this again**:
- XPBD particle moving upward → contributes upward momentum to grid
- MPM below feels this upward velocity
- Creates artificial "suction" when XPBD separates

### 4. **No Position Update for XPBD from MPM** 🤔
You said: "giving MPM particles their XPBD resultant velocity (but not position)"

**But actually**: XPBD particles **don't get MPM velocity back**!

Looking at G2P:
```python
if materialLabel[p] == 1:  # only mpm particles need grid information
    # Update velocity, position, F from grid
```

So the coupling is **one-way only**:
- XPBD → Grid (via P2G)
- Grid → MPM only (via G2P)
- XPBD never reads from grid

This means:
- ✓ XPBD position is independent (good for explicit XPBD solver)
- ✗ XPBD doesn't feel MPM's influence except through occasional grid momentum mixing
- ✗ Asymmetric: XPBD affects MPM, but MPM barely affects XPBD

## Recommended Fix

### Option A: Pure Segregated Approach (Safest)
**XPBD and MPM completely independent, contact forces only**:

```python
# In P2G:
if is_xpbd:
    # XPBD contributes NOTHING to grid during MPM step
    pass  
else:
    wp.atomic_add(grid_v_in, ix, iy, iz, v_in_add)
    wp.atomic_add(grid_m, ix, iy, iz, weight * particle_mass[p])

# Add separate contact force kernel after XPBD step
# that detects MPM-XPBD overlap and applies penalty forces
```

**Pros**: 
- No double gravity
- No pulling
- Clean separation
- Each solver handles its own particles

**Cons**: 
- Need to implement explicit contact detection
- More complex

### Option B: Correct the Double Gravity (Quick Fix)
Keep your current approach but fix the double gravity:

```python
# In grid_normalization_and_gravity:
grid_x, grid_y, grid_z = wp.tid()
if grid_m[grid_x, grid_y, grid_z] > 1e-15:
    v_out = grid_v_in[grid_x, grid_y, grid_z] / grid_m[grid_x, grid_y, grid_z]
    # Don't add gravity here - it's already in XPBD velocities
    # and we want consistent treatment
    grid_v_out[grid_x, grid_y, grid_z] = v_out
```

Then add gravity **only in MPM constitutive update** or **only in XPBD integration**, not both places.

**Pros**: Simple fix
**Cons**: Still have pulling problem, temporal mismatch, one-way coupling

### Option C: Proper Two-Way Subcycling (Complex)
Update XPBD every MPM step with small substeps:

```python
# Every MPM step:
for substep in range(mpmStepsPerXpbdStep):
    mpmSimulationStep(...)
    xpbdSubstep(dt)  # Small XPBD update matching MPM frequency
```

**Pros**: Proper coupling, no temporal lag
**Cons**: Expensive, defeats purpose of larger XPBD timestep

## What You Should Check

1. **Run a free-fall test**: Drop XPBD particles in empty space
   - Measure acceleration
   - Should be `g = 9.81 m/s²`
   - If you get `~19.6 m/s²`, you have double gravity

2. **Roof collapse test**: 
   - Does roof collapse create downward "suction" on lower MPM?
   - Check if falling XPBD drags MPM unnaturally

3. **Static pile test**:
   - XPBD pile on MPM block
   - Does it create compression stress? (Spoiler: probably not much with current approach)

## My Recommendation

**Go back to mass-only coupling** or **implement Option A (segregated with contacts)**.

The velocity coupling you have now:
- ❌ Applies gravity twice to XPBD
- ❌ Reintroduces pulling problem
- ❌ Has temporal lag issues
- ❌ Asymmetric (XPBD→MPM but not MPM→XPBD)

If you want XPBD to transfer forces to MPM, you need **proper contact mechanics** that:
- Only activates during compression
- Uses penalty or constraint forces
- Respects Newton's 3rd law (both directions)
- Handles static and dynamic cases

The quick visualization you have working might look okay because:
- Gravity doubling might be hidden by other numerical effects
- Temporal lag of 10 steps might be small compared to total simulation time
- Pulling might not be visible in your current geometry

But fundamentally, the coupling is incorrect and will cause problems in edge cases.

# MPMWarpYDW - AI Coding Agent Instructions

## terminal conda exposure: on windows, use the conda exe located in the usual anaconda install path and inside the fs5 env
## Project Overview
GPU-accelerated coupled MPM-XPBD simulation framework for modeling material failure, fragmentation, and granular flow in geomaterials. Continuum mechanics (MPM) seamlessly transitions to discrete particle dynamics (XPBD) when damage criteria are met.

## Critical Architecture Patterns

### 1. NVIDIA Warp Kernel-Based Execution
**Everything compute-intensive runs on GPU via Warp kernels:**
```python
@wp.kernel  # Thread-parallel, one thread per particle/grid cell
def compute_stress_from_F_trial(particle_F: wp.array(dtype=wp.mat33), ...):
    p = wp.tid()  # Thread ID = particle ID
    # All operations must be Warp-compatible types
```

**Key constraints:**
- Use `wp.vec3`, `wp.mat33`, `wp.array` types, NOT numpy arrays in kernels
- Call kernels via `wp.launch(kernel=func, dim=nPoints, inputs=[...], device=device)`
- Scatter operations (P2G) require `wp.atomic_add()` for thread safety
- No Python control flow (if/for) in kernels - use Warp equivalents

### 2. Three-Phase Material Model
**Material labels determine physics:**
- `materialLabel=0`: MPM continuum (in contact with XPBD, **cannot transition**)
- `materialLabel=1`: MPM continuum (not in contact, **can transition** to XPBD if damage >= 1)
- `materialLabel=2`: XPBD discrete particles (post-failure, contact-based)

**MPM-XPBD contact locking:**
- Before each XPBD step, all MPM particles (label 0) are reset to label 1
- During contact detection, if an MPM particle contacts an XPBD particle, it's set to label 0
- This prevents MPM particles from transitioning while supporting XPBD debris

**One-way transition when `damage >= 1.0` AND `materialLabel == 1`:**
```python
# In mpmRoutines.py return mapping
if damage[p] >= 1.0 and materialLabel[p] == 1:  # Only label 1 can transition
    materialLabel[p] = 2  # MPM → XPBD
    # Energy release: elastic strain energy → kinetic energy
    v_release = wp.sqrt(2.0 * efficiency * strain_energy / rho)
```

**Critical bug prevention:** If `strainCriteria` is too large AND `softening > hardening`, particles can reach `ys <= 0` while still MPM, creating "ghost particles" with zero stiffness. Always validate transition thresholds.

### 3. Hybrid MPM-XPBD Coupling Strategy
**NOT a simple momentum exchange - uses asymmetric coupling:**

**XPBD → MPM (via P2G):**
- XPBD particles transfer **mass only** to grid, NO momentum
- Prevents spurious tensile forces when XPBD particles move away
```python
# In xpbdRoutines.py
if materialLabel[p] == 2:  # XPBD particle
    wp.atomic_add(grid_m, grid_idx, m_p)  # Mass only, no momentum
```

**MPM ↔ XPBD (via direct contact):**
- Particle-particle contact forces handle compression/interaction
- Physically realistic, no momentum smearing artifacts

### 4. Logarithmic Strain Multiplicative Plasticity
**NOT small-strain additive plasticity:**
```python
# Decompose F via SVD: F = U·Σ·V^T
epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))  # Logarithmic strain
# Return mapping operates in log-strain space
# Reconstruct: F_elastic = U · exp(ε_corrected) · V^T
```

**Why:** Handles large rotations and extreme deformations correctly. Small-strain assumptions (ε = ∇u) fail at >5-10% strain.

## File Organization & Responsibilities

```
runMPMYDW.py              # Main driver: loads HDF5, initializes states, runs simulation loop
utils/
  ├── getArgs.py          # JSON/CLI config parser (--config loads JSON first)
  ├── simulationRoutines.py  # High-level orchestration (mpmSimulationStep, xpbdSimulationStep)
  ├── mpmRoutines.py      # MPM kernels: stress, P2G, G2P, return mapping, damage
  ├── xpbdRoutines.py     # XPBD kernels: contacts, integration, hash grid
  ├── simStates.py        # State containers (SimState, MPMState, XPBDState)
  └── fs5RendererCore.py  # OpenGL visualization
benchmarks/
  ├── generalBenchmark/   # Elastic validation, analytical comparison
  ├── spatialBenchmark/   # Heterogeneous properties, coupling tests
  └── cavingBenchmark/    # Large-scale rock fragmentation
```

## Development Workflows

### Running Simulations
```bash
# Standard run with JSON config (preferred)
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_elastic_validation.json

# Override specific parameters
python runMPMYDW.py --config config.json --dt 1e-4 --E 1e9 --render 1

# Quick test (small domain, few steps)
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json
```

### Creating Input Domains
**HDF5 structure (see `exampleDomains/createInputHDF5.py`):**
```python
with h5py.File("domain.h5", "w") as f:
    f.create_dataset("x", data=positions.T)  # Shape: (3, n_particles), NOT (n_particles, 3)
    f.create_dataset("particle_volume", data=volumes)  # Shape: (n_particles,)
    
    # Optional spatial properties (override JSON defaults):
    f.create_dataset("E", data=E_array)  # Young's modulus per particle
    f.create_dataset("ys", data=ys_array)  # Yield stress per particle
    # Also: density, nu, alpha, hardening, softening, eta_shear, eta_bulk, strainCriteria
```

**If property NOT in HDF5:** Falls back to JSON/CLI uniform value.

### Validation & Debugging
```bash
# Analytical comparison (elastic regime only)
cd benchmarks/generalBenchmark
python analytical_validation.py --E 1e9 --nu 0.2 --density 5000

# Verify HDF5 file structure
cd benchmarks/cavingBenchmark
python verify_hdf5_fields.py random_rock_domain.h5

# Visualize spatial properties
python visualizeRandomRockDomain.py
```

### CFL Condition Analysis
**New feature (Lines 198-270 in runMPMYDW.py):**
- Automatically estimates CFL number on startup
- Calculates wave speeds: `c_p = sqrt((K + 4G/3) / ρ)`
- Provides safety factor recommendations (MPM stable: CFL < 0.3)
- Shows time scales (cell crossing time, domain crossing time)

## Critical Implementation Details

### Geostatic Initialization (K0 Condition)
**Must initialize deformation gradient F, NOT stress:**
```python
# WRONG: Initialize stress (gets overwritten by compute_stress_from_F_trial)
particle_stress[p] = geostatic_stress_tensor

# CORRECT: Initialize F to encode prestress (see initialize_geostatic_F kernel)
sigma_v = -rho * g * depth
sigma_h = K0 * sigma_v
# Solve for strains, then F = exp(ε)
```

### Drucker-Prager vs Von Mises
**Select via `constitutive_model` parameter:**
- `0`: Von Mises (pressure-independent, use for metals/clays)
- `1`: Drucker-Prager (pressure-dependent, use for rocks/soils/granular)

**Drucker-Prager specifics:**
```python
# Yield: σ_eq ≤ (1-D)·(σ_y - α·p)
# α = pressure sensitivity (typ. 0.2-0.5 for rocks)
# Non-associated flow: β = 0.3·α (dilatancy parameter)
# Volume correction: ε_vol += β·Δγ (expansion during shear)
```

### Time-Based Rendering
**Use `--render_interval` (simulation time), NOT step count:**
```python
# In runMPMYDW.py main loop
shouldRenderSave = sim.t >= (nextRenderTime - 0.5 * sim.dt)  # Tolerance for float precision
if shouldRenderSave:
    nextRenderTime += args.render_interval
```

**Why tolerance:** Floating-point accumulation causes `sim.t` to drift below exact multiples.

## Common Pitfalls & Solutions

### 1. Stress Sign Conventions
- **Compression = NEGATIVE** in Cauchy stress
- **Compression = POSITIVE** in some geomechanics contexts
- Check sign carefully when comparing to analytical solutions

### 2. Boundary Friction
- `boundFriction=0`: Frictionless walls (K0 conditions difficult)
- `boundFriction=0.3-0.5`: Typical soil friction
- Low friction → block slides → K0 deviation increases

### 3. Plasticity Parameter Balance
```python
# Safe defaults for rocks:
ys = 10e6  # 10 MPa cohesion
alpha = 0.3  # ~30° friction angle
hardening = 0.0  # No strain hardening
softening = 0.0  # No softening initially
strainCriteria = 0.01  # 1% plastic strain to full damage
```

**Dangerous combo:** `hardening=0, softening>0, strainCriteria>>1` → ghost particles

### 4. Grid Resolution
- `grid_particle_spacing_scale = 2.0`: 2× particle diameter (coarse, fast)
- `grid_particle_spacing_scale = 4.0`: 4× diameter (fine, accurate, DEFAULT)
- Too coarse → stress oscillations, inaccurate contacts

## Testing Strategy

### Elastic Validation (Linear Regime)
```bash
# Test case in benchmarks/generalBenchmark/config_elastic_validation.json
# E=1e9 Pa, ρ=5000 kg/m³, H~20m → ε_max ≈ 0.09% (well within linear limit)
# Expected: < 5% error vs analytical solution
```

### Large Deformation Tests
- E=1e6 Pa → strains ≈ 50-100% → nonlinear regime
- Analytical solutions INVALID beyond ε > 5-10%
- Use as stress-test for numerical stability, NOT validation

### Coupling Tests (see `benchmarks/spatialBenchmark/config_coupling_test.json`)
- Rigid slab (ys=1e9 Pa) + weak block (ys=1e4 Pa)
- Block yields immediately → transitions to XPBD
- Validates phase transition mechanics

## Performance Optimization

### GPU Utilization
- Warp kernels scale with particle count (millions feasible)
- Grid operations scale with `gridDims.x * gridDims.y * gridDims.z`
- Contact detection: O(n) with spatial hashing (avoid naive O(n²))

### Timestep Selection
```python
# CFL limit: dt < dx / c_p
# For E=1e9 Pa, ρ=5000, dx=2m:
#   c_p ≈ 700 m/s → dt_max ≈ 0.003s
# Recommended: dt = 0.3 * dt_max for stability
```

### Convergence Tuning
```python
residualThreshold = 1e-8  # Very strict (slow)
residualThreshold = 1e-1  # Loose (fast, may miss equilibrium)
# Residual = Σ(||v||/r) / N_active (normalized by particle radius)
```

## When to Ask for Clarification

1. **Material parameter selection**: Physics domain-specific (ask user for material type)
2. **Boundary condition implementation**: Many variants (K0, roller, fixed, etc.)
3. **Output format preferences**: VTK, USD, custom formats
4. **Performance vs accuracy tradeoffs**: User-specific requirements

## Key References in Codebase

- **Stress computation:** `utils/mpmRoutines.py` lines 150-250 (compute_stress_from_F_trial)
- **Return mapping:** `utils/mpmRoutines.py` lines 250-500 (drucker_prager_return_mapping, von_mises_return_mapping)
- **Phase transition logic:** `utils/mpmRoutines.py` lines 420-450 (damage >= 1.0 check)
- **CFL analysis:** `runMPMYDW.py` lines 198-270
- **Geostatic init:** `utils/mpmRoutines.py` lines 62-145 (initialize_geostatic_F)

```markdown
// filepath: d:\sourceCodes\MPMWarpYDW\README.md
# MPMWarpYDW

A GPU-accelerated 3D coupled simulation framework combining the Material Point Method (MPM) with Extended Position Based Dynamics (XPBD) for modeling large deformation, damage, phase transitions, and granular flow in geomaterials.

## Overview

This framework simulates the progressive failure and fragmentation of continuum materials (e.g., rock, soil) by:
1. **MPM Phase**: Modeling continuum deformation with elasto-plastic constitutive laws, damage accumulation, and stress-based failure
2. **Phase Transition**: Automatically converting failed material points into discrete particles when damage reaches critical values
3. **XPBD Phase**: Simulating post-failure granular dynamics with particle-particle contacts, friction, and cohesion

The coupling enables seamless simulation of processes like rock avalanches, landslides, cave propagation, and fragmentation under gravity and dynamic loading.

## Key Features

### Material Point Method (MPM)
- **Multiplicative elastoplasticity** with logarithmic strain measures (valid for large deformations)
- **Dual constitutive models**: 
  - **Von Mises** (pressure-independent, J2 plasticity)
  - **Drucker-Prager** (pressure-dependent, frictional materials with non-associated dilatancy)
- **Return mapping** in principal strain space with volumetric-deviatoric split
- **Damage mechanics** based on accumulated plastic strain with configurable softening
- **Phase transition** from continuum to discrete particles at critical damage (D ≥ 1.0)
- **Kelvin-Voigt viscoelasticity** for rate-dependent behavior and numerical stabilization
- **APIC transfer scheme** (Affine Particle-In-Cell) for angular momentum conservation
- **Optional RPIC** (Rotated PIC) for controllable numerical damping
- **Grid-based boundary conditions** with Coulomb friction
- **Geostatic initialization** with K₀ coefficient for pre-stressed domains
- **Spatially-varying properties**: Load heterogeneous material fields from HDF5

### Extended Position Based Dynamics (XPBD)
- **Particle-particle contact resolution** using spatial hash grid for O(n) neighbor search
- **Particle-boundary contacts** with friction
- **Static/dynamic friction** transition based on velocity threshold
- **Particle cohesion** for modeling cementation between fragments
- **Particle sleeping** mechanism for computational efficiency
- **Particle swelling** to model volume expansion (bulking) during fragmentation
- **Velocity clamping** to prevent excessive velocities during phase change

### MPM-XPBD Coupling
- **Contact-based momentum transfer**: Direct particle-particle contact forces handle all MPM-XPBD interaction
- **Contact-based transition locking (optional)**: MPM particles in contact with XPBD debris are prevented from transitioning (maintains structural support)
- **Energy-based velocity initialization**: Elastic strain energy converts to kinetic energy during phase transition

### Simulation Control
- **Two-phase big step structure**:
  - Phase 1: Combined MPM+XPBD (coupled simulation)
  - Phase 2: XPBD-only (optional settling phase with frozen MPM)
- **Damage-based early termination**: Stop combined phase when damage stops accumulating
- **Sleep-based termination**: Stop XPBD phase when sufficient particles are sleeping
- **Gravity pulse loading**: Apply temporary increased gravity at start of each big step
- **Restart from checkpoints**: Resume simulation from VTP files saved at big step boundaries

### Numerical Features
- **GPU acceleration** using NVIDIA Warp
- **Explicit time integration** with subcycling (fast MPM steps, slower XPBD steps)
- **Convergence-based adaptive stepping**: Inner loop continues until velocity residual converges
- **Automatic CFL analysis**: Estimates stability limits and wave speeds on startup
- **Optional creep model**: Damage-dependent long-term strength reduction

## Requirements

- Python 3.8+
- [NVIDIA Warp](https://github.com/NVIDIA/warp) (GPU-accelerated physics simulation)
- numpy, h5py, matplotlib, scipy, pyglet, pyvista, lxml, vtk

Install dependencies with:
```sh
pip install warp-lang numpy h5py matplotlib scipy pyglet pyvista lxml vtk
```

**Hardware**: CUDA-capable NVIDIA GPU (recommended: RTX 3000+ series, 8GB+ VRAM)

## Quick Start

### Running a Benchmark

```bash
# Activate conda environment (if using)
conda activate fs5

# Run the general benchmark (quick test)
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json

# Run with real-time visualization
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json --render 1

# Run and auto-open ParaView after completion
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json --open_paraview 1
```

### Creating Custom Simulations

1. **Prepare particle domain**: Create HDF5 file with required fields
   ```python
   import h5py
   import numpy as np
   
   # Required fields:
   # - 'x': particle positions, shape (3, n_particles) - NOTE: transposed!
   # - 'particle_volume': volume per particle, shape (n_particles,)
   
   with h5py.File("domain.h5", "w") as f:
       f.create_dataset("x", data=positions.T)  # Shape: (3, N), NOT (N, 3)
       f.create_dataset("particle_volume", data=volumes)
       
       # Optional: spatially-varying properties (per-particle arrays)
       f.create_dataset("E", data=E_array)
       f.create_dataset("ys", data=ys_array)
       f.create_dataset("density", data=density_array)
       # etc.
   ```

2. **Configure simulation**: Create a JSON config file (see examples in `benchmarks/`)

3. **Run**: `python runMPMYDW.py --config your_config.json`

### Restart from Checkpoint

```bash
# Resume simulation from a checkpoint VTP file (must be from big step boundary)
python runMPMYDW.py --config ./config.json --restart ./output/sim_step_0001_000500_t0000_5000_particles.vtp
```

Checkpoint files contain full simulation state including:
- Particle positions, velocities, deformation gradients
- Material properties (μ, λ, σ_y, etc.)
- Damage and accumulated plastic strain
- XPBD state (initial positions, cumulative distances)

## Mathematical Model

### Constitutive Framework: Multiplicative Elastoplasticity

The framework uses **multiplicative decomposition** of the deformation gradient in **logarithmic (Hencky) strain space**. This is essential for large deformation problems where small-strain assumptions (ε = ∇u) break down beyond ~5-10% strain.

#### Kinematic Setup

1. **Deformation gradient update** (velocity gradient integration):
   ```
   F_trial = (I + ∇v·Δt) · F_n
   ```

2. **Polar decomposition via SVD**:
   ```
   F = U · Σ · V^T
   ```
   where Σ = diag(σ₁, σ₂, σ₃) are the principal stretches.

3. **Logarithmic (Hencky) strain**:
   ```
   ε = log(Σ) = [log(σ₁), log(σ₂), log(σ₃)]
   ```
   
   **Why logarithmic strain?** It provides:
   - Additive decomposition: ε_total = ε_elastic + ε_plastic
   - Symmetric behavior in tension/compression
   - Correct behavior under large rotations (rotation encoded in U, V)

4. **Volumetric-deviatoric split**:
   ```
   ε_vol = tr(ε)/3 = (ε₁ + ε₂ + ε₃)/3
   ε_dev = ε - ε_vol · I
   ```

#### Stress Computation

**Kirchhoff stress** (work-conjugate to logarithmic strain):
```
τ = 2μ·ε + λ·tr(ε)·I
```

where μ, λ are Lamé parameters:
```
μ = E / (2(1 + ν))
λ = E·ν / ((1 + ν)(1 - 2ν))
```

**Cauchy stress** (for output/visualization):
```
σ = τ / det(F)
```

### Plasticity Models

#### Von Mises (constitutive_model=0)

**Yield criterion** (J₂ plasticity):
```
f = ||τ_dev|| - √(2/3)·(1 - D)·σ_y ≤ 0
```

**Associated flow rule** (no dilatancy):
```
Δε_plastic = Δγ · (τ_dev / ||τ_dev||)
```

**Plastic multiplier**:
```
Δγ = ||ε_dev|| - σ_y_eff / (2μ)
```

#### Drucker-Prager (constitutive_model=1)

**Yield criterion** (pressure-dependent):
```
f = ||τ_dev|| - √(2/3)·(1 - D)·(σ_y - α·p) ≤ 0
```

where:
- p = tr(τ)/3 is the mean Kirchhoff stress (compression negative)
- α is the pressure sensitivity (related to friction angle φ by α ≈ 2·sin(φ)/(√3·(3 - sin(φ))))
- Typical values: α = 0.2-0.5 for rocks/soils

**Non-associated flow rule** (with dilatancy):
```
Δε_plastic = Δγ · [n_dev + β·I/3]
```

where:
- n_dev = τ_dev / ||τ_dev|| (deviatoric flow direction)
- β = 0.3·α is the dilatancy parameter
- The β·I/3 term produces volumetric expansion during shear (critical for granular materials)

**Physical interpretation**:
- Compression (p < 0): -α·p > 0 → increases effective yield stress → stronger under confinement
- Tension (p > 0): -α·p < 0 → decreases effective yield stress → weaker in tension
- If α = 0: reduces to Von Mises (pressure-independent)

### Damage Evolution

**Isotropic scalar damage** based on accumulated plastic strain:

```
D = min(1.0, Σ(Δε_plastic) / ε_critical)
```

where ε_critical = `strainCriteria` parameter.

**Damage effects**:
1. **Stiffness degradation**: Effective moduli = (1 - D) × nominal moduli
2. **Strength degradation**: σ_y_eff = (1 - D) × σ_y
3. **Phase transition trigger**: When D ≥ 1.0 AND materialLabel == 1

### Phase Transition

When a particle reaches critical damage (D ≥ 1.0):

1. **Material label change**: 1 (MPM) → 2 (XPBD)

2. **Energy release**: Elastic strain energy converts to kinetic energy
   ```
   u = 0.5 · τ : ε   (strain energy density)
   v_release = √(2 · η · u / ρ)
   ```
   where η = `eff` (efficiency factor, 0-1)

3. **Initial XPBD position recorded** for swelling calculations

4. **Stress zeroing**: Failed material carries no stress (handled by XPBD contacts)

### Viscoelasticity (Kelvin-Voigt)

Rate-dependent stress contribution for numerical stability:

```
τ_visc = 2·η_shear·ε̇_dev + η_bulk·tr(ε̇)·I
```

where ε̇ = 0.5(∇v + ∇v^T) is the strain rate tensor.

**Usage**: 
- Set η_shear, η_bulk > 0 for rate-dependent behavior
- Helps stabilize rapid loading and phase transitions
- Typical values: 10³ - 10⁶ Pa·s depending on strain rate regime

### Geostatic Initialization

For problems with pre-existing stress state (K₀ conditions):

```
σ_v = -ρ·g·(z_top - z)        (vertical stress from overburden)
σ_h = K₀·σ_v                   (horizontal stress)
```

**Critical implementation detail**: The deformation gradient F is initialized to encode the prestress, NOT the stress tensor directly. This ensures consistency with the multiplicative plasticity formulation:

```
F_initial = diag(exp(ε₁), exp(ε₂), exp(ε₃))
```

where strains are computed from the target geostatic stress state via the constitutive law.

### XPBD Contact Resolution

Position-based dynamics with iterative constraint projection:

**Particle-particle contact constraint**:
```
C = ||x_i - x_j|| - (r_i + r_j) ≥ 0
```

**Normal correction** (penetration resolution):
```
Δx_n = C · n · w_i / (w_i + w_j)
```
where n = (x_i - x_j)/||x_i - x_j|| and w = 1/m (inverse mass).

**Friction** (Coulomb model):
```
Δx_f = clamp(Δv_t · Δt, -μ·|Δx_n|, μ·|Δx_n|)
```

**Static/dynamic transition**:
```
μ = μ_static   if ||v|| < v_threshold
μ = μ_dynamic  otherwise
```

**Iteration scheme**: Multiple Gauss-Seidel iterations with SOR relaxation.

### MPM-XPBD Coupling

The coupling uses **direct contact-based momentum transfer**:

**Contact detection**: During XPBD iterations, particle-particle contacts are detected between all active particles regardless of material type (MPM or XPBD).

**Momentum transfer**: When an MPM particle (materialLabel ≤ 1) contacts an XPBD particle (materialLabel == 2):
- Contact forces are computed using the same XPBD constraint projection
- Position corrections are converted to velocity impulses for MPM particles
- The `xpbd_mpm_coupling_strength` parameter scales this impulse

**Transition locking**: 
- Before each XPBD step: Reset all MPM particles from label 0 → label 1
- During contact detection: If MPM contacts XPBD, set label → 0
- In return mapping: Only label == 1 particles can transition to label 2
- **Purpose**: Prevents cascading failure where supporting MPM material fails because debris is pulling on it

```python
# Contact-based transition locking flow:
# 1. Reset: materialLabel[p] = 1 for all MPM particles in contact (was 0)
# 2. Detect: if MPM[i] touches XPBD[j], set materialLabel[i] = 0
# 3. Return mapping: only materialLabel == 1 can transition when damage >= 1.0
```

## Configuration Parameters

### Complete Parameter Reference

#### Time Stepping
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 0 (auto) | MPM timestep. If 0, computed from CFL condition |
| `dtxpbd` | float | 0.01 | XPBD timestep (typically 10-100× larger than dt) |
| `bigStepDuration` | float | 10.0 | Duration of each big step (seconds) |
| `bigSteps` | int | 1 | Number of outer convergence loops |
| `residualThreshold` | float | 1e-8 | Velocity convergence criterion |

#### Material Properties
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `density` | float | 2500 | Material density (kg/m³) |
| `E` | float | 1e9 | Young's modulus (Pa) |
| `nu` | float | 0.3 | Poisson's ratio |
| `constitutive_model` | int | 1 | 0=Von Mises, 1=Drucker-Prager |
| `ys` | float | 1e6 | Yield stress / cohesion (Pa) |
| `alpha` | float | 0.3 | Pressure sensitivity (Drucker-Prager) |
| `hardening` | float | 0.0 | Strain hardening coefficient |
| `softening` | float | 0.0 | Strain softening coefficient |
| `eta_shear` | float | 0.0 | Shear viscosity (Pa·s) |
| `eta_bulk` | float | 0.0 | Bulk viscosity (Pa·s) |

#### Damage & Phase Transition
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strainCriteria` | float | 0.01 | Critical plastic strain for full damage |
| `eff` | float | 0.0 | Energy release efficiency (0-1) |
| `mpm_contact_transition_lock` | int | 1 | Prevent MPM→XPBD transition when in contact with XPBD |

#### Grid & Domain
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `domainFile` | str | - | Path to HDF5 particle domain file |
| `grid_padding` | float | 10 | Padding around particle domain (m) |
| `grid_particle_spacing_scale` | float | 2.0 | Grid spacing = particle_diameter × scale |
| `boundary_padding_mask` | str | "111111" | Which boundaries to pad (xmin,xmax,ymin,ymax,zmin,zmax) |
| `K0` | float | 0.5 | Lateral earth pressure coefficient |
| `z_top` | float | null | Top elevation for geostatic stress (null = auto) |
| `initialise_geostatic` | int | 0 | Enable geostatic stress initialization |

#### XPBD Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xpbd_iterations` | int | 4 | Contact solver iterations per step |
| `xpbd_relaxation` | float | 1.0 | SOR relaxation parameter |
| `xpbd_mpm_coupling_strength` | float | 0.25 | XPBD→MPM velocity coupling factor |
| `dynamicParticleFriction` | float | 0.3 | Dynamic friction coefficient |
| `staticParticleFriction` | float | 0.5 | Static friction coefficient |
| `staticVelocityThreshold` | float | 1e-5 | Static/dynamic friction transition |
| `particle_cohesion` | float | 0.0 | Cohesive attraction distance (m) |
| `sleepThreshold` | float | 0.5 | Sleep velocity threshold (m/s per radius) |

#### Simulation Control
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xpbdOnlyDuration` | float | 0.0 | XPBD-only phase duration after combined phase |
| `damage_stall_threshold` | float | 1e-9 | Mean damage change threshold for early termination |
| `damage_stall_steps` | int | 100 | Steps below threshold to trigger early termination |
| `xpbd_sleep_termination_ratio` | float | 0.95 | XPBD sleep ratio for phase 2 early termination |
| `gravity_pulse_factor` | float | 1.0 | Gravity multiplier during pulse |
| `gravity_pulse_duration` | float | 0.0 | Duration of gravity pulse (seconds) |

#### Damping & Integration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rpic_damping` | float | 0.0 | RPIC damping (0=APIC, 1=full RPIC, -1=PIC) |
| `grid_v_damping_scale` | float | 1.0 | Grid velocity scaling per step |
| `update_cov` | int | 1 | Enable covariance tracking |
| `particle_v_max` | float | 100.0 | Maximum particle velocity (m/s) |
| `volumetric_locking_correction` | int | 0 | Enable F-bar volumetric locking correction |

#### Swelling (Fragmentation Bulking)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `swellingRatio` | float | 0.0 | Particle radius increase ratio |
| `swellingActivationFactor` | float | 0.0 | Distance threshold for activation |
| `swellingMaxFactor` | float | 1.0 | Distance threshold for full swelling |

#### Boundaries & Visualization
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boundaryCondition` | str | "friction" | "friction" or "restitution" |
| `boundFriction` | float | 0.3 | Boundary friction coefficient |
| `boundRestitution` | float | 0.0 | Boundary restitution coefficient |
| `minBoundsXPBD` | vec3 | null | XPBD-specific minimum bounds |
| `maxBoundsXPBD` | vec3 | null | XPBD-specific maximum bounds |
| `xpbd_deactivation_z_datum` | float | null | Z below which XPBD particles deactivate |
| `render` | int | 0 | Enable real-time OpenGL rendering |
| `render_interval` | float | 0.1 | Time between renders/saves (seconds) |
| `color_mode` | str | "damage" | Particle coloring: "damage", "state", "effective_ys", "velocity", "stress" |
| `saveFlag` | int | 0 | Enable VTP/VTI output |
| `outputFolder` | str | "./output/" | Output directory |
| `open_paraview` | int | 0 | Auto-open ParaView after simulation |

#### Restart
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `restart` | str | null | Path to checkpoint VTP file for restart |

## Project Structure

```
MPMWarpYDW/
├── runMPMYDW.py                 # Main simulation driver
├── README.md                    # This file
├── .github/
│   └── copilot-instructions.md # AI coding assistant instructions
│
├── utils/                       # Core simulation modules
│   ├── getArgs.py              # JSON/CLI configuration parser
│   ├── simulationRoutines.py   # High-level MPM/XPBD orchestration
│   ├── mpmRoutines.py          # MPM kernels (stress, P2G, G2P, damage, return mapping)
│   ├── xpbdRoutines.py         # XPBD kernels (contacts, integration, hash grid)
│   ├── simStates.py            # State containers (SimState, MPMState, XPBDState)
│   ├── fs5PlotUtils.py         # VTP/VTI output, restart save/load, ParaView integration
│   └── fs5RendererCore.py      # OpenGL real-time visualization
│
├── exampleDomains/              # Input particle geometry examples
│   ├── createInputHDF5.py      # Script to generate particle domains
│   └── *.h5, *.vtp             # Example geometries
│
├── benchmarks/                  # Validation and test cases
│   ├── generalBenchmark/       # Basic validation tests
│   │   ├── config_*.json       # Various test configurations
│   │   ├── analytical_validation.py  # Compare to analytical solutions
│   │   └── create_block_particles.py # Generate test domains
│   │
│   ├── spatialBenchmark/       # Heterogeneous material tests
│   │   ├── config_coupling_*.json    # MPM-XPBD coupling tests
│   │   ├── createCouplingTestColumn.py  # Generate coupling test domains
│   │   ├── check_xpbd_completely_weightless.py  # Verify coupling physics
│   │   └── analyze_stress_profiles.py  # Post-processing analysis
│   │
│   ├── dynamicBenchmark/       # Dynamic impact tests
│   │   ├── config_dynamic.json # Drop test configuration
│   │   └── create_dynamic_domain.py  # Generate impact test domain
│   │
│   └── cavingBenchmark/        # Large-scale rock caving
│       ├── config_random_rock.json   # Production configuration
│       ├── createRandomRockDomain.py # Generate heterogeneous rock domain
│       └── visualizeRandomRockDomain.py  # Visualize property fields
│
└── output/                      # Simulation results (generated)
    ├── sim_step_*_particles.vtp  # Particle data (ParaView time series)
    └── sim_step_*_grid.vti       # Grid data (optional)
```

## Simulation Workflow

### Initialization Sequence

1. **Load particle domain** from HDF5
2. **Compute grid dimensions** from particle bounds + padding
3. **Initialize material properties**:
   - Load from HDF5 if available (spatially-varying)
   - Fall back to JSON/CLI uniform values
4. **Convert elastic moduli** to Lamé parameters (μ, λ, K)
5. **Initialize particle state** (F=I, σ=0, D=0)
6. **Geostatic initialization** (if enabled):
   - Compute target stress from depth
   - Initialize F to encode prestress
7. **CFL analysis**: Estimate timestep limits, wave speeds

### Main Loop Structure

```
for bigStep in [restart_bigStep ... bigSteps]:
    
    # === PHASE 1: Combined MPM+XPBD ===
    apply_gravity_pulse()  # if configured
    
    while (counter < nSteps) and (residual > threshold):
        
        # MPM Step (every dt)
        compute_stress_from_F_trial()  # includes return mapping, damage
        P2G()                          # scatter mass, momentum, stress forces
        grid_operations()              # normalize, gravity, boundaries
        G2P()                          # gather velocity, update F, position
        
        # XPBD Step (every dtxpbd)
        if counter % mpmStepsPerXpbdStep == 0:
            build_spatial_hash_grid()
            reset_mpm_contact_labels()  # label 0 → 1
            for iteration in xpbd_iterations:
                solve_boundary_contacts()
                solve_particle_contacts()  # sets label → 0 if MPM contacts XPBD
                apply_position_corrections()
            apply_mpm_contact_impulses()
            update_velocities()
            sleep_particles()
            swell_particles()
        
        # Convergence & early termination
        compute_residual()
        check_damage_stall()
    
    # === PHASE 2: XPBD-only (optional) ===
    if xpbdOnlyDuration > 0:
        for step in xpbdOnlySteps:
            xpbdSimulationStep(xpbd_only=True)  # MPM frozen
            check_sleep_termination()
    
    # === Big step completion ===
    save_checkpoint()  # VTP with full restart data
    deactivate_particles_below_datum()
    apply_creep()  # damage-dependent strength reduction
```

### Output Files

**Particle VTP files** (`sim_step_BBBB_CCCCCC_tTTTT_TTTT_particles.vtp`):
- `positions`: (N, 3) particle centers
- `radius`: (N,) particle radii
- `velocity`: (N, 3) velocity vectors
- `damage`: (N,) damage scalar [0, 1]
- `mean_stress`: (N,) tr(σ)/3
- `von_mises`: (N,) equivalent stress
- `stress_tensor`: (N, 6) symmetric stress components (xx, yy, zz, xy, xz, yz)
- `ys`: (N,) current yield stress
- `effective_ys`: (N,) damage-modified yield stress
- `active_label`: (N,) 0=sleeping, 1=active
- `material_label`: (N,) 0=locked MPM, 1=MPM, 2=XPBD

**Checkpoint data** (at big step boundaries):
- All above plus: F, C, accumulated_strain, ys_base, material properties
- Enables full simulation restart

**Grid VTI files** (optional):
- `mass`: 3D scalar field of grid mass distribution

## Implementation Details

### Material Label System

```
materialLabel = 0: MPM continuum (in contact with XPBD, CANNOT transition)
materialLabel = 1: MPM continuum (not in contact, CAN transition when D ≥ 1)
materialLabel = 2: XPBD discrete particles (post-failure)
```

**Contact-based transition locking** (`mpm_contact_transition_lock`):
1. Before each XPBD iteration set: Reset all label 0 → label 1
2. During contact detection: If MPM particle contacts XPBD particle, set label → 0
3. In return mapping: Only label == 1 particles can transition to label 2

**Purpose**: Prevents cascading failure where supporting MPM material fails because the debris it supports is pulling on it.

### Critical Bug Prevention

**Ghost particle problem**: If `strainCriteria` is very large AND `softening > hardening`, particles can reach `ys ≤ 0` while still MPM. These become "ghost particles" with zero stiffness but still participating in grid operations.

**Solution** (implemented): Force transition to XPBD when `ys ≤ 0`, regardless of damage level:
```python
if yield_stress[p] < 0.0 or yield_eff < 0.0:
    damage[p] = 1.0  # Force full damage
```

### Numerical Stability Considerations

**CFL condition**:
```
dt < α · dx / c_p
```
where:
- c_p = √((K + 4μ/3) / ρ) is P-wave speed
- α ≈ 0.3-0.5 for MPM stability (lower than classical FEM CFL)
- dx = grid_particle_spacing_scale × particle_diameter

**Cell-crossing instability**: Particles moving > 0.5 cells per timestep cause stress oscillations. The code enforces:
```
max(||Δx||) < 0.5 · dx
```

**Tensile instability**: MPM has reduced accuracy in tension. Critical strain before numerical issues:
```
ε_critical ≈ 1 / grid_particle_spacing_scale
```

For `grid_particle_spacing_scale = 2.0`: stable up to ~50% strain
For `grid_particle_spacing_scale = 4.0`: stable up to ~25% strain

### Grid Resolution Guidelines

| Scale | Grid spacing | Accuracy | Cost | Use case |
|-------|-------------|----------|------|----------|
| 2.0 | 2× particle diameter | Lower | Fast | Quick tests, large domains |
| 4.0 | 4× particle diameter | Higher | Medium | **Default, production** |
| 8.0 | 8× particle diameter | Highest | Slow | Validation, small domains |

### Boundary Condition Notes

**Friction boundaries** (`boundaryCondition = "friction"`):
- Coulomb friction applied tangentially
- `boundFriction = 0`: Free-slip walls (K₀ conditions difficult to maintain)
- `boundFriction = 0.3-0.5`: Typical soil/rock friction

**K₀ initialization accuracy**:
- High friction (>0.3): Good K₀ maintenance
- Low friction (<0.1): Block slides, K₀ stress state degrades
- Recommendation: Use `boundFriction ≥ 0.3` for geostatic problems

## Validation & Testing

### Analytical Validation (Elastic Regime)

```bash
cd benchmarks/generalBenchmark
python analytical_validation.py --E 1e9 --nu 0.2 --density 5000
```

**Expected accuracy**: < 5% error vs analytical solution for:
- ε_max < 5% (linear elastic regime)
- Sufficient grid resolution (scale ≥ 4.0)
- Adequate damping (rpic_damping ~ 0.2)

### Coupling Validation

```bash
cd benchmarks/spatialBenchmark
python check_xpbd_completely_weightless.py
python analyze_stress_profiles.py
```

**Tests verify**:
- XPBD particles transfer weight to MPM via contacts
- Density scaling works correctly
- Stress profiles match analytical expectations

### Large Deformation Tests

```bash
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_elastic_soft.json
```

**Note**: Analytical solutions are INVALID beyond ε > 5-10%. Use large deformation tests for:
- Numerical stability verification
- Qualitative behavior assessment
- Phase transition testing

## Performance Optimization

### GPU Utilization
- Particle operations: O(N) - scales linearly
- Grid operations: O(G³) - scales with grid volume
- Contact detection: O(N) with spatial hashing (avoids O(N²))

### Memory Requirements
- ~500 bytes/particle (all state arrays)
- ~100 bytes/grid cell (mass, momentum, velocity)
- Example: 1M particles + 256³ grid ≈ 2.5 GB VRAM

### Tuning Recommendations

| Parameter | Effect of increase | Typical range |
|-----------|-------------------|---------------|
| `dt` | Faster but less stable | Auto (CFL-based) |
| `xpbd_iterations` | More accurate contacts | 4-10 |
| `rpic_damping` | More dissipation | 0.0-0.3 |
| `residualThreshold` | Faster convergence | 1e-8 to 1e-1 |
| `grid_particle_spacing_scale` | More accurate but slower | 2.0-4.0 |

### Convergence Tips
1. **Stiff materials** (E > 10⁹): Use small dt, high xpbd_iterations
2. **Soft materials** (E < 10⁷): Watch for excessive strain
3. **High gravity loading**: Use viscosity for stabilization
4. **Phase transitions**: Ensure smooth energy release (eff < 1.0)

## Troubleshooting

### Common Issues

**Particles exploding/flying away**:
- Check CFL condition (reduce dt)
- Reduce gravity_pulse_factor
- Increase viscosity (eta_shear, eta_bulk)
- Check for division by zero (ys = 0 without transition)

**Stress oscillations**:
- Increase grid_particle_spacing_scale
- Add rpic_damping
- Check boundary conditions

**Phase transition not occurring**:
- Check strainCriteria (might be too large)
- Verify damage is accumulating (output damage field)
- Ensure materialLabel = 1 (not 0 from contact locking)

**Simulation not converging**:
- Relax residualThreshold
- Increase minSteps
- Check for steady-state oscillations (add damping)

**Memory errors**:
- Reduce particle count
- Reduce grid resolution
- Use smaller domain

### Diagnostic Tools

```bash
# Verify HDF5 file structure
cd benchmarks/cavingBenchmark
python verify_hdf5_fields.py random_rock_domain.h5

# Check CFL on startup
python runMPMYDW.py --config config.json
# (CFL analysis printed automatically)

# Visualize stress profiles
cd benchmarks/spatialBenchmark
python analyze_stress_profiles.py
```

## References

### MPM Fundamentals
- Sulsky, D., Chen, Z., & Schreyer, H. L. (1994). A particle method for history-dependent materials. *Computer Methods in Applied Mechanics and Engineering*.
- Stomakhin, A., Schroeder, C., Chai, L., Teran, J., & Selle, A. (2013). A material point method for snow simulation. *ACM TOG*.

### Transfer Schemes
- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine particle-in-cell method. *ACM TOG*. (APIC)
- Jiang, C., Schroeder, C., Teran, J., Stomakhin, A., & Selle, A. (2017). The material point method for simulating continuum materials. *SIGGRAPH Course*.

### XPBD
- Macklin, M., Müller, M., & Chentanez, N. (2016). XPBD: Position-based simulation of compliant constrained dynamics. *Motion in Games*.

### Constitutive Modeling
- de Souza Neto, E. A., Peric, D., & Owen, D. R. J. (2011). *Computational Methods for Plasticity*. Wiley. (Logarithmic strain formulation)
- Simo, J. C. (1992). Algorithms for static and dynamic multiplicative plasticity. *CMAME*. (Return mapping algorithms)

## Citation

If you use this code in academic work, please cite:
```
[Publication information - to be added]
```

## License

[License information - to be specified]

## Contact

[Contact information - to be added]

## Acknowledgments

- Built on [NVIDIA Warp](https://github.com/NVIDIA/warp) for GPU acceleration
- MPM transfer schemes based on APIC (Jiang et al. 2015)
- XPBD formulation based on Macklin et al. (2016)
- Multiplicative plasticity following de Souza Neto et al. (2011)
- Particle Mechanics Protocols based on FS5
- Solid Mechanics Model based on LR4
```
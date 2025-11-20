# MPMWarpYDW

A GPU-accelerated 3D coupled simulation framework combining the Material Point Method (MPM) with Extended Position Based Dynamics (XPBD) for modeling large deformation, damage, phase transitions, and granular flow in geomaterials.

## Overview

This framework simulates the progressive failure and fragmentation of continuum materials (e.g., rock, soil) by:
1. **MPM Phase**: Modeling continuum deformation with elasto-plastic constitutive laws, damage accumulation, and stress-based failure
2. **Phase Transition**: Automatically converting failed material points into discrete particles when damage reaches critical values
3. **XPBD Phase**: Simulating post-failure granular dynamics with particle-particle contacts, friction, and cohesion

The coupling enables seamless simulation of processes like rock avalanches, landslides, and fragmentation under gravity and dynamic loading.

## Key Features

### Material Point Method (MPM)
- **Multiplicative elastoplasticity** with logarithmic strain measures
- **Dual constitutive models**: 
  - **Von Mises** (pressure-independent, J2 plasticity)
  - **Drucker-Prager** (pressure-dependent, frictional materials with dilatancy)
- **Return mapping** with non-associated flow for realistic granular behavior
- **Damage mechanics** based on accumulated plastic strain
- **Phase transition** from continuum to discrete particles at critical damage
- **Kelvin-Voigt viscoelasticity** for rate-dependent behavior
- **APIC transfer scheme** (Affine Particle-In-Cell) for reduced numerical dissipation
- **Optional RPIC** (Rotated PIC) for damping control
- **Grid-based boundary conditions** with Coulomb friction
- **Spatially-varying properties**: Load heterogeneous material fields from HDF5

### Extended Position Based Dynamics (XPBD)
- **Particle-particle contact resolution** using hash grid for efficient neighbor search
- **Particle-boundary contacts** with friction
- **Static/dynamic friction** transition based on velocity threshold
- **Particle cohesion** for modeling cementation
- **Particle sleeping** mechanism to improve efficiency
- **Particle swelling** to model volume expansion during fragmentation
- **Velocity clamping** to prevent excessive velocities during phase change

### Coupling Mechanics
- **Hybrid coupling approach**: 
  - **Mass via grid**: XPBD particles contribute mass to MPM grid (prevents spurious pulling forces)
  - **Momentum via contact**: Direct particle contact forces handle interaction (physically realistic)
- **Energy-based velocity initialization**: Elastic strain energy converts to kinetic energy during phase transition
- **Smooth velocity clipping**: Prevents shock loading when particles first enter XPBD regime

### Numerical Features
- **GPU acceleration** using NVIDIA Warp
- **Explicit time integration** with subcycling (fast MPM steps, slower XPBD steps)
- **Convergence-based adaptive stepping**: Outer loop continues until velocity residual converges
- **Optional creep model**: Damage-dependent long-term strength reduction

## Requirements

- Python 3.8+
- [NVIDIA Warp](https://github.com/NVIDIA/warp) (GPU-accelerated physics simulation)
- numpy, h5py, matplotlib, scipy, pyglet, pyvista, lxml, vtk

Install dependencies with:
```sh
pip install warp-lang numpy h5py matplotlib scipy pyglet pyvista lxml vtk
```

**Hardware**: CUDA-capable NVIDIA GPU (recommended: RTX 3000+ series)

## Quick Start

### Running a Benchmark

```bash
# Activate conda environment (if using)
conda activate fs5

# Run the general benchmark (quick test)
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json
```

### Creating Custom Simulations

1. **Prepare particle domain**: Create HDF5 file with required fields
   ```python
   # Required fields:
   # - 'x': particle positions, shape (3, n_particles)
   # - 'particle_volume': volume per particle, shape (n_particles,)
   
   # Optional spatial property fields:
   # - 'density', 'E', 'nu' - elastic properties
   # - 'ys', 'alpha' - yield properties
   # - 'hardening', 'softening' - plasticity
   # - 'eta_shear', 'eta_bulk' - viscosity
   # - 'strainCriteria' - damage threshold
   
   # See examples in:
   # - exampleDomains/createInputHDF5.py
   # - benchmarks/spatialBenchmark/createInputHDF5.py
   # - benchmarks/cavingBenchmark/createRandomRockDomain.py
   ```

2. **Configure simulation**: Create a JSON config file (see `benchmarks/generalBenchmark/config_quick_test.json`)

3. **Run**: `python runMPMYDW.py --config your_config.json`

### Spatially-Varying Material Properties

The simulator supports **heterogeneous materials** with spatially-varying properties stored in HDF5 files. This enables realistic modeling of rock masses with spatial variability using Gaussian random fields or structured property distributions.

#### Available Benchmarks:

**1. Spatial Benchmark** - Demonstration of property loading:
```bash
cd benchmarks/spatialBenchmark
python createInputHDF5.py  # Generate example with spatial properties
cd ../..
python runMPMYDW.py --config ./benchmarks/spatialBenchmark/config_spatial.json
```

**2. Caving Benchmark** - Large-scale heterogeneous rock domain:
```bash
cd benchmarks/cavingBenchmark  
python createRandomRockDomain.py  # Generate 50×50×200m domain (522k particles)
python visualizeRandomRockDomain.py  # Inspect property fields
cd ../..
python runMPMYDW.py --config ./benchmarks/cavingBenchmark/config_random_rock.json
```

**3. Coupling Test** - MPM-XPBD interaction validation:
```bash
python runMPMYDW.py --config ./benchmarks/spatialBenchmark/config_coupling_test.json
```

#### Supported Spatial Properties:
- `density`, `E`, `nu` - Elastic properties
- `ys` - Yield stress (Von Mises or Drucker-Prager cohesion)
- `alpha` - Drucker-Prager pressure sensitivity (friction angle effect)
- `hardening`, `softening` - Plasticity evolution parameters
- `eta_shear`, `eta_bulk` - Viscosity coefficients
- `strainCriteria` - Failure strain threshold for damage

If properties are not in HDF5, command-line/JSON config values are used as uniform defaults.

See `benchmarks/spatialBenchmark/QUICKSTART.md` and `benchmarks/cavingBenchmark/README_random_rock.md` for details.

## Mathematical Model

### MPM Constitutive Models

The framework supports two plasticity models selected via `constitutive_model` parameter:

#### **Von Mises (constitutive_model=0)**: Pressure-independent J2 plasticity
- Suitable for metals and materials with negligible friction
- Yield criterion: σ_eq ≤ (1 - D)·σ_y
- Associated flow (no dilatancy)

#### **Drucker-Prager (constitutive_model=1)**: Pressure-dependent frictional plasticity
- Suitable for granular materials, rocks, soils
- Yield criterion: σ_eq ≤ (1 - D)·(σ_y - α·p)
  - α = pressure sensitivity (related to friction angle)
  - p = mean stress (compression negative)
- Non-associated flow with dilatancy (volume expansion during shear)
- β = 0.3·α (dilation angle parameter)

Both models use **multiplicative elasto-plasticity** in logarithmic strain space:

1. **Elastic trial deformation gradient**: 
   - F_trial = (I + ∇v·dt)·F_n

2. **Logarithmic strain decomposition**:
   - F = U·Σ·V^T (SVD decomposition)
   - ε = log(Σ) = [log(σ₁), log(σ₂), log(σ₃)]
   - Deviatoric strain: ε_dev = ε - (tr(ε)/3)·I

3. **Kirchhoff stress** (logarithmic model):
   - τ = 2μ·ε + λ·tr(ε)·I
   - τ_dev = deviatoric part

4. **Yield criterion** (damage-modified):
   - **Von Mises**: σ_eq = ||τ_dev|| ≤ (1 - D)·σ_y
   - **Drucker-Prager**: σ_eq ≤ (1 - D)·(σ_y - α·p) where p = tr(τ)/3

5. **Return mapping** (if yielding):
   - Δγ = ||ε_dev|| - σ_y_eff/(2μ)
   - ε_plastic += √(2/3)·Δγ
   - ε_dev_corrected = ε_dev - (Δγ/||ε_dev||)·ε_dev
   - **Drucker-Prager only**: Add volumetric correction for dilatancy
     - ε_vol_plastic = β·Δγ (β = 0.3·α)

6. **Damage evolution**:
   - dD = dε_plastic / ε_critical
   - D = min(1.0, D + dD)

7. **Phase transition** (when D ≥ 1.0):
   - Material label: 1 (MPM) → 2 (XPBD)
   - Elastic strain energy → kinetic energy:
     - u = 0.5·τ:ε (strain energy density)
     - v_release = √(2·η·u/ρ) (η = efficiency factor)

8. **Viscous damping** (Kelvin-Voigt):
   - Strain rate: ε̇ = 0.5(∇v + ∇v^T)
   - Viscous stress: τ_visc = 2η_shear·ε̇_dev + η_bulk·tr(ε̇)·I

### XPBD Contact Resolution

Uses position-based dynamics with constraint projection:

1. **Particle-particle contacts**:
   - Constraint: C = ||x_i - x_j|| - (r_i + r_j) ≥ 0
   - Normal correction: λ_n = C
   - Friction: λ_f = max(μ·λ_n, -||v_t||·dt)
   - Position update: x += (δ_n + δ_f)·(w_i/(w_i + w_j))

2. **Particle-boundary contacts**:
   - Similar constraint for each boundary plane
   - Static/dynamic friction based on velocity threshold

3. **Iteration scheme**:
   - Multiple Gauss-Seidel iterations per timestep
   - Relaxation parameter for stability

### Grid Operations

1. **Particle-to-Grid (P2G)**:
   - Transfer mass: m_i += Σ_p w_ip·m_p
   - Transfer momentum: (mv)_i += Σ_p w_ip·[m_p·v_p + dt·f_p]
   - Stress force: f_p = -V_p·σ_p·∇w_ip

2. **Grid Update**:
   - Normalize: v_i = (mv)_i / m_i
   - Add gravity: v_i += g·dt
   - Apply boundary conditions (friction walls)

3. **Grid-to-Particle (G2P)**:
   - Update velocity: v_p = Σ_i w_ip·v_i
   - Update position: x_p += v_p·dt
   - Update APIC matrix: C_p = Σ_i v_i⊗∇w_ip
   - Update deformation: F_trial = (I + ∇v·dt)·F_n

## Configuration Parameters

### Time Stepping
- `dt`: MPM timestep (typically 1e-3 to 1e-4 s)
- `dtxpbd`: XPBD timestep (typically 10× larger than dt)
- `nSteps`: Steps per convergence loop
- `bigSteps`: Number of outer convergence loops
- `residualThreshold`: Velocity convergence criterion (m/s per particle radius)

### Material Properties
- `density`: Material density (kg/m³)
- `E`: Young's modulus (Pa)
- `nu`: Poisson's ratio (dimensionless)
- `constitutive_model`: 0=Von Mises, 1=Drucker-Prager
- `ys`: Yield stress (Pa) - for Von Mises, or cohesion for Drucker-Prager
- `alpha`: Pressure sensitivity (Drucker-Prager only, dimensionless, typically 0.2-0.5)
- `hardening`: Strain hardening parameter (dimensionless)
- `softening`: Strain softening parameter (dimensionless)
- `eta_shear`: Shear viscosity (Pa·s)
- `eta_bulk`: Bulk viscosity (Pa·s)

### Damage & Phase Transition
- `strainCriteria`: Critical plastic strain for full damage (dimensionless)
- `eff`: Energy release efficiency during phase change (0-1)

### Grid & Domain
- `domainFile`: Path to HDF5 particle domain file
  - **Required fields**: `x` (shape 3×n_particles), `particle_volume` (shape n_particles)
  - **Optional fields**: Spatial property arrays (see Spatially-Varying Properties section)
- `grid_padding`: Padding around particle domain (m)
- `grid_particle_spacing_scale`: Grid spacing = particle_diameter × scale
- `K0`: Lateral earth pressure coefficient for initial geostatic stress (dimensionless, typically 0.4-0.7)

### XPBD Parameters
- `xpbd_iterations`: Contact solver iterations per step
- `xpbd_relaxation`: SOR relaxation parameter (0-1)
- `dynamicParticleFriction`: Friction coefficient (moving particles)
- `staticParticleFriction`: Friction coefficient (static particles)
- `staticVelocityThreshold`: Velocity threshold for static/dynamic transition
- `particle_cohesion`: Cohesive distance (m)
- `sleepThreshold`: Sleep velocity threshold (m/s per radius)

### Damping & Integration
- `rpic_damping`: RPIC damping factor (0 = APIC, 1 = full RPIC, -1 = PIC)
- `grid_v_damping_scale`: Grid velocity scaling per step
- `update_cov`: Enable covariance tracking (for future anisotropy)

### Swelling (Fragmentation)
- `swellingRatio`: Particle radius increase ratio after phase change
- `swellingActivationFactor`: Distance threshold for swelling activation
- `swellingMaxFactor`: Distance threshold for full swelling

### Boundaries & Visualization
- `boundFriction`: Friction on domain boundaries
- `gravity`: Gravitational acceleration (m/s², typically -9.81)
- `render`: Enable real-time rendering (0/1)
- `color_mode`: Particle coloring scheme ("damage", "state", "effective_ys", "velocity", etc.)
- `saveFlag`: Enable output file saving (0/1)
- `outputFolder`: Directory for output VTK files

## Project Structure

```
MPMWarpYDW/
├── runMPMYDW.py                 # Main simulation driver
├── config.json                  # Default configuration
├── README.md                    # This file
│
├── utils/                       # Core simulation modules
│   ├── getArgs.py              # Configuration argument parser
│   ├── simulationRoutines.py   # High-level MPM/XPBD orchestration
│   ├── mpmRoutines.py          # MPM kernels (stress, P2G, G2P, damage)
│   ├── xpbdRoutines.py         # XPBD kernels (contacts, integration)
│   ├── fs5PlotUtils.py         # Visualization utilities
│   └── fs5RendererCore.py      # OpenGL renderer
│
├── exampleDomains/              # Input particle geometries
│   ├── createInputHDF5.py      # Script to generate particle domains
│   ├── annular_arch_particles.h5   # Example: annular arch geometry
│   └── annular_arch_particles.vtp  # VTK visualization
│
├── benchmarks/                  # Test cases
│   ├── generalBenchmark/
│   │   ├── config_quick_test.json  # Quick validation test
│   │   └── README.md               # Benchmark documentation
│   ├── spatialBenchmark/           # Spatial properties demos
│   │   ├── createInputHDF5.py      # Generate domain with spatial properties
│   │   ├── createCouplingTest.py   # Generate MPM-XPBD coupling test
│   │   ├── config_spatial.json     # Configuration for spatial demo
│   │   ├── config_coupling_test.json  # MPM-XPBD coupling validation
│   │   ├── plot_yield_surface.py   # Visualize Drucker-Prager yield surfaces
│   │   ├── QUICKSTART.md           # Quick start guide
│   │   └── README.md               # Detailed documentation
│   └── cavingBenchmark/            # Large-scale rock caving simulations
│       ├── createRandomRockDomain.py  # Generate 50×50×200m heterogeneous domain
│       ├── visualizeRandomRockDomain.py  # Visualize property fields
│       ├── verify_hdf5_fields.py   # Verify HDF5 file correctness
│       ├── config_random_rock.json # Configuration for random rock domain
│       └── README_random_rock.md   # Detailed documentation
│
└── output/                      # Simulation results (VTK files)
    ├── sim_step_XXXXXX_particles.vtp
    └── sim_step_XXXXXX_grid.vti
```

## Simulation Workflow

### Initialization
1. Load particle domain from HDF5
2. Compute grid dimensions and spacing
3. Initialize material properties (E, ν, σ_y, etc.)
4. Convert elastic moduli to Lamé parameters (μ, λ)
5. Initialize particle state (F=I, stress=0, damage=0)

### Main Loop (bigSteps)
For each big step:

1. **Reset convergence**: residual = ∞
2. **Inner MPM loop** (until convergence or nSteps):
   
   **MPM Step** (every dt):
   - Compute stress from F_trial (return mapping + damage)
   - P2G: Transfer stress and APIC momentum
   - Grid operations: Normalize, add gravity, apply boundaries
   - G2P: Update particle x, v, C, F_trial
   
   **XPBD Step** (every dtxpbd):
   - Build spatial hash grid
   - Integrate particles (gravity)
   - Solve contacts (multiple iterations):
     - Particle-boundary contacts
     - Particle-particle contacts
     - Apply position corrections
   - Update particle velocities from positions
   - Apply sleeping, velocity clipping, swelling
   
   **Convergence Check**:
   - Compute residual = Σ(||v_p||/r_p) / N_active
   - If residual < threshold and counter > minSteps: break

3. **Creep update**: Apply time-dependent strength reduction (optional)
4. **Visualization/Output**: Save VTK files if enabled

### Output Files
- **Particles** (VTP): Position, velocity, radius, stress, damage, material label, accumulated strain
- **Grid** (VTI): Grid mass distribution (for visualization)

## Important Implementation Details

### Material Failure and Phase Transition

When `damage >= 1.0` or `yield_stress <= 0`, material transitions from MPM (label=1) to XPBD (label=2):

1. **Stress-free transition**: Material properties set to zero (μ=0, λ=0, ys=0) to prevent spurious forces
2. **Energy release**: Elastic strain energy converts to kinetic energy with efficiency factor `eff`
3. **Velocity initialization**: v_release = √(2·eff·u/ρ) added in direction of motion
4. **One-way transition**: XPBD particles never return to MPM

⚠️ **Critical**: If `strainCriteria` is very large (preventing XPBD transition) but `softening > hardening`, material can reach `ys <= 0` and become a "ghost" MPM particle (zero stiffness but still MPM). This causes P2G failure. **Solution**: Either:
- Use `strainCriteria` small enough to allow transition before ys→0
- Ensure `hardening >= softening` to prevent yield stress degradation
- Force transition when `ys <= 0` even if `damage < 1.0`

### Plasticity Irreversibility

Plastic deformation is enforced as **permanent** through three mechanisms:

1. **Monotonic plastic strain accumulation**: `accumulated_strain += Δε_plastic` (never decreases)
2. **Elastic strain reduction**: Plastic part removed from trial strain during return mapping
3. **Implicit reference configuration update**: Only elastic deformation gradient F_elastic is stored and returned

Upon unloading: stress→0, elastic strain→0, but `accumulated_strain > 0` remains permanently.

### MPM-XPBD Coupling Strategy

The coupling uses a **hybrid approach** to handle interaction between continuum (MPM) and discrete (XPBD) phases:

- **Mass transfer via grid (P2G)**: XPBD particles contribute **only mass** to the MPM grid, not momentum
  - Prevents spurious "pulling" forces when XPBD particles move away from MPM continuum
  - Allows MPM to feel the presence of XPBD material without artificial momentum coupling
  
- **Momentum transfer via contact forces**: Direct particle-particle contact detection between XPBD and MPM particles
  - Properly handles compression and interaction forces
  - Contact forces computed separately from grid operations
  - More physically realistic than momentum smearing through the grid

This hybrid approach provides stable, physically-meaningful coupling without artificial tensile forces or momentum artifacts.

## Code Architecture

### Kernel Design
All compute-intensive operations are implemented as **Warp kernels** for GPU execution:

- **@wp.kernel**: Thread-parallel operations (one thread per particle/grid cell)
- **@wp.func**: Device functions callable from kernels
- Atomic operations for scatter operations (P2G)

### Material Labels
- `0`: Inactive (not simulated)
- `1`: MPM continuum material
- `2`: XPBD discrete particles (post-failure)

### Active Labels
- `0`: Sleeping/inactive
- `1`: Active (participating in simulation)

## Physics Validity & Limitations

### Assumptions
- Small rotation per timestep (for logarithmic strain validity)
- Isotropic materials (no pre-existing fabric or anisotropy)
- Rate-independence (viscous terms for numerical stability only, not rate-dependent strength)
- No thermal coupling
- No pore pressure/fluid coupling
- Adiabatic energy release during phase transition

### Constitutive Model Limitations

**Von Mises**:
- ✅ Suitable for: Metals, clays (undrained), materials with negligible friction
- ❌ Not suitable for: Granular materials, rocks, sands (use Drucker-Prager)

**Drucker-Prager**:
- ✅ Suitable for: Rocks, soils, granular materials, frictional-cohesive materials
- ✅ Pressure-dependent yielding (deeper = stronger)
- ✅ Non-associated flow with dilatancy (volume expansion during shear)
- ⚠️ Simplified: No tension cutoff, no corner flow (Mohr-Coulomb equivalent)

### Numerical Considerations
- **CFL condition**: dx/dt > max(particle velocity)
- **Yield stress**: Should be >> bulk modulus × strain increment for stability
- **Damage softening**: Risk of "ghost particles" if `strainCriteria` too large (see Implementation Details)
- **Grid resolution**: Typically 2-4× particle diameter for convergence
- **XPBD iterations**: 4-10 iterations needed for accurate contact resolution

### Validation
- Energy conservation (elastic regime, pre-damage)
- Momentum conservation (P2G + G2P)
- Yield surface accuracy (return mapping within tolerance)
- Contact resolution (penetration < 1% particle radius)
- Plastic irreversibility (load-unload cycles verify permanent strain)

## Citation

If you use this code in academic work, please cite:
```
[Your publication information here]
```

## License

[Specify license]

## Contact

[Your contact information]

## Acknowledgments

- Built on [NVIDIA Warp](https://github.com/NVIDIA/warp)
- MPM transfer schemes based on APIC (Jiang et al. 2015)
- XPBD formulation based on Macklin et al. (2016)

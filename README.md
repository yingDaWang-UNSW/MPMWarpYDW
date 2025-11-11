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
- **Von Mises yield criterion** with return mapping
- **Damage mechanics** based on accumulated plastic strain
- **Phase transition** from continuum to discrete particles at critical damage
- **Kelvin-Voigt viscoelasticity** for rate-dependent behavior
- **APIC transfer scheme** (Affine Particle-In-Cell) for reduced numerical dissipation
- **Optional RPIC** (Rotated PIC) for damping control
- **Grid-based boundary conditions** with Coulomb friction

### Extended Position Based Dynamics (XPBD)
- **Particle-particle contact resolution** using hash grid for efficient neighbor search
- **Particle-boundary contacts** with friction
- **Static/dynamic friction** transition based on velocity threshold
- **Particle cohesion** for modeling cementation
- **Particle sleeping** mechanism to improve efficiency
- **Particle swelling** to model volume expansion during fragmentation
- **Velocity clamping** to prevent excessive velocities during phase change

### Coupling Mechanics
- **Two-way coupling** via grid: XPBD particles contribute mass and momentum to MPM grid
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

1. **Prepare particle domain**: Create HDF5 file with particle positions and volumes
   ```python
   # See exampleDomains/createInputHDF5.py for example
   ```

2. **Configure simulation**: Create a JSON config file (see `benchmarks/generalBenchmark/config_quick_test.json`)

3. **Run**: `python runMPMYDW.py --config your_config.json`

## Mathematical Model

### MPM Constitutive Model

The framework uses a **multiplicative elasto-plasticity** model in the logarithmic strain space:

1. **Elastic trial deformation gradient**: 
   - F_trial = (I + ∇v·dt)·F_n

2. **Logarithmic strain decomposition**:
   - F = U·Σ·V^T (SVD decomposition)
   - ε = log(Σ) = [log(σ₁), log(σ₂), log(σ₃)]
   - Deviatoric strain: ε_dev = ε - (tr(ε)/3)·I

3. **Kirchhoff stress** (logarithmic model):
   - τ = 2μ·ε + λ·tr(ε)·I
   - τ_dev = deviatoric part

4. **Von Mises yield criterion** with damage:
   - σ_eq = ||τ_dev|| (equivalent stress)
   - σ_y_eff = (1 - D)·σ_y (damage-modified yield stress)
   - Yield condition: σ_eq ≤ σ_y_eff

5. **Return mapping** (if yielding):
   - Δγ = ||ε_dev|| - σ_y_eff/(2μ)
   - ε_plastic += √(2/3)·Δγ
   - ε_corrected = ε - (Δγ/||ε_dev||)·ε_dev

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
- `ys`: Yield stress (Pa)
- `hardening`: Strain hardening parameter (dimensionless)
- `softening`: Strain softening parameter (dimensionless)
- `eta_shear`: Shear viscosity (Pa·s)
- `eta_bulk`: Bulk viscosity (Pa·s)

### Damage & Phase Transition
- `strainCriteria`: Critical plastic strain for full damage (dimensionless)
- `eff`: Energy release efficiency during phase change (0-1)

### Grid & Domain
- `domainFile`: Path to HDF5 particle domain file
- `grid_padding`: Padding around particle domain (m)
- `grid_particle_spacing_scale`: Grid spacing = particle_diameter × scale

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
- `saveFlag`: Enable output file saving (0/1)

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
│   └── generalBenchmark/
│       ├── config_quick_test.json  # Quick validation test
│       ├── README.md               # Benchmark documentation
│       └── run_benchmark.bat       # Windows batch script
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
- **Particles** (VTP): Position, velocity, radius, stress, damage, material label
- **Grid** (VTI): Grid mass distribution (for visualization)

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
- Small rotation per timestep (for logarithmic strain)
- Isotropic materials (no pre-existing fabric)
- Rate-independence (viscous terms for numerical stability only)
- No thermal coupling
- No pore pressure/fluid coupling

### Numerical Considerations
- **CFL condition**: dx/dt > max(particle velocity)
- **Yield stress**: Should be >> bulk modulus × strain increment for stability
- **Grid resolution**: Typically 4× particle diameter
- **Damage convergence**: Requires multiple big steps for quasi-static loading

### Validation
- Energy conservation (elastic regime)
- Momentum conservation (P2G + G2P)
- Yield surface accuracy (return mapping)
- Contact resolution (penetration < 1% particle radius)

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

# Rock Caving Benchmark

This benchmark demonstrates the Drucker-Prager constitutive model for simulating rock caving systems, where pressure-dependent yielding and tension cutoff are critical.

## Physical System

The benchmark simulates an arch structure under gravity, representing a simplified caving scenario where:
- Rock material exhibits pressure-dependent strength (higher confining pressure = higher yield strength)
- Tensile failure occurs when tension exceeds the cutoff
- Material yields and flows plastically when the Drucker-Prager criterion is satisfied
- Damaged material (D=1) transitions to discrete XPBD particles for post-failure dynamics

## Drucker-Prager Model Parameters

### Material Properties
- **Density**: 2700 kg/m³ (typical rock)
- **Young's Modulus**: 10 GPa (soft-to-medium strength rock)
- **Poisson's Ratio**: 0.25
- **Constitutive Model**: 1 (Drucker-Prager)

### Drucker-Prager Parameters
- **Cohesion (c)**: 5 MPa - represents inherent shear strength
- **Friction Angle (φ)**: 35° - controls pressure sensitivity (α = 6sinφ/(3-sinφ) ≈ 1.45 in plane strain)
- **Dilation Angle (ψ)**: 5° - controls volumetric expansion during plastic flow (non-associated flow rule)
- **Tension Cutoff**: 1 MPa - maximum tensile stress before failure

### Damage Parameters
- **Strain Criteria**: 0.01 (1% plastic strain triggers damage accumulation)
- **Hardening**: 0.0 (no strain hardening)
- **Softening**: 0.0 (no post-peak softening in this simplified version)

## Yield Criterion

The Drucker-Prager yield function in plane strain:
```
F = q + α·p - k
```
where:
- `q = √(3J₂)` is the deviatoric stress (von Mises equivalent)
- `p = -tr(σ)/3` is the mean pressure (positive in compression)
- `α = 6sinφ/(3-sinφ)` is the friction coefficient
- `k = 6c·cosφ/(3-sinφ)` is the cohesion parameter
- Cohesion degrades with damage: `c_eff = c·(1-D)`

Plastic flow uses non-associated rule:
```
∂g/∂σ = ∂q/∂σ + β·∂p/∂σ
```
where `β = 6sinψ/(3-sinψ)` controls dilatancy.

## Running the Benchmark

From the repository root, run:
```bash
python runMPMYDW.py --config benchmarks/cavingBenchmark/config_caving.json
```

Or use PowerShell script:
```powershell
.\benchmarks\cavingBenchmark\run_benchmark.bat
```

## Expected Behavior

1. **Initial Phase**: Elastic deformation under gravity
2. **Yielding**: Material near the arch base experiences high compression, activating Drucker-Prager plasticity
3. **Tension Failure**: Material at the arch crown exceeds tension cutoff, initiating cracks
4. **Progressive Failure**: Damage accumulates (D → 1), weakening cohesion
5. **Transition to XPBD**: Fully damaged particles (D=1) become discrete, exhibiting post-failure flow/collapse

## Key Physics

- **Pressure-Dependent Strength**: Unlike Von Mises, yield strength increases with confining pressure
- **Tension Cutoff**: Prevents unrealistic tensile strength in rock
- **Non-Associated Flow**: Dilatancy (ψ < φ) allows volumetric expansion without excessive dilation
- **Cohesion Degradation**: c_eff = c·(1-D) couples damage to strength loss

## Output

Results saved to `output/` directory:
- `sim_step_XXXXXX_particles.vtp`: Particle positions, stress, damage, etc.
- `sim_step_XXXXXX_grid.vti`: Grid velocity field

Visualize with ParaView or similar VTK viewer.

## Comparison to Von Mises

To compare with Von Mises (J2) plasticity, set:
```json
"constitutive_model": 0
```
in config and adjust `ys` (yield stress) to desired value. Von Mises does not account for pressure-dependency or tension cutoff.

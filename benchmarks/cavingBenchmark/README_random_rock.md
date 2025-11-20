# Random Rock Domain Benchmark

## Overview
A 50m × 50m × 200m heterogeneous rock domain with spatially varying material properties generated using Gaussian random fields. Designed to simulate realistic rock caving behavior with weak-to-medium strength rock prone to failure.

## Domain Specifications

### Geometry
- **Dimensions**: 50m (X) × 50m (Y) × 200m (Z)
- **Spacing**: 1.0 m
- **Total particles**: 522,801
- **Total mass**: ~1.36 × 10⁹ kg
- **Volume**: 500,000 m³

### Material Properties (Spatially Varying)

All properties are generated as Gaussian random fields with specified correlation lengths to create realistic spatial heterogeneity:

| Property | Range | Mean | Correlation Length |
|----------|-------|------|-------------------|
| Young's modulus (E) | 5-20 GPa | 12.5 GPa | 10 m (medium) |
| Yield stress (ys) | 10-100 MPa | 55 MPa | 5 m (short) |
| Cohesion (c) | 1-10 MPa | 5.5 MPa | 5 m (short) |
| Friction angle (φ) | 25-45° | 35° | 10 m (medium) |
| Alpha (Drucker-Prager) | 0.189-0.356 | 0.273 | derived from φ |
| Dilation angle (ψ) | 7.5-13.5° | 10.5° | 0.3 × φ |
| Density (ρ) | 2400-2800 kg/m³ | 2600 kg/m³ | 20 m (long) |
| Strain criteria (εf) | 0.001-0.05 | 0.0255 | 5 m (short) |
| Poisson's ratio (ν) | 0.2-0.3 | 0.25 | 20 m (long) |
| Shear viscosity (η) | 1×10⁵-1×10⁶ Pa·s | 5×10⁵ Pa·s | 10 m (medium) |

### Plasticity Parameters
- **Hardening**: 0.1 (slight initial hardening)
- **Softening**: 0.15 (moderate softening → net softening behavior)
- **Constitutive model**: Drucker-Prager (pressure-dependent yielding)

## Files

### Generation Scripts
- **`createRandomRockDomain.py`**: Generates the HDF5 domain file with random properties
  - Uses `scipy.ndimage.gaussian_filter` for spatial correlation
  - Reproducible with `random.seed(42)`
  
- **`visualizeRandomRockDomain.py`**: Visualizes property distributions
  - Creates slice plots at Z = 50m, 100m, 150m
  - Generates histograms showing property statistics
  - Outputs PNG files for inspection

### Domain Files
- **`random_rock_domain_50x50x200.h5`**: Main domain file (generated)
  - Contains all particle properties and state variables
  - ~100-200 MB file size

### Configuration
- **`config_random_rock.json`**: Simulation configuration
  - Timesteps: dt=1×10⁻⁵ s (MPM), dtxpbd=1×10⁻⁴ s (XPBD)
  - Drucker-Prager constitutive model
  - XPBD parameters tuned for rock contact
  - All material properties loaded from HDF5

### Visualizations
- **`random_rock_properties_visualization.png`**: Slice plots of all properties
- **`random_rock_properties_histograms.png`**: Statistical distributions

## Usage

### 1. Generate Domain
```bash
cd benchmarks/cavingBenchmark
python createRandomRockDomain.py
```
This creates `random_rock_domain_50x50x200.h5`.

### 2. Visualize Properties (Optional)
```bash
python visualizeRandomRockDomain.py
```
Opens matplotlib windows and saves PNG files.

### 3. Run Simulation
```bash
cd ../..
python runMPMYDW.py --config benchmarks/cavingBenchmark/config_random_rock.json
```

## Expected Behavior

### Physical Process
1. **Initial loading**: Gravitational loading creates compressive stress field
   - Bottom experiences ~5 MPa overburden stress (200m × 2600 kg/m³ × 9.81 m/s²)
   - Weak zones will yield first (low ys, low cohesion)

2. **Plastic yielding**: Material plastically deforms where stress exceeds yield
   - Drucker-Prager criterion: σ_eq > (1-D)(ys - α·p)
   - Pressure-dependent: deeper material has higher effective yield strength
   - Plastic strain accumulates, damage increases

3. **Progressive failure**: As damage reaches 1.0, material transitions to XPBD
   - Loses cohesion and stiffness
   - Becomes granular debris
   - Forms failure surfaces along weak zones

4. **Caving/collapse**: XPBD particles interact via contact forces
   - Volume preservation (incompressibility)
   - Friction between particles
   - Potential for large-scale collapse if geometry permits

### Computational Requirements
- **Memory**: ~4-8 GB (500k particles)
- **Time**: Depends on failure extent
  - If stable: 10-20 sec per 1000 steps
  - If widespread failure: 30-60 sec per 1000 steps (many XPBD particles)
- **Recommended**: 50,000 steps (~8-12 hours on modern CPU)

## Validation

### Check Property Fields
Look at the visualization PNGs:
- **Smooth patches**: Gaussian correlation creates realistic zones
- **No sharp boundaries**: Gradual transitions between properties
- **Reasonable ranges**: All values within physically realistic bounds

### Monitor Simulation
- **Stress distribution**: Should show ~5 MPa at bottom, decreasing upward
- **Damage evolution**: Weak zones (low ys, low E) should damage first
- **Phase transition**: Material with D=1.0 should switch to XPBD (materialLabel=2)
- **Caving pattern**: If failures connect, should see downward propagation

## Customization

### Modify Property Ranges
Edit `createRandomRockDomain.py`:
```python
# Example: Make rock weaker (more prone to failure)
ys_mean = 30e6  # Reduce from 55 MPa to 30 MPa
ys_std = 15e6

# Example: Increase heterogeneity
correlation_length_short = 3.0  # Reduce from 5m to 3m (more variable)
```

### Modify Domain Size
```python
Lx, Ly, Lz = 100.0, 100.0, 100.0  # Cubic domain
spacing = 2.0  # Coarser resolution (faster simulation)
```

### Adjust Failure Behavior
Edit `config_random_rock.json`:
```json
{
  "eff": 1.0,  // Energy release efficiency (0.5 = 50% converts to kinetic)
  "strainCriteria": 0.02,  // Lower = fails sooner (if overriding HDF5)
  "gravity": -19.62  // Double gravity to accelerate failure
}
```

## Notes

- **Correlation lengths** control spatial variability:
  - Short (5m): Highly variable, patchy properties
  - Medium (10m): Moderate zones
  - Long (20m): Large uniform regions

- **Net softening** (softening > hardening) ensures yield stress degrades during plastic flow, promoting progressive failure

- **Drucker-Prager** accounts for confining pressure effect: deeper rock is stronger

- **Strain criteria range** (0.001-0.05) means some regions fail after 0.1% plastic strain, others after 5%

## References
- Gaussian random field generation: `scipy.ndimage.gaussian_filter`
- Rock property ranges: Typical values for sedimentary to weak crystalline rock
- Drucker-Prager model: Classic pressure-dependent plasticity for geomaterials

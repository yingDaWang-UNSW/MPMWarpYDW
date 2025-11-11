# General Benchmark

This benchmark provides a quick test to confirm the MPM-XPBD simulation is running correctly.

## Files

- `annular_arch_particles.h5` - Particle domain data in HDF5 format
- `annular_arch_particles.vtp` - Particle domain visualization in VTP format
- `createInputHDF5.py` - Script to regenerate the particle domain if needed
- `config_quick_test.json` - Configuration for a quick test run

## Running the Benchmark

To run this benchmark from the project root directory:

```bash
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json
```

## Test Configuration

The quick test configuration uses:
- **Time steps**: 5000 steps per big step (for quick verification)
- **Big steps**: 5 outer loop iterations
- **Residual threshold**: 0.5 (convergence criterion)
- **MPM dt**: 0.001s
- **XPBD dt**: 0.01s
- **Rendering**: Disabled (for faster execution)
- **Saving**: Enabled (to verify output generation)
- **Domain**: Annular arch particle geometry

This configuration is designed to run quickly while still exercising all major simulation components.

## Expected Behavior

The simulation should:
1. Load the particle domain successfully
2. Initialize the MPM grid
3. Run the MPM-XPBD coupled simulation
4. Generate output files in the `output/` directory
5. Complete without errors
6. Show convergence of particle velocities over time

## Customization

You can modify `config_quick_test.json` to:
- Increase `nSteps` for longer simulations per big step
- Increase `bigSteps` for more outer loop iterations
- Adjust `residualThreshold` for tighter/looser convergence
- Enable `render: 1` to visualize the simulation
- Adjust material properties (E, nu, ys, etc.)
- Change grid resolution via `grid_particle_spacing_scale`

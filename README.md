# MPMWarpYDW

This repository contains a 3D Coupled Material Point Method (MPM) and Extended Position Based Dynamics simulation framework with visualization and utility tools.

## Directory Structure

- `runMPMYDW.py`  
  Main script to run MPM simulations.

- `exampleDomains/`  
  Example input domains and scripts for generating input data.

- `output/`  
  Output directory for simulation results (e.g., VTI/VTP files for visualization).

- `utils/`  
  Utility Python modules:
  - `fs5RendererCore.py`: OpenGL-based 3D renderer for visualizing simulation results.
  - `fs5PlotUtils.py`: Plotting and file export utilities (e.g., VTK, HDF5/XDMF).
  - `mpmRoutines.py`: Core MPM kernels and routines (e.g., stress computation, P2G/G2P, boundary conditions).
  - `xpbdRoutines.py`: Extended Position-Based Dynamics routines.

## Requirements

- Python 3.8+
- [Warp](https://github.com/NVIDIA/warp)
- numpy, h5py, matplotlib, scipy, pyglet, pyvista, lxml, vtk

Install dependencies with:
```sh
pip install warp-lang numpy h5py matplotlib scipy pyglet pyvista lxml vtk
```

## Running a Simulation

1. Prepare input data (see `exampleDomains/createInputHDF5.py`).
2. Run the main simulation:
   ```sh
   python runMPMYDW.py
   ```
   Outputs will be saved in the `output/` directory.

## Visualization

- The renderer (`fs5RendererCore.py`) provides real-time OpenGL visualization.
- Output files (`.vti`, `.vtp`) can be viewed in ParaView or similar tools.
- Use `fs5PlotUtils.py` for exporting results to VTK or HDF5/XDMF formats.

## Main Components

- **MPM Kernels**: See [`mpmRoutines.py`](utils/mpmRoutines.py) for stress computation, P2G, G2P, and boundary handling.
- **XPBD Kernels**: See [`xpbdRoutines.py`](utils/mpmRoutines.py) for particle-particle contact, and boundary handling.
- **Coupling Scheme**: XPBD contributes to the MPM P2G, but not stress or G2P. MPM contributes to XPBD collisions as static particles.

- **Rendering**: See [`fs5RendererCore.py`](utils/fs5RendererCore.py) for OpenGL-based visualization.
- **Plotting/Export**: See [`fs5PlotUtils.py`](utils/fs5PlotUtils.py) for file export and plotting utilities.

## License

See source files for NVIDIA copyright.

---

For more details, see comments in the code and example scripts.
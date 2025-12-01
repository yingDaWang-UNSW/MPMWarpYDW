#TODO: heterogeneity and anisotopy: heterogeneity is spatially scalar, thats fine. internal constitutive anisotropy requires orientation tracking based on deformation
import warp as wp
import numpy as np
import h5py
import time
wp.init()
wp.config.verify_cuda = True
device = "cuda:0"
from utils import fs5PlotUtils
from utils import fs5RendererCore
from utils import mpmRoutines
from utils import simulationRoutines
from utils.simStates import SimState, MPMState, XPBDState
import numpy as np
np.seterr(over='raise')
from utils.getArgs import get_args
args = get_args()

# --- Domain & grid ---
domainFile = args.domainFile
h5file = h5py.File(domainFile, "r")
x = np.array(h5file["x"]).T
particle_volume = np.array(h5file["particle_volume"])
particleDiameter = np.mean(particle_volume) ** 0.33
dx = particleDiameter * args.grid_particle_spacing_scale
# x = x + np.random.rand(x.shape[0], x.shape[1])*0.05*dx # jitter to prevent stress chains
# delete the middle 20% of particles in z
# x = x[~((x[:, 2] > np.percentile(x[:, 2], 40)) & (x[:, 2] < np.percentile(x[:, 2], 60)))]
nPoints = x.shape[0]
print(f"Number of particles: {nPoints}")

# ========== Compute Domain Bounds ==========
# Compute initial bounds from particle positions with padding
particle_min = np.min(x, 0)
particle_max = np.max(x, 0)
mpmPadding = 2  # Increased from 2 to reduce boundary artifacts


invdx = 1.0 / dx

# Parse boundary padding mask (6-digit binary: -X,+X,-Y,+Y,-Z,+Z)
# 1 = use args.grid_padding (loose), 0 = use mpmPadding (tight)
boundary_mask = args.boundary_padding_mask
if len(boundary_mask) != 6 or not all(c in '01' for c in boundary_mask):
    raise ValueError(f"boundary_padding_mask must be 6-digit binary string, got: {boundary_mask}")

padding_flags = [int(c) for c in boundary_mask]
print(f"Boundary padding mask: {boundary_mask} (-X,+X,-Y,+Y,-Z,+Z)")
print(f"  1 = loose padding ({args.grid_padding*dx:.2f}m), 0 = tight padding ({mpmPadding*dx:.2f}m)")

# Apply padding based on mask
# Format: [minX, maxX, minY, maxY, minZ, maxZ]
minBounds = np.array([
    particle_min[0] - (args.grid_padding*dx if padding_flags[0] else mpmPadding*dx),  # -X
    particle_min[1] - (args.grid_padding*dx if padding_flags[2] else mpmPadding*dx),  # -Y
    particle_min[2] - (args.grid_padding*dx if padding_flags[4] else mpmPadding*dx)   # -Z
])

maxBounds = np.array([
    particle_max[0] + (args.grid_padding*dx if padding_flags[1] else mpmPadding*dx),  # +X
    particle_max[1] + (args.grid_padding*dx if padding_flags[3] else mpmPadding*dx),  # +Y
    particle_max[2] + (args.grid_padding*dx if padding_flags[5] else mpmPadding*dx)   # +Z
])

gridDims = np.ceil((maxBounds - minBounds) / dx).astype(int)
print(f'GRID: {gridDims}')
print(f'Domain bounds: X=[{minBounds[0]:.2f}, {maxBounds[0]:.2f}], Y=[{minBounds[1]:.2f}, {maxBounds[1]:.2f}], Z=[{minBounds[2]:.2f}, {maxBounds[2]:.2f}]')

# Check if HDF5 contains spatial property arrays
spatial_properties_available = {}
spatial_property_names = ['density', 'E', 'nu', 'ys', 'alpha', 'hardening', 'softening', 
                          'eta_shear', 'eta_bulk', 'strainCriteria']
for prop_name in spatial_property_names:
    if prop_name in h5file:
        spatial_properties_available[prop_name] = True
        print(f"  Found spatial array for: {prop_name}")
    else:
        spatial_properties_available[prop_name] = False

if any(spatial_properties_available.values()):
    print(f"Loading {sum(spatial_properties_available.values())} spatial property arrays from HDF5")

# ========== Initialize State Classes ==========
print(f"Running simulation with dt={args.dt}, dtxpbd={args.dtxpbd}, nSteps={args.nSteps}, bigSteps={args.bigSteps}, residualThreshold={args.residualThreshold}")

# Create state containers
sim = SimState(args, nPoints, device=device)
mpm = MPMState(args, sim.nPoints, tuple(gridDims), device=device)
xpbd = XPBDState(args, sim.nPoints, device=device)

# Store domain bounds in sim state as wp.vec3 for kernel use
sim.minBounds = wp.vec3(minBounds[0], minBounds[1], minBounds[2])
sim.maxBounds = wp.vec3(maxBounds[0], maxBounds[1], maxBounds[2])
sim.dx = dx
sim.invdx = invdx
mpm.boundaryPadding = mpmPadding  # Grid cells from edge where boundary conditions apply

# --- Material properties ---
# Load from HDF5 if available, otherwise use args/defaults
if spatial_properties_available.get('density', False):
    density = np.array(h5file["density"], dtype=np.float32)
    print(f"  Loaded density from HDF5: {density.min():.2e} to {density.max():.2e} kg/m³")
else:
    density = np.full(sim.nPoints, args.density, dtype=np.float32)

if spatial_properties_available.get('E', False):
    E = np.array(h5file["E"], dtype=np.float32)
    print(f"  Loaded E from HDF5: {E.min():.2e} to {E.max():.2e} Pa")
else:
    E = np.full(sim.nPoints, args.E, dtype=np.float32)

if spatial_properties_available.get('nu', False):
    nu = np.array(h5file["nu"], dtype=np.float32)
    print(f"  Loaded nu from HDF5: {nu.min():.3f} to {nu.max():.3f}")
else:
    nu = np.full(sim.nPoints, args.nu, dtype=np.float32)

if spatial_properties_available.get('ys', False):
    ys = np.array(h5file["ys"], dtype=np.float32)
    print(f"  Loaded ys from HDF5: {ys.min():.2e} to {ys.max():.2e} Pa")
else:
    ys = np.full(sim.nPoints, args.ys, dtype=np.float32)
    # Optional: Apply spatial modifications here if not using HDF5
    # ys = ys * (1 - (x[:, 2] - minBounds[2]) / (maxBounds[2] - minBounds[2]))**3

if spatial_properties_available.get('hardening', False):
    hardening = np.array(h5file["hardening"], dtype=np.float32)
    print(f"  Loaded hardening from HDF5: {hardening.min():.3f} to {hardening.max():.3f}")
else:
    hardening = np.full(sim.nPoints, args.hardening, dtype=np.float32)

if spatial_properties_available.get('softening', False):
    softening = np.array(h5file["softening"], dtype=np.float32)
    print(f"  Loaded softening from HDF5: {softening.min():.3f} to {softening.max():.3f}")
else:
    softening = np.full(sim.nPoints, args.softening, dtype=np.float32)

if spatial_properties_available.get('eta_shear', False):
    eta_shear = np.array(h5file["eta_shear"], dtype=np.float32)
    print(f"  Loaded eta_shear from HDF5: {eta_shear.min():.2e} to {eta_shear.max():.2e} Pa·s")
else:
    eta_shear = np.full(sim.nPoints, args.eta_shear, dtype=np.float32)

if spatial_properties_available.get('eta_bulk', False):
    eta_bulk = np.array(h5file["eta_bulk"], dtype=np.float32)
    print(f"  Loaded eta_bulk from HDF5: {eta_bulk.min():.2e} to {eta_bulk.max():.2e} Pa·s")
else:
    eta_bulk = np.full(sim.nPoints, args.eta_bulk, dtype=np.float32)

materialLabel = np.ones(sim.nPoints, dtype=np.int32)

activeLabel = np.ones(sim.nPoints, dtype=np.int32)

# --- Constitutive model selection and parameters ---
constitutive_model = args.constitutive_model

# Load alpha (Drucker-Prager) from HDF5 if available (before closing file)
if spatial_properties_available.get('alpha', False):
    alpha_array = np.array(h5file["alpha"], dtype=np.float32)
    print(f"  Loaded alpha from HDF5: {alpha_array.min():.3f} to {alpha_array.max():.3f}")
else:
    alpha_array = np.full(sim.nPoints, args.alpha, dtype=np.float32)

# Load strainCriteria from HDF5 if available (before closing file)
if spatial_properties_available.get('strainCriteria', False):
    strainCriteria_array = np.array(h5file["strainCriteria"], dtype=np.float32)
    print(f"  Loaded strainCriteria from HDF5: {strainCriteria_array.min():.3f} to {strainCriteria_array.max():.3f}")
else:
    strainCriteria_array = np.full(sim.nPoints, args.strainCriteria, dtype=np.float32)

# Close HDF5 file after loading all properties
h5file.close()

# --- Transfer material properties to GPU and initialize state classes ---
yMod = wp.array(E, dtype=wp.float32, device=device)
poissonRatio = wp.array(nu, dtype=wp.float32, device=device)
mpm.mu = wp.zeros(sim.nPoints, dtype=wp.float32, device=device)
mpm.lam = wp.zeros(sim.nPoints, dtype=wp.float32, device=device)
mpm.bulk = wp.zeros(sim.nPoints, dtype=wp.float32, device=device)
wp.launch(kernel=mpmRoutines.compute_mu_lam_bulk_from_E_nu,
          dim=sim.nPoints,
          inputs=[yMod, poissonRatio],
          outputs=[mpm.mu, mpm.lam, mpm.bulk],
          device=device)

# Assign material properties to MPM state
mpm.alpha = wp.array(alpha_array, dtype=wp.float32, device=device)
mpm.strainCriteria = wp.array(strainCriteria_array, dtype=wp.float32, device=device)
mpm.ys_base = wp.array(ys.copy(), dtype=wp.float32, device=device)
mpm.ys = wp.array(ys, dtype=wp.float32, device=device)
mpm.hardening = wp.array(hardening, dtype=wp.float32, device=device)
mpm.softening = wp.array(softening, dtype=wp.float32, device=device)
mpm.eta_shear = wp.array(eta_shear, dtype=wp.float32, device=device)
mpm.eta_bulk = wp.array(eta_bulk, dtype=wp.float32, device=device)

# Assign shared particle arrays to sim state
sim.materialLabel = wp.array(materialLabel, dtype=wp.int32, device=device)
sim.activeLabel = wp.array(activeLabel, dtype=wp.int32, device=device)
sim.particle_density = wp.array(density, dtype=wp.float32, device=device)
sim.particle_x = wp.array(x, dtype=wp.vec3, device=device)
sim.particle_v = wp.zeros(shape=sim.nPoints, dtype=wp.vec3, device=device)
sim.particle_vol = wp.array(particle_volume, dtype=float, device=device)
sim.particle_mass = wp.zeros(shape=sim.nPoints, dtype=float, device=device)
sim.particleBCMask = wp.zeros(shape=sim.nPoints, dtype=wp.int32, device=device)

wp.launch(kernel=mpmRoutines.get_float_array_product,
          dim=sim.nPoints,
          inputs=[sim.particle_density, sim.particle_vol, sim.particle_mass],
          device=device)

# ========== CFL Condition Estimation ==========
print(f"\n{'='*60}")
print("CFL CONDITION ANALYSIS")
print(f"{'='*60}")

# Compute wave speeds for stability analysis
E_mean = np.mean(E)
nu_mean = np.mean(nu)
density_mean = np.mean(density)

# Bulk modulus K and Shear modulus G
K_mean = E_mean / (3 * (1 - 2*nu_mean))
G_mean = E_mean / (2 * (1 + nu_mean))

# P-wave speed (longitudinal wave, fastest)
c_p = np.sqrt((K_mean + 4*G_mean/3) / density_mean)

# S-wave speed (shear wave)
c_s = np.sqrt(G_mean / density_mean)

# CFL condition: dt < dx / c_max
# For explicit time integration: dt < C * dx / c_p, where C ≈ 0.1-0.5 for MPM
cfl_limit = dx / c_p
cfl_number = args.dt * c_p / dx

print(f"Material Properties (mean):")
print(f"  E = {E_mean/1e6:.2f} MPa")
print(f"  ν = {nu_mean:.3f}")
print(f"  ρ = {density_mean:.1f} kg/m³")
print(f"  K = {K_mean/1e6:.2f} MPa (bulk modulus)")
print(f"  G = {G_mean/1e6:.2f} MPa (shear modulus)")

print(f"\nWave Speeds:")
print(f"  P-wave (longitudinal): c_p = {c_p:.1f} m/s")
print(f"  S-wave (shear):        c_s = {c_s:.1f} m/s")

print(f"\nGrid & Time Step:")
print(f"  dx = {dx:.4f} m")
print(f"  dt = {args.dt:.2e} s")

print(f"\nCFL Analysis:")
print(f"  CFL limit:  dt < {cfl_limit:.2e} s  (for C=1.0)")
print(f"  CFL number: {cfl_number:.3f}")

# Safety factor recommendations
safety_factors = {
    0.5: "Highly stable (conservative)",
    0.3: "Recommended for MPM",
    0.2: "Safe",
    0.1: "Very conservative"
}

print(f"\n  Safety Factor Recommendations:")
for factor, desc in safety_factors.items():
    recommended_dt = factor * cfl_limit
    if args.dt <= recommended_dt:
        status = "✅"
    else:
        status = "⚠️"
    print(f"    {status} C={factor}: dt < {recommended_dt:.2e} s  ({desc})")

if cfl_number < 0.1:
    print(f"\n  ✅ EXCELLENT: Very stable (CFL = {cfl_number:.3f} << 0.5)")
elif cfl_number < 0.3:
    print(f"\n  ✅ GOOD: Stable for MPM (CFL = {cfl_number:.3f} < 0.5)")
elif cfl_number < 0.5:
    print(f"\n  ⚠️  MARGINAL: Near stability limit (CFL = {cfl_number:.3f})")
elif cfl_number < 1.0:
    print(f"\n  ⚠️  RISKY: May be unstable (CFL = {cfl_number:.3f} > 0.5)")
else:
    print(f"\n  ❌ UNSTABLE: CFL condition violated (CFL = {cfl_number:.3f} > 1.0)!")
    print(f"     Simulation will likely diverge!")

# Time to traverse one grid cell
traverse_time = dx / c_p
steps_per_traverse = traverse_time / args.dt

print(f"\nTime Scales:")
print(f"  Time for wave to cross one cell: {traverse_time:.2e} s")
print(f"  Timesteps per cell crossing:     {steps_per_traverse:.1f} steps")

# Estimate based on domain size
domain_diagonal = np.sqrt((maxBounds[0]-minBounds[0])**2 + 
                          (maxBounds[1]-minBounds[1])**2 + 
                          (maxBounds[2]-minBounds[2])**2)
wave_crossing_time = domain_diagonal / c_p
steps_for_wave_crossing = wave_crossing_time / args.dt

print(f"  Domain diagonal:                 {domain_diagonal:.1f} m")
print(f"  Wave crossing time (diagonal):   {wave_crossing_time:.3f} s")
print(f"  Timesteps to cross domain:       {steps_for_wave_crossing:.0f} steps")

print(f"{'='*60}\n")

# Compute z_top for geostatic stress if not provided
if args.z_top is None:
    z_top = float(np.max(x[:, 2]))  # Use max z-coordinate of particles
else:
    z_top = args.z_top

# === Initialize geostatic deformation gradient ===
# Instead of initializing stress (which gets overwritten), we initialize F
# to represent the prestressed state. This way, elastic stress from F will
# automatically include geostatic compression.
g_mag = np.abs(args.gravity)  # Magnitude of gravity
if args.initialise_geostatic:
    wp.launch(
        kernel=mpmRoutines.initialize_geostatic_F,
        dim=sim.nPoints,
        inputs=[sim.particle_x, mpm.particle_F, mpm.mu, mpm.lam, sim.particle_density, g_mag, z_top, mpm.K0],
        device=device
    )
else:
    # set F to identity
    wp.launch(
        kernel=mpmRoutines.set_mat33_to_identity,
        dim=sim.nPoints,
        inputs=[mpm.particle_F],
        device=device
    )

# Copy F to F_trial for first step
wp.launch(kernel=mpmRoutines.set_mat33_to_copy,
          dim=sim.nPoints,
          inputs=[mpm.particle_F, mpm.particle_F_trial],
          device=device)

# Initialize damage (can add random perturbation if needed)
damagetemp = np.zeros(sim.nPoints, dtype=np.float32)
np.random.seed(0)  # for reproducibility
damagetemp = np.random.rand(sim.nPoints).astype(np.float32) * 0  # random damage between 0 and 0.1
mpm.particle_damage = wp.array(damagetemp, dtype=float, device=device)

# Compute particle radius
particle_radius = np.array(sim.particle_vol.numpy()**(1/3)*np.pi/6)
sim.particle_radius = wp.array(particle_radius, dtype=float, device=device)

# Initialize XPBD swelling radii
xpbd.particleBaseRadius = wp.array(particle_radius, dtype=float, device=device)
particleMaxRadius = particle_radius * (1 + xpbd.swellingRatio) ** (1 / 3)
xpbd.particleMaxRadius = wp.array(particleMaxRadius, dtype=float, device=device)

# ========== Compute XPBD Boundary Bounds ==========
# XPBD bounds should be INSIDE the MPM grid bounds to ensure particles stay within the grid
# This prevents XPBD particles from escaping the MPM domain
xpbd.max_radius = np.max(particleMaxRadius)

# XPBD boundary padding: ensure particles with max radius don't touch grid boundary cells
xpbd_padding_distance = -xpbd.max_radius*1.5 - (mpm.boundaryPadding-1) * sim.dx

xpbd.minBoundsXPBD = wp.vec3(
    sim.minBounds[0] - xpbd_padding_distance, 
    sim.minBounds[1] - xpbd_padding_distance, 
    sim.minBounds[2] - xpbd_padding_distance
)
xpbd.maxBoundsXPBD = wp.vec3(
    sim.maxBounds[0] + xpbd_padding_distance, 
    sim.maxBounds[1] + xpbd_padding_distance, 
    sim.maxBounds[2] + xpbd_padding_distance
)

print(f'MPM grid boundary padding: {mpm.boundaryPadding} cells = {mpm.boundaryPadding * sim.dx:.3f}m')
print(f'XPBD boundary padding: {xpbd_padding_distance:.3f}m')
print(f'XPBD domain: X=[{xpbd.minBoundsXPBD[0]:.2f}, {xpbd.maxBoundsXPBD[0]:.2f}], '
      f'Y=[{xpbd.minBoundsXPBD[1]:.2f}, {xpbd.maxBoundsXPBD[1]:.2f}], '
      f'Z=[{xpbd.minBoundsXPBD[2]:.2f}, {xpbd.maxBoundsXPBD[2]:.2f}]')

# ========== Renderer Initialization ==========
if sim.render:
    if args.render_backend == "usd":
        # USD renderer - GPU-direct rendering with zero CPU transfers
        from utils import usdRenderer
        usd_output_path = f"{sim.outputFolder}/usd"
        renderer = usdRenderer.WarpUSDRenderer(
            output_path=usd_output_path,
            fps=60,
            up_axis="Z"
        )
        print(f"Initialized USD renderer: {usd_output_path}")
        maxStress = 0.0
        maxStrain = 0.0
        
    elif args.render_backend == "opengl":
        # OpenGL renderer - interactive real-time viewer
        renderer = fs5RendererCore.OpenGLRenderer(        
            title=f"MPM",
            scaling=1.0,
            fps=60,
            up_axis="z",
            screen_width=1024,
            screen_height=768,
            near_plane=0.01,
            far_plane=10000,
            camera_fov=75.0,
            background_color=(0,0,0),
            draw_grid=True,
            draw_sky=False,
            draw_axis=True,
            show_info=True,
            render_wireframe=False,
            axis_scale=1.0,
            vsync=False,
            headless=False,
            enable_backface_culling=True
        )
        renderer._camera_speed = 0.5
        # orient the renderer to look at the centroid of the particles
        renderer = fs5PlotUtils.look_at_centroid(x, renderer, renderer.camera_fov)
        maxStress = 0.0
        maxStrain = 0.0
    else:
        raise ValueError(f"Unknown render_backend: {args.render_backend}")

# ========== Main Simulation Loop ==========
startTime=time.time()
nextRenderTime = args.render_interval  # Next simulation time to render/save
for bigStep in range(0, sim.bigSteps):
    print(f"Starting big step {bigStep+1} of {sim.bigSteps}, meanys: {np.mean(mpm.ys.numpy()):.2f}")

    counter=0
    residualCPU=1e10
    minSteps = 1000  # minimum steps to ensure initial conditions are met
    while counter < sim.nSteps and (counter < minSteps or residualCPU > sim.residualThreshold):
        stepStartTime = time.time()
        sim.residual.zero_()  # reset residual at each step
        sim.numActiveParticles.zero_()  # reset active particle count at each step
        # if counter > 10000:
        #     sim.gravity = wp.vec3(0.0, 0.0, 0.0)  # disable gravity after 10000 steps to allow settling
        # perform the mpm simulation step
        simulationRoutines.mpmSimulationStep(sim, mpm, xpbd, device)

        # perform the xpbd step if timestep has reached the threshold
        if np.mod(counter, sim.mpmStepsPerXpbdStep) == 0 and counter>0:

            simulationRoutines.xpbdSimulationStep(sim, mpm, xpbd, device)

        # utility kernels to compute residual and count active particles and other stuff
        wp.launch(
            kernel=simulationRoutines.velocityConvergence,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                sim.particle_v,
                sim.particle_radius,
                sim.residual
            ],
            device=device
        )

        wp.launch(
            kernel=simulationRoutines.countActiveParticles,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.numActiveParticles
            ],
            device=device
        )

        residualCPU = sim.residual.numpy()[0]/sim.numActiveParticles.numpy()[0]
        counter=counter+1
        sim.t += sim.dt
        
        # Print status every 100 steps
        if np.mod(counter,100)==0:
            print(f'Step: {counter}, simulationTime: {sim.t:.4f}s, deltaTime: +{time.time()-stepStartTime:.4f}s, realTime: {time.time()-startTime:.4f}s, residual: {residualCPU:.4e}, mean accumulated plastic strain: {np.mean(mpm.particle_accumulated_strain.numpy()):.4f}, active particles: {sim.numActiveParticles.numpy()[0]}')
        
        # Check if we should render/save this timestep (use small tolerance for floating point comparison)
        shouldRenderSave = sim.t >= (nextRenderTime - 0.5 * sim.dt)
        
        # Render based on simulation time interval
        if sim.render and shouldRenderSave:
            if args.render_backend == "usd":
                from utils import usdRenderer
                maxStress = usdRenderer.render_mpm_usd(renderer, sim, mpm, bigStep, counter, maxStress=maxStress)
            else:  # opengl
                maxStress = fs5PlotUtils.render_mpm(renderer, sim, mpm, bigStep, counter, maxStress=maxStress)
        
        # Save based on simulation time interval (use same timing as render)
        if sim.saveFlag and shouldRenderSave:
            fs5PlotUtils.save_mpm(sim, mpm, bigStep, counter)
        
        # Update next render time after save/render
        if shouldRenderSave:
            nextRenderTime += args.render_interval
    # when convergence is reached, reduce the yield stress by 10% to mimic creep over a long time TODO: the creep should be a function of damage in some spatially varying way

    # wp.launch(
    #     kernel=mpmRoutines.creep_by_damage_with_baseline,
    #     dim=sim.nPoints,
    #     inputs=[
    #         sim.materialLabel,
    #         mpm.particle_damage,
    #         mpm.ys,
    #         mpm.ys_base,
    #         counter*sim.dt,         # or sim.dt * sim.mpmStepsPerXpbdStep
    #         0.0,           # base creep rate A_base (undamaged)
    #         0.0,           # damage-based creep A_damage
    #         1.5             # damage exponent beta
    #     ],
    #     device=device
    # )

# ========== Finalize Renderer ==========
if sim.render and args.render_backend == "usd":
    renderer.finalize()
    print(f"USD rendering complete. View with: usdview {renderer.usd_file}")

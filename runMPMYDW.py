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
# x = x + np.random.rand(x.shape[0], x.shape[1])*0.5 # jitter to prevent stress chains
# delete the middle 20% of particles in z
# x = x[~((x[:, 2] > np.percentile(x[:, 2], 40)) & (x[:, 2] < np.percentile(x[:, 2], 60)))]
nPoints = x.shape[0]
print(f"Number of particles: {nPoints}")

# ========== Compute Domain Bounds ==========
# Compute initial bounds from particle positions with padding
particle_min = np.min(x, 0)
particle_max = np.max(x, 0)
mpmPadding = 2

particleDiameter = np.mean(particle_volume) ** 0.33
dx = particleDiameter * args.grid_particle_spacing_scale
invdx = 1.0 / dx

# Apply uniform padding in X and Y, extra padding in Z (for gravity direction)
minBounds = np.array([
    particle_min[0] - args.grid_padding*dx,
    particle_min[1] - args.grid_padding*dx,
    particle_min[2] - mpmPadding*dx # Extra padding below for stability
])

maxBounds = np.array([
    particle_max[0] + args.grid_padding*dx,
    particle_max[1] + args.grid_padding*dx,
    particle_max[2] + args.grid_padding * 2.0 * dx  # Extra padding above for falling particles
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
wp.launch(
    kernel=mpmRoutines.initialize_geostatic_F,
    dim=sim.nPoints,
    inputs=[sim.particle_x, mpm.particle_F, mpm.mu, mpm.lam, sim.particle_density, g_mag, z_top, mpm.K0],
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
    # initialise renderer
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
    enable_backface_culling=True)
    renderer._camera_speed = 0.5
    # orient the renderer to look at the centroid of the particles
    renderer=fs5PlotUtils.look_at_centroid(x,renderer,renderer.camera_fov)
    maxStress=0.0
    maxStrain=0.0

# ========== Main Simulation Loop ==========
startTime=time.time()
for bigStep in range(0, sim.bigSteps):
    print(f"Starting big step {bigStep+1} of {sim.bigSteps}, meanys: {np.mean(mpm.ys.numpy()):.2f}")

    counter=0
    residualCPU=1e10
    minSteps = 1000  # minimum steps to ensure initial conditions are met
    while counter < sim.nSteps and (counter < minSteps or residualCPU > sim.residualThreshold):
        stepStartTime = time.time()
        sim.residual.zero_()  # reset residual at each step
        sim.numActiveParticles.zero_()  # reset active particle count at each step
        if counter > 10000:
            sim.gravity = wp.vec3(0.0, 0.0, 0.0)  # disable gravity after 10000 steps to allow settling
        # perform the mpm simulation step
        simulationRoutines.mpmSimulationStep(sim, mpm, xpbd, device)

        # perform the xpbd step if timestep has reached the threshold
        if np.mod(counter, sim.mpmStepsPerXpbdStep) == 0 and counter>0:

            simulationRoutines.xpbdSimulationStep(sim, mpm, xpbd, device)

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
        sim.t += sim.dt
        counter=counter+1
        if np.mod(counter,100)==0:
            print(f'Step: {counter}, simulationTime: {sim.t:.4f}s, deltaTime: +{time.time()-stepStartTime:.4f}s, realTime: {time.time()-startTime:.4f}s, residual: {residualCPU:.4e}, mean accumulated plastic strain: {np.mean(mpm.particle_accumulated_strain.numpy()):.4f}, active particles: {sim.numActiveParticles.numpy()[0]}')
            
            if sim.render:
                maxStress = fs5PlotUtils.render_mpm(renderer, sim, mpm, bigStep, counter, maxStress=maxStress)
            if sim.saveFlag:
                fs5PlotUtils.save_mpm(sim, mpm, bigStep, counter)
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

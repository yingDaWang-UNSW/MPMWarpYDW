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
import numpy as np
np.seterr(over='raise')
from utils.getArgs import get_args
args = get_args()

# Example usage:
dt = args.dt
dtxpbd = args.dtxpbd
mpmStepsPerXpbdStep = int(dtxpbd / dt)
nSteps = args.nSteps
bigSteps = args.bigSteps
residualThreshold = args.residualThreshold

print(f"Running simulation with dt={dt}, dtxpbd={dtxpbd}, nSteps={nSteps}, bigSteps={bigSteps}, residualThreshold={residualThreshold}")

# --- Simulation parameters ---
dt = args.dt
dtxpbd = args.dtxpbd
mpmStepsPerXpbdStep = int(dtxpbd / dt)
nSteps = args.nSteps
bigSteps = args.bigSteps
residualThreshold = args.residualThreshold

rpic_damping = args.rpic_damping
grid_v_damping_scale = args.grid_v_damping_scale
update_cov = args.update_cov

render = args.render
saveFlag = args.saveFlag

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

minBounds = np.min(x, 0) - args.grid_padding
maxBounds = np.max(x, 0) + args.grid_padding
minBounds[2] = np.min(x, 0)[2] - 6
maxBounds[2] = np.max(x, 0)[2] + 2*args.grid_padding  # ensure z bounds are larger to avoid particles going out of bounds
particleDiameter = np.mean(particle_volume) ** 0.33
dx = particleDiameter * args.grid_particle_spacing_scale
invdx = 1.0 / dx
gridDims = np.ceil((maxBounds - minBounds) / dx).astype(int)
print(f'GRID: {gridDims}')

x_vals = np.linspace(minBounds[0] + 0.5 * dx, maxBounds[0] - 0.5 * dx, gridDims[0])
y_vals = np.linspace(minBounds[1] + 0.5 * dx, maxBounds[1] - 0.5 * dx, gridDims[1])
z_vals = np.linspace(minBounds[2] + 0.5 * dx, maxBounds[2] - 0.5 * dx, gridDims[2])
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
centroids = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
print(f"Total grid centroids: {centroids.shape[0]}")

# --- Material properties ---
# Load from HDF5 if available, otherwise use args/defaults
if spatial_properties_available.get('density', False):
    density = np.array(h5file["density"], dtype=np.float32)
    print(f"  Loaded density from HDF5: {density.min():.2e} to {density.max():.2e} kg/m³")
else:
    density = np.full(nPoints, args.density, dtype=np.float32)

if spatial_properties_available.get('E', False):
    E = np.array(h5file["E"], dtype=np.float32)
    print(f"  Loaded E from HDF5: {E.min():.2e} to {E.max():.2e} Pa")
else:
    E = np.full(nPoints, args.E, dtype=np.float32)

if spatial_properties_available.get('nu', False):
    nu = np.array(h5file["nu"], dtype=np.float32)
    print(f"  Loaded nu from HDF5: {nu.min():.3f} to {nu.max():.3f}")
else:
    nu = np.full(nPoints, args.nu, dtype=np.float32)

if spatial_properties_available.get('ys', False):
    ys = np.array(h5file["ys"], dtype=np.float32)
    print(f"  Loaded ys from HDF5: {ys.min():.2e} to {ys.max():.2e} Pa")
else:
    ys = np.full(nPoints, args.ys, dtype=np.float32)
    # Optional: Apply spatial modifications here if not using HDF5
    # ys = ys * (1 - (x[:, 2] - minBounds[2]) / (maxBounds[2] - minBounds[2]))**3

if spatial_properties_available.get('hardening', False):
    hardening = np.array(h5file["hardening"], dtype=np.float32)
    print(f"  Loaded hardening from HDF5: {hardening.min():.3f} to {hardening.max():.3f}")
else:
    hardening = np.full(nPoints, args.hardening, dtype=np.float32)

if spatial_properties_available.get('softening', False):
    softening = np.array(h5file["softening"], dtype=np.float32)
    print(f"  Loaded softening from HDF5: {softening.min():.3f} to {softening.max():.3f}")
else:
    softening = np.full(nPoints, args.softening, dtype=np.float32)

if spatial_properties_available.get('eta_shear', False):
    eta_shear = np.array(h5file["eta_shear"], dtype=np.float32)
    print(f"  Loaded eta_shear from HDF5: {eta_shear.min():.2e} to {eta_shear.max():.2e} Pa·s")
else:
    eta_shear = np.full(nPoints, args.eta_shear, dtype=np.float32)

if spatial_properties_available.get('eta_bulk', False):
    eta_bulk = np.array(h5file["eta_bulk"], dtype=np.float32)
    print(f"  Loaded eta_bulk from HDF5: {eta_bulk.min():.2e} to {eta_bulk.max():.2e} Pa·s")
else:
    eta_bulk = np.full(nPoints, args.eta_bulk, dtype=np.float32)

materialLabel = np.ones(nPoints, dtype=np.int32)

activeLabel = np.ones(nPoints, dtype=np.int32)

# --- Constitutive model selection and parameters ---
constitutive_model = args.constitutive_model

# Load alpha (Drucker-Prager) from HDF5 if available (before closing file)
if spatial_properties_available.get('alpha', False):
    alpha_array = np.array(h5file["alpha"], dtype=np.float32)
    print(f"  Loaded alpha from HDF5: {alpha_array.min():.3f} to {alpha_array.max():.3f}")
else:
    alpha_array = np.full(nPoints, args.alpha, dtype=np.float32)

# Load strainCriteria from HDF5 if available (before closing file)
if spatial_properties_available.get('strainCriteria', False):
    strainCriteria_array = np.array(h5file["strainCriteria"], dtype=np.float32)
    print(f"  Loaded strainCriteria from HDF5: {strainCriteria_array.min():.3f} to {strainCriteria_array.max():.3f}")
else:
    strainCriteria_array = np.full(nPoints, args.strainCriteria, dtype=np.float32)

# Close HDF5 file after loading all properties
h5file.close()

K0 = args.K0

# --- Gravity, boundary, coupling ---
gravity = wp.vec3(0.0, 0.0, args.gravity)
boundFriction = args.boundFriction
eff = args.eff
strainCriteria = wp.array(strainCriteria_array, dtype=wp.float32, device=device)

# --- XPBD Contact Threshold (prevents XPBD "pulling" on MPM when separating) ---
# -1e20: disabled (original behavior)
# 0.0: only compression (XPBD only contributes when approaching)
# >0.0: allow small separation velocity threshold

# --- Transfer to GPU ---
yMod = wp.array(E, dtype=wp.float32, device=device)
poissonRatio = wp.array(nu, dtype=wp.float32, device=device)
mu = wp.zeros(nPoints, dtype=wp.float32, device=device)
lam = wp.zeros(nPoints, dtype=wp.float32, device=device)
bulk = wp.zeros(nPoints, dtype=wp.float32, device=device)
wp.launch(kernel=mpmRoutines.compute_mu_lam_bulk_from_E_nu,
          dim=nPoints,
          inputs=[yMod, poissonRatio],
          outputs=[mu, lam, bulk],
          device=device)

# Create Drucker-Prager parameter arrays (use alpha_array from HDF5 or defaults)
alpha = wp.array(alpha_array, dtype=wp.float32, device=device)

ys_base = ys.copy()  # keep a copy of the base yield stress for creep calculations`
ys_base = wp.array(ys_base, dtype=wp.float32, device=device)
ys = wp.array(ys, dtype=wp.float32, device=device)
materialLabel = wp.array(materialLabel, dtype=wp.int32, device=device)
activeLabel = wp.array(activeLabel, dtype=wp.int32, device=device)
particle_density = wp.array(density, dtype=wp.float32, device=device)
hardening = wp.array(hardening, dtype=wp.float32, device=device)
softening = wp.array(softening, dtype=wp.float32, device=device)
eta_shear = wp.array(eta_shear, dtype=wp.float32, device=device)
eta_bulk = wp.array(eta_bulk, dtype=wp.float32, device=device)

# --- Dynamic material point parameters ---
particle_x = wp.array(x, dtype=wp.vec3)
particle_x_initial = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_vol = wp.array(particle_volume, dtype=float)
particle_mass = wp.zeros(shape=nPoints, dtype=float)
wp.launch(kernel=mpmRoutines.get_float_array_product,
          dim=nPoints,
          inputs=[particle_density, particle_vol, particle_mass],
          device=device)

# Compute z_top for geostatic stress if not provided
if args.z_top is None:
    z_top = float(np.max(x[:, 2]))  # Use max z-coordinate of particles
else:
    z_top = args.z_top

particle_v = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_F = wp.zeros(shape=nPoints, dtype=wp.mat33)
particle_F_trial = wp.zeros(shape=nPoints, dtype=wp.mat33)

# === Initialize geostatic deformation gradient ===
# Instead of initializing stress (which gets overwritten), we initialize F
# to represent the prestressed state. This way, elastic stress from F will
# automatically include geostatic compression.
g_mag = np.abs(args.gravity)  # Magnitude of gravity
# wp.launch(kernel=mpmRoutines.set_mat33_to_identity,
#           dim=nPoints,
#           inputs=[particle_F],
#           device=device)
wp.launch(
    kernel=mpmRoutines.initialize_geostatic_F,
    dim=nPoints,
    inputs=[particle_x, particle_F, mu, lam, particle_density, g_mag, z_top, K0],
    device=device
)

# # Check if geostatic stress exceeds yield stress
# # Compute representative stress at mid-depth
# mid_depth = (z_top - np.min(x[:, 2])) / 2.0
# sigma_v_mid = args.density * g_mag * mid_depth
# sigma_h_mid = K0 * sigma_v_mid

# # Compute vertical pressure gradient
# domain_height = z_top - np.min(x[:, 2])
# sigma_v_bottom = args.density * g_mag * domain_height
# sigma_v_top = 0.0  # No overburden at surface
# vertical_gradient = (sigma_v_bottom - sigma_v_top) / domain_height if domain_height > 0 else 0.0

# # Von Mises equivalent for geostatic state: sqrt(3/2 * |s|^2) where s is deviatoric
# # For σ_h = σ_h, σ_v: mean = (2σ_h + σ_v)/3, deviatoric: s_h = σ_h - mean, s_v = σ_v - mean
# mean_stress = (2*sigma_h_mid + sigma_v_mid) / 3.0
# s_h = sigma_h_mid - mean_stress
# s_v = sigma_v_mid - mean_stress
# von_mises_geostatic = np.sqrt(1.5 * (2*s_h**2 + s_v**2))

# print(f"Initialized geostatic deformation gradient (z_top={z_top:.2f}, K0={K0:.2f})")
# print(f"  Domain height: {domain_height:.2f} m")
# print(f"  Vertical pressure gradient: dσ_v/dz = {vertical_gradient:.2e} Pa/m (= ρg = {args.density * g_mag:.2e} Pa/m)")
# print(f"  Stress at bottom: σ_v={sigma_v_bottom:.2e} Pa, σ_h={K0*sigma_v_bottom:.2e} Pa")
# print(f"  Mid-depth stress: σ_v={sigma_v_mid:.2e} Pa, σ_h={sigma_h_mid:.2e} Pa")
# print(f"  Von Mises equivalent (mid-depth): {von_mises_geostatic:.2e} Pa")
# print(f"  Yield stress: {args.ys:.2e} Pa")
# if von_mises_geostatic > args.ys:
#     print(f"     WARNING: Geostatic stress ({von_mises_geostatic:.2e}) > Yield stress ({args.ys:.2e})")
#     print(f"     Material will plastically yield! Increase yield stress or reduce K0.")

# Copy F to F_trial for first step
wp.launch(kernel=mpmRoutines.set_mat33_to_copy,
          dim=nPoints,
          inputs=[particle_F, particle_F_trial],
          device=device)

particle_stress = wp.zeros(shape=nPoints, dtype=wp.mat33)

particle_accumulated_strain = wp.zeros(shape=nPoints, dtype=float, device=device)
particle_damage = wp.zeros(shape=nPoints, dtype=float, device=device)

damagetemp=np.zeros(nPoints, dtype=np.float32)
# make random damage
np.random.seed(0)  # for reproducibility
damagetemp = np.random.rand(nPoints).astype(np.float32) * 0  # random damage between 0 and 0.1

particle_damage = wp.array(damagetemp, dtype=float, device=device)

particle_C = wp.zeros(shape=nPoints, dtype=wp.mat33)
particle_init_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)
particle_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)
particle_radius = np.array(particle_vol.numpy()**(1/3)*np.pi/6)
# particle_radius[materialLabel.numpy() == 2] = 0.5 * particle_radius[materialLabel.numpy() == 2]
# particle_radius[materialLabel.numpy() == 1] = particle_radius[materialLabel.numpy() == 1]
particle_radius = wp.array(particle_radius, dtype=float, device=device)

# --- Grid arrays ---
grid_m = wp.zeros(shape=gridDims, dtype=float, device=device)
grid_v_in = wp.zeros(shape=gridDims, dtype=wp.vec3, device=device)
grid_v_out = wp.zeros(shape=gridDims, dtype=wp.vec3, device=device)
minBounds = wp.vec3(*minBounds)
maxBounds = wp.vec3(*maxBounds)

# --- XPBD parameters ---
particle_x_initial_xpbd = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_x_initial_xpbd.fill_([1e6, 1e6, 1e6])
particle_v_initial_xpbd = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_cumDist_xpbd = np.zeros(shape=nPoints, dtype=float)
# particle_cumDist_xpbd[materialLabel.numpy() == 2] = 0.0
particle_cumDist_xpbd = wp.array(particle_cumDist_xpbd, dtype=float, device=device)

# particles already material type 2 have large arbitrary distance
particle_distance_total = wp.zeros(shape=nPoints, dtype=float)

particle_x_integrated = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_v_integrated = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_x_deltaInt = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_v_deltaInt = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_delta = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_grid = wp.HashGrid(128, 128, 128)

xpbd_relaxation = args.xpbd_relaxation
dynamicParticleFriction = args.dynamicParticleFriction
staticVelocityThreshold = args.staticVelocityThreshold
staticParticleFriction = args.staticParticleFriction
xpbd_iterations = args.xpbd_iterations
particle_cohesion = args.particle_cohesion
sleepThreshold = args.sleepThreshold

# --- Swelling ---
swellingRatio = args.swellingRatio
swellingActivationFactor = args.swellingActivationFactor
swellingMaxFactor = args.swellingMaxFactor
particleBaseRadius = particle_radius.numpy()
particleMaxRadius = particleBaseRadius * (1 + swellingRatio) ** (1 / 3)
particleBaseRadius = wp.array(particleBaseRadius)
particleMaxRadius = wp.array(particleMaxRadius)

# --- Velocity limits & bounds ---
particle_v_max = args.particle_v_max
max_radius = np.max(particleMaxRadius.numpy())
padding = 3 - max_radius
minBoundsXPBD = wp.vec3(minBounds[0] + padding * dx, minBounds[1] + padding * dx, minBounds[2] + padding * dx)
maxBoundsXPBD = wp.vec3(maxBounds[0] - padding * dx, maxBounds[1] - padding * dx, maxBounds[2] - padding * dx)

# xpbdParticleCount = np.sum(materialLabel.numpy() == 2) # count of xpbd particles, i.e. particles with material label 2
#BOUNDARY CONDITIONS (DO THIS LATER)###############################################################################################################################

if render:
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

# compute the geostatic stress for yz calculation - do not apply in simulation - material is 

# ys = np.full(nPoints, 1e10, dtype=np.float32)  # set yield stress to a very high value to simulate geostatic conditions
# ys_base = ys.copy()  # keep a copy of the base yield stress for creep calculations
# ys_base = wp.array(ys_base, dtype=wp.float32, device=device)
# ys = wp.array(ys, dtype=wp.float32, device=device)




residual = wp.array([1e10], dtype=float, device=device)  # residual for velocity convergence
numActiveParticles = wp.array([0], dtype=wp.int32, device=device)  # number of active particles

t=0
startTime=time.time()
for bigStep in range(0, bigSteps):
    print(f"Starting big step {bigStep+1} of {bigSteps}, meanys: {np.mean(ys.numpy()):.2f}")

    counter=0
    residualCPU=1e10
    minSteps = 1000  # minimum steps to ensure initial conditions are met
    while counter < nSteps and (counter < minSteps or residualCPU > residualThreshold):
        stepStartTime = time.time()
        residual.zero_()  # reset residual at each step
        numActiveParticles.zero_()  # reset active particle count at each step
        if counter > 10000:
            gravity = wp.vec3(0.0, 0.0, 0.0)  # disable gravity after 10000 steps to allow settling
        # perform the mpm simulation step
        simulationRoutines.mpmSimulationStep(
            particle_x,
            particle_v,
            particle_x_initial_xpbd,
            particle_v_initial_xpbd,
            particle_F,
            particle_F_trial,
            particle_stress,
            particle_accumulated_strain,
            particle_damage,
            particle_C,
            particle_vol,
            particle_mass,
            particle_density,
            particle_cov,
            activeLabel,
            materialLabel,
            mu,
            lam,
            ys,
            hardening,
            softening,
            strainCriteria,
            eta_shear,
            eta_bulk,
            eff,
            dx,
            invdx,
            minBounds,
            rpic_damping,
            dt,
            gravity,
            grid_m,
            grid_v_in,
            grid_v_out,
            gridDims,
            grid_v_damping_scale,
            boundFriction,
            update_cov,
            nPoints,
            device,
            constitutive_model,
            alpha,
        )
        # perform the xpbd step if timestep has reached the threshold
        if np.mod(counter, mpmStepsPerXpbdStep) == 0 and counter>0:
            
            simulationRoutines.xpbdSimulationStep(
                particle_grid,
                particle_x,
                particle_v,
                particle_x_integrated,
                particle_v_integrated,
                particle_delta,
                particle_v_deltaInt,
                particle_x_deltaInt,
                particle_mass,
                particle_radius,
                particle_cohesion,
                particle_v_max,
                particle_cumDist_xpbd,
                particle_v_initial_xpbd,
                particle_x_initial_xpbd,
                particleMaxRadius,
                particleBaseRadius,
                activeLabel,
                materialLabel,
                gravity,
                minBoundsXPBD,
                maxBoundsXPBD,
                dtxpbd,
                xpbd_iterations,
                xpbd_relaxation,
                dynamicParticleFriction,
                staticParticleFriction,
                staticVelocityThreshold,
                sleepThreshold,
                swellingRatio,
                swellingActivationFactor,
                swellingMaxFactor,
                max_radius,
                nPoints,
                dx,
                device,
            )

        wp.launch(
            kernel=simulationRoutines.velocityConvergence,
            dim=nPoints,
            inputs=[
                activeLabel,
                materialLabel,
                particle_v,
                particle_radius,
                residual
            ],
            device=device
        )

        wp.launch(
            kernel=simulationRoutines.countActiveParticles,
            dim=nPoints,
            inputs=[
                activeLabel,
                numActiveParticles
            ],
            device=device
        )

        residualCPU = residual.numpy()[0]/numActiveParticles.numpy()[0]
        t=t+dt
        counter=counter+1
        if np.mod(counter,100)==0:
            # # DIAGNOSTIC: Check velocities and positions
            # particle_v_cpu = particle_v.numpy()
            # particle_x_cpu = particle_x.numpy()
            # mpm_mask = materialLabel.numpy() == 1
            # xpbd_mask = materialLabel.numpy() == 2
            
            # if np.any(mpm_mask):
            #     mpm_vz = particle_v_cpu[mpm_mask, 2]
            #     mpm_z = particle_x_cpu[mpm_mask, 2]
            #     print(f'MPM:  mean vz={np.mean(mpm_vz):.3f} m/s, mean z={np.mean(mpm_z):.2f} m')
            
            # if np.any(xpbd_mask):
            #     xpbd_vz = particle_v_cpu[xpbd_mask, 2]
            #     xpbd_z = particle_x_cpu[xpbd_mask, 2]
            #     print(f'XPBD: mean vz={np.mean(xpbd_vz):.3f} m/s, mean z={np.mean(xpbd_z):.2f} m')
            
            print(f'Step: {counter}, simulationTime: {t:.4f}s, deltaTime: +{time.time()-stepStartTime:.4f}s, realTime: {time.time()-startTime:.4f}s, residual: {residualCPU:.4e}, mean accumulated plastic strain: {np.mean(particle_accumulated_strain.numpy()):.4f}, active particles: {numActiveParticles.numpy()[0]}')
            
            if render:
                maxStress = fs5PlotUtils.render_mpm(
                    renderer,
                    particle_x,
                    particle_v,
                    particle_radius,
                    ys,
                    ys_base,
                    alpha,
                    particle_damage,
                    particle_stress,
                    materialLabel,
                    grid_m,
                    minBounds,
                    dx,
                    bigStep,
                    counter,
                    nPoints,
                    saveFlag=saveFlag,
                    color_mode=args.color_mode,   # or "von_mises", "damage", "vz"
                    maxStress=maxStress
                )
            if saveFlag:
                fs5PlotUtils.save_mpm(
                    args.outputFolder,
                    particle_x,
                    particle_v,
                    particle_radius,
                    ys,
                    ys_base,
                    alpha,
                    particle_damage,
                    particle_stress,
                    materialLabel,
                    grid_m,
                    minBounds,
                    dx,
                    bigStep,
                    counter,
                    nPoints,
                    color_mode=args.color_mode,   # or "von_mises", "damage", "vz"
                )
    # when convergence is reached, reduce the yield stress by 10% to mimic creep over a long time TODO: the creep should be a function of damage in some spatially varying way 

    # wp.launch(
    #     kernel=mpmRoutines.creep_by_damage_with_baseline,
    #     dim=nPoints,
    #     inputs=[
    #         materialLabel,
    #         particle_damage,
    #         ys,
    #         ys_base,
    #         counter*dt,         # or dt * mpmStepsPerXpbdStep
    #         0.0,           # base creep rate A_base (undamaged)
    #         0.0,           # damage-based creep A_damage
    #         1.5             # damage exponent beta
    #     ],
    #     device=device
    # )

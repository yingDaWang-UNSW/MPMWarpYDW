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

print(f"Running simulation with dt={dt}, dtxpbd={dtxpbd}, nSteps={nSteps}")

# --- Simulation parameters ---
dt = args.dt
dtxpbd = args.dtxpbd
mpmStepsPerXpbdStep = int(dtxpbd / dt)
nSteps = args.nSteps

rpic_damping = args.rpic_damping
grid_v_damping_scale = args.grid_v_damping_scale
update_cov = args.update_cov

render = args.render
saveFlag = args.saveFlag

# --- Domain & grid ---
domainFile = args.domainFile
h5file = h5py.File(domainFile, "r")
x, particle_volume = h5file["x"], h5file["particle_volume"]
x = np.array(x).T
# x = x + np.random.rand(x.shape[0], x.shape[1])  # jitter to prevent stress chains
nPoints = x.shape[0]
print(f"Number of particles: {nPoints}")

minBounds = np.min(x, 0) - args.grid_padding
maxBounds = np.max(x, 0) + args.grid_padding
minBounds[2] = np.min(x, 0)[2] - 10
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
density = np.full(nPoints, args.density, dtype=np.float32)
E = np.full(nPoints, args.E, dtype=np.float32)
nu = np.full(nPoints, args.nu, dtype=np.float32)
ys = np.full(nPoints, args.ys, dtype=np.float32)
# make ys lower at the top of the domain quadratically
ys = ys * (1 - (x[:, 2] - minBounds[2]) / (maxBounds[2] - minBounds[2]))**3
# make ys at the bottom infinite
ys[x[:, 2] < minBounds[2] + 0.1 * (maxBounds[2] - minBounds[2])] = 1e10

hardening = np.full(nPoints, args.hardening, dtype=np.float32)
softening = np.full(nPoints, args.softening, dtype=np.float32)
eta_shear = np.full(nPoints, args.eta_shear, dtype=np.float32)
eta_bulk = np.full(nPoints, args.eta_bulk, dtype=np.float32)

materialLabel = np.ones(nPoints, dtype=np.int32)
activeLabel = np.ones(nPoints, dtype=np.int32)

# --- Gravity, boundary, coupling ---
gravity = wp.vec3(0.0, 0.0, args.gravity)
boundFriction = args.boundFriction
eff = args.eff
strainCriteria = wp.full(nPoints, args.strainCriteria, dtype=wp.float32, device=device)

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

particle_v = wp.zeros(shape=nPoints, dtype=wp.vec3)
particle_F = wp.zeros(shape=nPoints, dtype=wp.mat33)
particle_F_trial = wp.zeros(shape=nPoints, dtype=wp.mat33)
wp.launch(kernel=mpmRoutines.set_mat33_to_identity,
          dim=nPoints,
          inputs=[particle_F_trial],
          device=device)

particle_stress = wp.zeros(shape=nPoints, dtype=wp.mat33)

# z_top = maxBounds[2]  # top of domain
# K0 = args.K0
# wp.launch(
#     kernel=mpmRoutines.initialize_geostatic_stress,
#     dim=nPoints,
#     inputs=[particle_x, particle_stress, particle_density, abs(args.gravity), z_top, K0],
#     device=device
# )
# gravity = wp.vec3(0.0, 0.0, 0.0)

particle_accumulated_strain = wp.zeros(shape=nPoints, dtype=float, device=device)
particle_damage = wp.zeros(shape=nPoints, dtype=float, device=device)
particle_C = wp.zeros(shape=nPoints, dtype=wp.mat33)
particle_init_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)
particle_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)
particle_radius = wp.array(particle_vol.numpy()**(1/3)*np.pi/6*0.95)

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
particle_cumDist_xpbd = wp.zeros(shape=nPoints, dtype=float)
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

residual = wp.array([1e10], dtype=float, device=device)  # residual for velocity convergence
numActiveParticles = wp.array([0], dtype=wp.int32, device=device)  # number of active particles

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
t=0
startTime=time.time()
for bigStep in range(0, 100):
    print(f"Starting big step {bigStep+1} of 100, meanys: {np.mean(ys.numpy()):.2f}")

    counter=0
    residualCPU=1e10
    minSteps = 1000  # minimum steps to ensure initial conditions are met
    while counter < nSteps and (counter < minSteps or residualCPU > 5e-1):
        stepStartTime = time.time()
        residual.zero_()  # reset residual at each step
        numActiveParticles.zero_()  # reset active particle count at each step
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
            device
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
            print(f'Step: {counter}, simulationTime: {t:.4f}s, deltaTime: +{time.time()-stepStartTime:.4f}s, realTime: {time.time()-startTime:.4f}s, residual: {residualCPU:.4e}, mean accumulated plastic strain: {np.mean(particle_accumulated_strain.numpy()):.4f}, active particles: {numActiveParticles.numpy()[0]}')
            if render:
                renderer.begin_frame()
                # colors=fs5PlotUtils.values_to_rgb(np.arange(0,nPoints,1),min_val=0, max_val=nPoints)
                # colors=fs5PlotUtils.values_to_rgb(ys.numpy(),min_val=0.0, max_val=ys.numpy().max())
                # colors=fs5PlotUtils.values_to_rgb(particle_radius.numpy(),min_val=particleBaseRadius.numpy().min(), max_val=particleMaxRadius.numpy().max())
                # colors=fs5PlotUtils.values_to_rgb(particle_damage.numpy(),min_val=0.0, max_val=1.0)
                colors = fs5PlotUtils.values_to_rgb((ys.numpy() / ys_base.numpy()), min_val=0.0, max_val=1.0)

                # colors=fs5PlotUtils.values_to_rgb(particle_v.numpy()[:,2],min_val=particle_v.numpy()[:,2].min(), max_val=particle_v.numpy()[:,2].max())

                # x=particle_stress.numpy()
                # # x is your (N, 3, 3) array of stress tensors
                # sigma = x.astype(np.float64)  # promote to float64 if needed

                # # Compute mean (hydrostatic) stress for each tensor
                # mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0  # shape (N,)

                # # Subtract mean stress from diagonal elements to get deviatoric tensor
                # identity = np.eye(3)
                # s = sigma# - mean_stress[:, None, None] * identity  # broadcasted subtraction

                # # Compute von Mises stress
                # von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1, 2)))  # shape (N,)
                # maxStress=np.max([np.max(von_mises),maxStress])
                # colors=fs5PlotUtils.values_to_rgb(von_mises,min_val=0.0, max_val=maxStress)

                renderer.render_points(points=particle_x, name="points", radius=particle_radius.numpy(), colors=colors, dynamic=True)
                # renderer.render_box(name='simBounds',pos=[grid_lim/2,grid_lim/2,grid_lim/2],extents=[grid_lim/2,grid_lim/2,grid_lim/2],rot=[0,0,0,1])
                renderer.end_frame()
                renderer.update_view_matrix()

                # engine_utils.save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=1, save_to_h5=0)
                # time.sleep(1)
            if saveFlag:
                # save points and fields for visualization
                fs5PlotUtils.save_grid_and_particles_vti_vtp(
                    output_prefix=f"./output/sim_step_{bigStep}_{counter:06d}",
                    grid_mass=grid_m.numpy(),              # shape (nx, ny, nz)
                    minBounds=minBounds,                         # (x0, y0, z0)
                    dx=dx,
                    particle_positions=particle_x.numpy(),          # shape (N, 3)
                    particle_radius=np.arange(0,nPoints,1)                         # for Point Gaussian
                )

    # when convergence is reached, reduce the yield stress by 10% to mimic creep over a long time TODO: the creep should be a function of damage in some spatially varying way 

    wp.launch(
        kernel=mpmRoutines.creep_by_damage_with_baseline,
        dim=nPoints,
        inputs=[
            materialLabel,
            particle_damage,
            ys,
            ys_base,
            counter*dt,         # or dt * mpmStepsPerXpbdStep
            0,           # base creep rate A_base (undamaged)
            2e-0,           # damage-based creep A_damage
            1.5             # damage exponent beta
        ],
        device=device
    )

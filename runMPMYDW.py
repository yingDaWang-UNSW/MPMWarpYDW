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
from utils import xpbdRoutines
import numpy as np
np.seterr(over='raise')

# simulation parameters
dt = 1e-3 #time step in seconds 
dtxpbd = 1e-2 #time step for xpbd in seconds
mpmStepsPerXpbdStep = int(dtxpbd/dt) #number of mpm steps per xpbd step
nSteps = 2500000 #number of simulation steps

rpic_damping = 0.0 #damping factor for the particle to grid transfer, 0.0 means no damping
grid_v_damping_scale = 1.1 #damping factor for the grid velocity, 1.0 or higher means no damping
update_cov = True #whether to update the covariance matrix during the grid to particle transfer

render = True #whether to render the simulation
saveFlag = True
#LOAD THE DATA AND INFER THE GRID##############################################################################################################

domainFile="./exampleDomains/annular_arch_particles.h5"
h5file = h5py.File(domainFile, "r")
x, particle_volume = h5file["x"], h5file["particle_volume"]
x=np.array(x).T
# x=x+np.random.rand(x.shape[0], x.shape[1]) # add some noise to the particle positions

nPoints= x.shape[0]
print(f"Number of particles: {nPoints}")

# infer the grid parameters from the particle positions
minBounds=np.min(x,0)-[25, 25, 25]
maxBounds=np.max(x,0)+[25, 25, 25]

particleDiameter = np.mean(particle_volume)**0.33

# Compute grid spacing (assume cubic)
dx = particleDiameter*2
invdx= 1.0 / dx

# Compute number of cells in each dimension (ensure int)
gridDims = np.ceil((maxBounds - minBounds) / dx).astype(int)

print(f'GRID: {gridDims}')

# Compute grid centroids
x_vals = np.linspace(minBounds[0] + 0.5*dx, maxBounds[0] - 0.5*dx, gridDims[0])
y_vals = np.linspace(minBounds[1] + 0.5*dx, maxBounds[1] - 0.5*dx, gridDims[1])
z_vals = np.linspace(minBounds[2] + 0.5*dx, maxBounds[2] - 0.5*dx, gridDims[2])

X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
centroids = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

print(f"Total grid centroids: {centroids.shape[0]}")

#PARAMETERS###############################################################################################################################

# point properties from CPU/file

density = 3000.0 #density of the material
density = np.full(nPoints, density, dtype=np.float32) #density per point

E=1e8 #youngs modulus
E=np.full(nPoints, E, dtype=np.float32) #youngs modulus per point

nu=0.3 #poisson ratio
nu=np.full(nPoints, nu, dtype=np.float32) #poisson ratio per point

ys=2e6 #yield stress
ys=np.full(nPoints, ys, dtype=np.float32) #yield stress per point

# custom parameters associated with the constitutive model

hardening=0
hardening=np.full(nPoints, hardening, dtype=np.int32) # 

xi=10
xi=np.full(nPoints, xi, dtype=np.float32) #

softening=1e6
softening=np.full(nPoints, softening, dtype=np.float32) #

# other parameters
materialLabel = np.ones(nPoints, dtype=np.int32) #material label per point. 1 = solid, 2 = particle, etc
activeLabel = np.ones(nPoints, dtype=np.int32) #activity label per point. 1 = on, 0 = off, or other. activity and mertial is separated to allow for different materials to be active or inactive at different times

gravity = -9.81 #gravity in m/s^2, negative value for downward direction

# GPU arrays
yMod = wp.array(E, dtype=wp.float32, device=device)
poissonRatio = wp.array(nu, dtype=wp.float32, device=device)
mu = wp.zeros(nPoints, dtype=wp.float32, device=device)
lam = wp.zeros(nPoints, dtype=wp.float32, device=device)
bulk = wp.zeros(nPoints, dtype=wp.float32, device=device)

wp.launch(kernel = mpmRoutines.compute_mu_lam_bulk_from_E_nu, 
          dim = nPoints, 
          inputs = [yMod, 
                    poissonRatio], 
          outputs=[mu,
                   lam,
                   bulk], 
          device=device)

ys = wp.array(ys, dtype=wp.float32, device=device) #yield stress per point

materialLabel = wp.array(materialLabel, dtype=wp.int32, device=device)
activeLabel = wp.array(activeLabel, dtype=wp.int32, device=device)

# mpmParticleCount = np.sum(materialLabel.numpy() == 1) 


gravity = wp.vec3(0.0, 0.0, gravity) #gravity vector

particle_density = wp.array(density, dtype=wp.float32, device=device)

hardening = wp.array(hardening, dtype=wp.int32, device=device)
xi = wp.array(xi, dtype=wp.float32, device=device)
softening = wp.array(softening, dtype=wp.float32, device=device)



# dynamic material point parameters
particle_x= wp.array(x, dtype=wp.vec3)  # current position
particle_x_initial = wp.zeros(shape=nPoints, dtype=wp.vec3)   # initial particle position
particle_vol = wp.array(particle_volume, dtype=float)  # particle volume
particle_mass = wp.zeros(shape=nPoints, dtype=float)  # particle volume
wp.launch(kernel=mpmRoutines.get_float_array_product, 
          dim=nPoints, 
          inputs=[particle_density,
                  particle_vol,
                  particle_mass], 
          device=device)

particle_v= wp.zeros(shape=nPoints,dtype=wp.vec3)  # current velocity
particle_F= wp.zeros(shape=nPoints,dtype=wp.mat33)  # particle elastic deformation gradient
particle_F_trial= wp.zeros(shape=nPoints,dtype=wp.mat33)  # particle elastic deformation gradient

# initial deformation gradient is set to identity
wp.launch(
    kernel=mpmRoutines.set_mat33_to_identity,
    dim=nPoints,
    inputs=[particle_F_trial],
    device=device,
)

particle_stress= wp.zeros(shape=nPoints,dtype=wp.mat33)  # particle elastic deformation gradient
particle_C = wp.zeros(shape=nPoints, dtype=wp.mat33) # particle elastic right Cauchy-Green deformation tensor
particle_init_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)  # initial covariance matrix
particle_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)  
particle_radius=wp.array(particle_vol.numpy()**.33*np.pi/6) # particle radius, assuming spherical particles

# grid parameters- mass, velocities, etc.

grid_m = wp.zeros(shape=gridDims, dtype=float, device=device,) # grid mass from particles
grid_v_in = wp.zeros(shape=gridDims, dtype=wp.vec3, device=device,) # grsid momentum from particles
grid_v_out = wp.zeros(shape=gridDims, dtype=wp.vec3, device=device,) # grid velocity to particles
minBounds = wp.vec3(minBounds[0], minBounds[1], minBounds[2])  # minimum bounds of the grid
maxBounds = wp.vec3(maxBounds[0], maxBounds[1], maxBounds[2]) 

boundFriction = 0.0 # friction coefficient for the bounding box velocity boundary condition


# conversion params
eff = 0.05

# xpbd parameters 
particle_x_initial_xpbd = wp.zeros(shape=nPoints, dtype=wp.vec3)   # initial particle position upon phase change to xpbd
particle_x_initial_xpbd.fill_([1e6, 1e6, 1e6])
particle_v_initial_xpbd = wp.zeros(shape=nPoints, dtype=wp.vec3)   # initial particle position upon phase change to xpbd
particle_cumDist_xpbd = wp.zeros(shape=nPoints, dtype=float)   # initial particle position upon phase change to xpbd

particle_x_integrated = wp.zeros(shape=nPoints, dtype=wp.vec3)   #  position to iterate on
particle_v_integrated = wp.zeros(shape=nPoints, dtype=wp.vec3)  # velocity to iterate on
particle_x_deltaInt = wp.zeros(shape=nPoints, dtype=wp.vec3)   #  position to iterate on
particle_v_deltaInt = wp.zeros(shape=nPoints, dtype=wp.vec3)  # velocity to iterate on
particle_delta = wp.zeros(shape=nPoints, dtype=wp.vec3)  # delta to iterate on
particle_grid = wp.HashGrid(128, 128, 128)

xpbd_relaxation = 1.0  # relaxation factor for xpbd
dynamicParticleFriction = 0.05  # dynamic friction for xpbd
staticVelocityThreshold = 1e-5  # threshold for static ground velocity
staticParticleFriction = 0.1  # static friction for xpbd
xpbd_iterations = 4
particle_cohesion=0.0
sleepThreshold=0.5

swellingRatio=0.15
swellingActivationFactor=5
swellingMaxFactor=20
particleBaseRadius=particle_radius.numpy()
particleMaxRadius=particleBaseRadius*(1+swellingRatio)**(1/3)
particleBaseRadius=wp.array(particleBaseRadius)
particleMaxRadius=wp.array(particleMaxRadius)

particle_v_max = 1000.0#np.inf
max_radius = np.max(particleMaxRadius.numpy())
minBoundsXPBD = wp.vec3(minBounds[0]+3*dx, minBounds[1]+3*dx, minBounds[2]+3*dx)  # minimum bounds of the grid
maxBoundsXPBD = wp.vec3(maxBounds[0]-3*dx, maxBounds[1]-3*dx, maxBounds[2]-3*dx) 
# xpbdParticleCount = np.sum(materialLabel.numpy() == 2) # count of xpbd particles, i.e. particles with material label 2
#BOUNDARY CONDITIONS (DO THIS LATER)###############################################################################################################################

startBounds=0.0
endBounds=1e6

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


t=0
counter=0
for step in range(nSteps):

    # perform the mpm simulation step
    # if mpmParticleCount>0:
    # print(f'MPM step {step+1}/{nSteps}, time: {t:.4f}s, counter: {counter}')

    # zero the grids
    grid_m.zero_()
    grid_v_in.zero_()
    grid_v_out.zero_()

    # apply boundary and external conditions on points


    # compute stress per point from deformation gradient on the mpm particle points
    wp.launch(kernel = mpmRoutines.compute_stress_from_F_trial, 
            dim = nPoints, 
            inputs = [activeLabel,
                        materialLabel, 
                        particle_x,
                        particle_v,
                        particle_x_initial_xpbd,
                        particle_v_initial_xpbd,
                        particle_F,
                        particle_F_trial,
                        mu,
                        lam,
                        ys,
                        hardening,
                        xi,
                        softening,
                        particle_density,
                        yMod,
                        eff,
                        particle_stress],
            device=device)

    # perform particle to grid transfer for all particles
    wp.launch(
        kernel=mpmRoutines.p2g_apic_with_stress,
        dim=nPoints,
        inputs=[activeLabel,
                materialLabel, 
                particle_stress,
                particle_x,
                particle_v,
                particle_C,
                particle_vol,
                particle_mass,
                dx,
                invdx,
                minBounds,
                rpic_damping,
                grid_m,
                grid_v_in,
                dt],
        device=device,
    )  

    # apply external forces and damping on the grid
    wp.launch(
        kernel=mpmRoutines.grid_normalization_and_gravity,
        dim=gridDims,
        inputs=[grid_m,
                grid_v_in,
                grid_v_out,
                gravity,
                dt],
        device=device,
    )
    # print(f'Grid mass max: {grid_m.numpy().min()} - {grid_m.numpy().max()}, Grid velocity range: {grid_v_out.numpy().min()} - {grid_v_out.numpy().max()}')
    if grid_v_damping_scale < 1.0:
        wp.launch(
            kernel=mpmRoutines.add_damping_via_grid,
            dim=gridDims,
            inputs=[grid_v_out,
                    grid_v_damping_scale],
            device=device,
        )

    # apply boundary conditions on the grid as required
    # apply the bounding box velocity boundary condition on the grid
    wp.launch(
        kernel=mpmRoutines.collideBounds,
        dim=gridDims,
        inputs=[grid_v_out,
                gridDims[0],
                gridDims[1],
                gridDims[2],
                boundFriction],
        device=device,
    )
    # perform grid to particle transfer on the mpm particle points only
    wp.launch(
        kernel=mpmRoutines.g2p,
        dim=nPoints,
        inputs=[dt,
                activeLabel,
                materialLabel,
                particle_x,
                particle_v,
                particle_C,
                particle_F,
                particle_F_trial,
                particle_cov,
                invdx,
                grid_v_out,
                update_cov,
                minBounds],
        device=device,
    )  # x, v, C, F_trial are updated

    # perform the xpbd step if timestep has reached the threshold
    if np.mod(counter, mpmStepsPerXpbdStep) == 0 and counter>0:
        print(f'XPBD Step: {counter}, Time: {t}')

        # wp.launch(
        #         kernel=mpmRoutines.set_value_to_float_array,
        #         dim=nPoints,
        #         inputs=[particle_radius, max_radius],
        #         device=device,
        #     )
        
        particle_grid.build(particle_x, dx)

        # integrate the particles using xpbd with only gravity - impluse forces are handled elsewhere in conjunction with mpm particles
        wp.copy(particle_x_integrated,particle_x)
        wp.copy(particle_v_integrated,particle_v)
        wp.launch(
            kernel=xpbdRoutines.integrateParticlesXPBD,
            dim=nPoints,
            inputs=[
                activeLabel,
                materialLabel,
                particle_x,
                particle_v,
                gravity,
                dtxpbd,
                particle_x_integrated,
                particle_v_integrated,
                particle_v_max
            ],
            device=device,
        )
        # iterate xpbd contacts
        for i in range(xpbd_iterations):
            particle_delta.zero_()

            wp.launch(
                kernel=xpbdRoutines.my_solve_particle_bound_contacts,
                dim=nPoints,
                inputs=[
                    activeLabel,
                    materialLabel,
                    particle_x_integrated,
                    particle_v_integrated,
                    particle_mass,
                    particle_radius,
                    dynamicParticleFriction,
                    staticVelocityThreshold,
                    staticParticleFriction,
                    minBoundsXPBD,
                    maxBoundsXPBD,
                    dtxpbd,
                    xpbd_relaxation,
                ],
                outputs=[particle_delta],
                device=device,
            )
            wp.launch(
                kernel=xpbdRoutines.my_solve_particle_particle_contacts,
                dim=nPoints,
                inputs=[
                    activeLabel,
                    materialLabel,
                    particle_grid.id,
                    particle_x_integrated,
                    particle_v_integrated,
                    particle_mass,
                    particle_radius,
                    dynamicParticleFriction,
                    staticVelocityThreshold,
                    staticParticleFriction,
                    particle_cohesion,
                    max_radius,
                    dtxpbd,
                    xpbd_relaxation,
                ],
                outputs=[particle_delta],
                device=device
            )
            # print(f'Particle delta max: {particle_delta.numpy().max()}')
            wp.copy(particle_v_deltaInt, particle_v_integrated)
            wp.copy(particle_x_deltaInt, particle_x_integrated)
            wp.launch(
                kernel=xpbdRoutines.my_apply_particle_deltas,
                dim=nPoints,
                inputs=[
                    activeLabel,
                    materialLabel,
                    particle_x,
                    particle_x_integrated,
                    particle_delta,
                    dtxpbd,
                    particle_v_max
                ],
                outputs=[
                    particle_x_deltaInt,
                    particle_v_deltaInt
                ],
                device=device,
            )
            wp.copy(particle_v_integrated, particle_v_deltaInt)
            wp.copy(particle_x_integrated, particle_x_deltaInt)

        # sleep particles here and compute path integral
        wp.launch(kernel=xpbdRoutines.sleepParticles, 
                    dim=nPoints, 
                    inputs=[
                        activeLabel,
                        materialLabel,
                        sleepThreshold,
                        particle_radius,
                        particle_x,
                        particle_x_integrated,
                        particle_cumDist_xpbd,
                        dtxpbd], 
                    device=device)

        particle_x.assign(particle_x_integrated)
        particle_v.assign(particle_v_integrated)
        
        # clip particle velocities here - if a particle has not travelled far (cumdist), take the max if its energy release velocity and its xpbd calculated velocity (prevent overexplosion from overlap)
        wp.launch(kernel=xpbdRoutines.clipParticleVelocitiesOnPhaseChange, 
                dim=nPoints, 
                inputs=[activeLabel,
                        materialLabel,
                        particle_cumDist_xpbd, 
                        particle_v_initial_xpbd, 
                        particle_v, 
                        particle_radius], 
                device=device)

        # swell particles here
        if swellingRatio > 0:
            # print('type2')
            # could use path integral or monotonic radius from displacement. latter is used.
            wp.launch(kernel=xpbdRoutines.swellParticlesType2, 
                    dim=nPoints, 
                    inputs=[activeLabel,
                            materialLabel,
                            particle_x_initial_xpbd, 
                            particle_x, 
                            particle_radius, 
                            swellingActivationFactor, 
                            swellingMaxFactor, 
                            particleMaxRadius, 
                            particleBaseRadius], 
                    device=device)

    t=t+dt
    counter=counter+1
    if np.mod(counter,250)==0:
        print(f'Step: {counter}, Time: {t}, r: {particle_radius.numpy().max()}')
        if render:
            renderer.begin_frame()
            # colors=fs5PlotUtils.values_to_rgb(np.arange(0,nPoints,1),min_val=0, max_val=nPoints)
            # colors=fs5PlotUtils.values_to_rgb(ys.numpy(),min_val=0.0, max_val=ys.numpy().max())
            # colors=fs5PlotUtils.values_to_rgb(particle_radius.numpy(),min_val=particleBaseRadius.numpy().min(), max_val=particleMaxRadius.numpy().max())

            colors=fs5PlotUtils.values_to_rgb(particle_v.numpy()[:,2],min_val=particle_v.numpy()[:,2].min(), max_val=particle_v.numpy()[:,2].max())

            x=particle_stress.numpy()
            # x is your (N, 3, 3) array of stress tensors
            sigma = x.astype(np.float64)  # promote to float64 if needed

            # Compute mean (hydrostatic) stress for each tensor
            mean_stress = np.trace(sigma, axis1=1, axis2=2) / 3.0  # shape (N,)

            # Subtract mean stress from diagonal elements to get deviatoric tensor
            identity = np.eye(3)
            s = sigma - mean_stress[:, None, None] * identity  # broadcasted subtraction

            # Compute von Mises stress
            von_mises = np.sqrt(1.5 * np.sum(s**2, axis=(1, 2)))  # shape (N,)

            colors=fs5PlotUtils.values_to_rgb(von_mises,min_val=np.min(von_mises), max_val=np.max(von_mises)+1)

            renderer.render_points(points=particle_x, name="points", radius=particle_radius.numpy(), colors=colors, dynamic=True)
            # renderer.render_box(name='simBounds',pos=[grid_lim/2,grid_lim/2,grid_lim/2],extents=[grid_lim/2,grid_lim/2,grid_lim/2],rot=[0,0,0,1])
            renderer.end_frame()
            renderer.update_view_matrix()

            # engine_utils.save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=1, save_to_h5=0)
            # time.sleep(1)
        if saveFlag:
            # save points and fields for visualization
            fs5PlotUtils.save_grid_and_particles_vti_vtp(
                output_prefix=f"./output/sim_step_{step:06d}",
                grid_mass=grid_m.numpy(),              # shape (nx, ny, nz)
                minBounds=minBounds,                         # (x0, y0, z0)
                dx=dx,
                particle_positions=particle_x.numpy(),          # shape (N, 3)
                particle_radius=np.arange(0,nPoints,1)                         # for Point Gaussian
            )
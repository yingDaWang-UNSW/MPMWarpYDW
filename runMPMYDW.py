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

# load the material points
domainFile="./exampleDomains/annular_arch_particles.h5"
h5file = h5py.File(domainFile, "r")
x, particle_volume = h5file["x"], h5file["particle_volume"]
x=np.array(x).T

# orient the renderer to look at the centroid of the particles
renderer=fs5PlotUtils.look_at_centroid(x,renderer,renderer.camera_fov)

nPoints= x.shape[0]
print(f"Number of particles: {nPoints}")

# infer the grid parameters from the particle positions
minBounds=np.min(x,0)-25
maxBounds=np.max(x,0)+25

particleDiameter = np.mean(particle_volume)**0.33

# Compute grid spacing (assume cubic)
dx = particleDiameter*2

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

# grid mass, velocities, etc.

grid_m = wp.zeros(shape=gridDims, dtype=float, device=device,)
grid_v_in = wp.zeros(shape=gridDims, dtype=float, device=device,)
grid_v_out = wp.zeros(shape=gridDims, dtype=float, device=device,)

#PARAMETERS###############################################################################################################################

# point properties from CPU/file

density = 3000.0 #density of the material
density = np.full(nPoints, density, dtype=np.float32) #density per point

E=1e10 #youngs modulus
E=np.full(nPoints, E, dtype=np.float32) #youngs modulus per point

nu=0.3 #poisson ratio
nu=np.full(nPoints, nu, dtype=np.float32) #poisson ratio per point

ys=1e6 #yield stress
ys=np.full(nPoints, ys, dtype=np.float32) #yield stress per point

# custom parameters associated with the constitutive model

hardening=0
hardening=np.full(nPoints, hardening, dtype=np.int32) #youngs modulus per point

xi=0.0
xi=np.full(nPoints, xi, dtype=np.float32) #youngs modulus per point

softening=1e6
softening=np.full(nPoints, softening, dtype=np.float32) #youngs modulus per point

# xpbd parameters






# other parameters
materialLabel = np.ones(nPoints, dtype=np.int32) #material label per point. 1 = solid, 2 = particle, etc

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

gravity = wp.vec3(0.0, 0.0, gravity) #gravity vector

rho = wp.array(density, dtype=wp.float32, device=device)

hardening = wp.array(hardening, dtype=wp.int32, device=device)
xi = wp.array(xi, dtype=wp.float32, device=device)
softening = wp.array(softening, dtype=wp.float32, device=device)



# dynamic point parameters
particle_x= wp.array(x, dtype=wp.vec3)  # current position
particle_F= wp.zeros(shape=nPoints,dtype=wp.mat33)  # particle elastic deformation gradient
particle_F_trial= wp.zeros(shape=nPoints,dtype=wp.mat33)  # particle elastic deformation gradient
particle_stress= wp.zeros(shape=nPoints,dtype=wp.mat33)  # particle elastic deformation gradient

#BOUNDARY CONDITIONS (DO THIS LATER)###############################################################################################################################


# simulation parameters
dt = 1e-4 #time step in seconds 
dtxpbd=1e-2 #time step for xpbd in seconds
mpmStepsPerXpbdStep = int(dtxpbd/dt) #number of mpm steps per xpbd step
nSteps = 1 #number of simulation steps

t=0
counter=0
for step in range(nSteps):
    print(f'MPM step {step+1}/{nSteps}, time: {t:.4f}s, counter: {counter}')

    # perform the mpm simulation step

    # zero the grids
    grid_m.zero_()
    grid_v_in.zero_()
    grid_v_out.zero_()

    # apply boundary conditions on points (do this later)


    # compute stress per point 
    wp.launch(kernel = mpmRoutines.compute_stress_from_F_trial, 
              dim = nPoints, 
              inputs = [materialLabel, 
                        particle_F,
                        particle_F_trial,
                        mu,
                        lam,
                        ys,
                        hardening,
                        xi,
                        softening],
              outputs=[particle_stress], 
              device=device)

    # perform particle to grid transfer - special consideration for material type 2 

    # compute grid updates including external forces and grid-based boundary conditions

    # perform grid to particle transfer

    # perform the xpbd step if timestep has reached the threshold
    if np.mod(counter, mpmStepsPerXpbdStep) == 0 and counter>0:
        # perform xpbd step
        print('Performing XPBD step')
        # in xpbd, the mpm points are treated as particles. The material 1 points are static, and the material 2 points are in motion. Deltas are calculated and applied for the material 2 points in relation to contact with all points.   
    
    t=t+dt
    counter=counter+1
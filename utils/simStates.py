"""
State classes for MPM-XPBD simulation
Organizes variables into logical groups for cleaner code structure
"""

import warp as wp
import numpy as np


class SimState:
    """Global simulation state - variables needed across MPM and XPBD"""
    def __init__(self, args, nPoints, device="cuda:0"):
        self.device = device
        self.nPoints = nPoints
        
        # Time stepping
        self.dt = args.dt
        self.dtxpbd = args.dtxpbd
        self.mpmStepsPerXpbdStep = int(args.dtxpbd / args.dt)
        self.nSteps = args.nSteps
        self.bigSteps = args.bigSteps
        self.residualThreshold = args.residualThreshold
        self.t = 0.0  # simulation time
        
        # Rendering and output
        self.render = args.render
        self.saveFlag = args.saveFlag
        self.color_mode = args.color_mode
        self.outputFolder = args.outputFolder
        
        # Physics
        self.gravity = wp.vec3(0.0, 0.0, args.gravity)
        
        # Shared particle arrays
        self.particle_x = None  # initialized later
        self.particle_v = None
        self.particle_mass = None
        self.particle_vol = None
        self.particle_radius = None
        self.particle_density = None
        self.materialLabel = None  # 1=MPM, 2=XPBD
        self.activeLabel = None    # 0=sleeping, 1=active
        self.particleBCMask = None  # boundary condition mask

        # Domain bounds
        self.minBounds = None
        self.maxBounds = None
        self.dx = None
        self.invdx = None
        
        # Convergence tracking
        self.residual = wp.array([1e10], dtype=float, device=device)
        self.numActiveParticles = wp.array([0], dtype=wp.int32, device=device)

class MPMState:
    """MPM-specific state variables"""
    def __init__(self, args, nPoints, gridDims, device="cuda:0"):
        self.device = device
        self.nPoints = nPoints
        self.gridDims = gridDims
        
        # MPM parameters
        self.rpic_damping = args.rpic_damping
        self.grid_v_damping_scale = args.grid_v_damping_scale
        self.update_cov = args.update_cov
        self.constitutive_model = args.constitutive_model  # 0=Von Mises, 1=Drucker-Prager
        self.K0 = args.K0  # lateral earth pressure coefficient
        
        # Boundary conditions
        self.boundaryCondition = args.boundaryCondition  # "friction", "restitution", or "absorbing"
        self.boundFriction = args.boundFriction  # Friction coefficient for "friction" mode
        self.boundRestitution = args.boundRestitution  # Coefficient of restitution for "restitution" mode
        self.boundaryPadding = None  # Number of grid cells from boundary where BCs apply (initialized later)

        # Material properties (per particle)
        self.mu = None  # shear modulus
        self.lam = None  # Lame's first parameter
        self.bulk = None  # bulk modulus
        self.ys = None  # yield stress
        self.ys_base = None  # base yield stress for creep
        self.alpha = None  # Drucker-Prager pressure sensitivity
        self.hardening = None
        self.softening = None
        self.eta_shear = None  # shear viscosity
        self.eta_bulk = None   # bulk viscosity
        self.strainCriteria = None  # critical plastic strain for damage
        
        # Particle state variables
        self.particle_F = wp.zeros(shape=nPoints, dtype=wp.mat33, device=device)
        self.particle_F_trial = wp.zeros(shape=nPoints, dtype=wp.mat33, device=device)
        self.particle_stress = wp.zeros(shape=nPoints, dtype=wp.mat33, device=device)
        self.particle_accumulated_strain = wp.zeros(shape=nPoints, dtype=float, device=device)
        self.particle_damage = wp.zeros(shape=nPoints, dtype=float, device=device)
        self.particle_C = wp.zeros(shape=nPoints, dtype=wp.mat33, device=device)  # APIC affine matrix
        self.particle_init_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)
        self.particle_cov = wp.zeros(shape=nPoints * 6, dtype=float, device=device)
        
        # Grid arrays
        self.grid_m = wp.zeros(shape=gridDims, dtype=float, device=device)
        self.grid_v_in = wp.zeros(shape=gridDims, dtype=wp.vec3, device=device)
        self.grid_v_out = wp.zeros(shape=gridDims, dtype=wp.vec3, device=device)
        self.gridBCIndex = wp.zeros(shape=gridDims, dtype=wp.int32, device=device)
        
        # Volumetric locking correction arrays (cell-averaged divergence)
        self.volumetric_locking_correction = args.volumetric_locking_correction
        self.cell_div_v_weighted = wp.zeros(shape=gridDims, dtype=float, device=device)  # Σ(∇·v_p * V_p)
        self.cell_vol_sum = wp.zeros(shape=gridDims, dtype=float, device=device)  # Σ(V_p)

        # Initial position tracking (for phase transition)
        self.particle_x_initial = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.eff = args.eff  # energy release efficiency


class XPBDState:
    """XPBD-specific state variables"""
    def __init__(self, args, nPoints, device="cuda:0"):
        self.device = device
        self.nPoints = nPoints
        
        # XPBD solver parameters
        self.xpbd_iterations = args.xpbd_iterations
        self.xpbd_relaxation = args.xpbd_relaxation
        self.xpbd_mpm_coupling_strength = args.xpbd_mpm_coupling_strength
        self.dynamicParticleFriction = args.dynamicParticleFriction
        self.staticParticleFriction = args.staticParticleFriction
        self.staticVelocityThreshold = args.staticVelocityThreshold
        self.particle_cohesion = args.particle_cohesion
        self.sleepThreshold = args.sleepThreshold
        self.particle_v_max = args.particle_v_max
        
        # Swelling parameters (particle expansion during fragmentation)
        self.swellingRatio = args.swellingRatio
        self.swellingActivationFactor = args.swellingActivationFactor
        self.swellingMaxFactor = args.swellingMaxFactor
        self.particleBaseRadius = None  # initialized later
        self.particleMaxRadius = None
        
        # XPBD particle state
        self.particle_x_initial_xpbd = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_x_initial_xpbd.fill_([1e6, 1e6, 1e6])  # large value for non-XPBD particles
        self.particle_v_initial_xpbd = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_cumDist_xpbd = wp.zeros(shape=nPoints, dtype=float, device=device)
        self.particle_distance_total = wp.zeros(shape=nPoints, dtype=float, device=device)
        
        # XPBD integration temporaries
        self.particle_x_integrated = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_v_integrated = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_x_deltaInt = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_v_deltaInt = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_delta = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)
        self.particle_v_before_contact = wp.zeros(shape=nPoints, dtype=wp.vec3, device=device)  # For C/F update
        
        # Spatial hash grid for neighbor search
        self.particle_grid = wp.HashGrid(128, 128, 128, device=device)
        
        # XPBD-specific bounds (smaller to prevent boundary issues)
        self.minBoundsXPBD = None  # initialized later
        self.maxBoundsXPBD = None
        self.max_radius = None

import warp as wp
from utils import xpbdRoutines
from utils import mpmRoutines


def mpmSimulationStep(sim, mpm, xpbd, device,):
    """
    Perform one MPM simulation step (stress update, P2G, grid ops, G2P).
    """
    # 1. Zero the grids
    mpm.grid_m.zero_()
    mpm.grid_v_in.zero_()
    mpm.grid_v_out.zero_()
    # 2. Compute stress at particles
    wp.launch(
        kernel=mpmRoutines.compute_stress_from_F_trial,
        dim=sim.nPoints,
        inputs=[
            sim.activeLabel,
            sim.materialLabel,
            sim.particle_x,
            sim.particle_v,
            xpbd.particle_x_initial_xpbd,
            xpbd.particle_v_initial_xpbd,
            mpm.particle_F,
            mpm.particle_F_trial,
            mpm.mu,
            mpm.lam,
            mpm.ys,
            mpm.hardening,
            mpm.softening,
            sim.particle_density,
            mpm.strainCriteria,
            mpm.eff,
            mpm.eta_shear,
            mpm.eta_bulk,
            mpm.particle_C,
            mpm.particle_stress,
            mpm.particle_accumulated_strain,
            mpm.particle_damage,
            mpm.constitutive_model,
            mpm.alpha
        ],
        device=device
    )

    # 3. Particle-to-grid transfer (APIC + stress) - MPM particles only
    wp.launch(
        kernel=mpmRoutines.p2g_apic_with_stress,
        dim=sim.nPoints,
        inputs=[
            sim.activeLabel,
            sim.materialLabel,
            mpm.particle_stress,
            sim.particle_x,
            sim.particle_v,
            mpm.particle_C,
            sim.particle_vol,
            sim.particle_mass,
            sim.dx,
            sim.invdx,
            sim.minBounds,
            mpm.rpic_damping,
            mpm.grid_m,
            mpm.grid_v_in,
            sim.dt,
        ],
        device=device
    )

    # 3a. Transfer XPBD gravitational force to grid (mass coupling without pulling)
    # This adds F=m*g as momentum impulse, WITHOUT adding XPBD mass to grid
    # Result: MPM feels XPBD weight continuously, creating stress field for elastic propagation
    # No ghost mass when XPBD moves away (grid_v_in cleared each frame)
    # wp.launch(
    #     kernel=mpmRoutines.apply_xpbd_gravity_force_to_grid,
    #     dim=sim.nPoints,
    #     inputs=[
    #         sim.activeLabel,
    #         sim.materialLabel,
    #         sim.particle_x,
    #         sim.particle_mass,
    #         sim.gravity,
    #         sim.dx,
    #         sim.invdx,
    #         sim.minBounds,
    #         sim.dt,
    #         mpm.grid_v_in,  # Add momentum to grid (before normalization)
    #     ],
    #     device=device
    # )

    # 4. Grid normalization + gravity
    wp.launch(
        kernel=mpmRoutines.grid_normalization_and_gravity,
        dim=mpm.gridDims,
        inputs=[
            mpm.grid_m,
            mpm.grid_v_in,
            mpm.grid_v_out,
            sim.gravity,
            sim.dt
        ],
        device=device
    )

    #4a. external forces on grid that can affect both mpm and xpbd particles


    # 5. Optional grid damping
    if mpm.grid_v_damping_scale < 1.0:
        wp.launch(
            kernel=mpmRoutines.add_damping_via_grid,
            dim=mpm.gridDims,
            inputs=[
                mpm.grid_v_out,
                mpm.grid_v_damping_scale
            ],
            device=device
        )

    # 6. Apply boundary conditions on grid
    # Choose boundary condition based on user setting
    if mpm.boundaryCondition == "friction":
        # Coulomb friction boundary (abrupt)
        wp.launch(
            kernel=mpmRoutines.collideBounds,
            dim=mpm.gridDims,
            inputs=[
                mpm.grid_v_out,
                mpm.gridDims[0],
                mpm.gridDims[1],
                mpm.gridDims[2],
                mpm.boundFriction,  # Friction coefficient μ
                sim.dt,  # Time step
                wp.abs(sim.gravity[2]),  # Gravity magnitude
                mpm.boundaryPadding
            ],
            device=device
        )
    elif mpm.boundaryCondition == "friction_gradual":
        # Coulomb friction boundary with gradual transition (reduces artifacts)
        wp.launch(
            kernel=mpmRoutines.collideBoundsGradualFriction,
            dim=mpm.gridDims,
            inputs=[
                mpm.grid_v_out,
                mpm.gridDims[0],
                mpm.gridDims[1],
                mpm.gridDims[2],
                mpm.boundFriction,  # Friction coefficient μ
                sim.dt,  # Time step
                wp.abs(sim.gravity[2]),  # Gravity magnitude
                mpm.boundaryPadding
            ],
            device=device
        )
    elif mpm.boundaryCondition == "restitution":
        # Restitution (elastic bounce) boundary with tangential friction
        wp.launch(
            kernel=mpmRoutines.collideBoundsRestitution,
            dim=mpm.gridDims,
            inputs=[
                mpm.grid_v_out,
                mpm.gridDims[0],
                mpm.gridDims[1],
                mpm.gridDims[2],
                mpm.boundRestitution,  # Coefficient of restitution
                mpm.boundFriction,  # Friction coefficient μ for tangential directions
                sim.dt,  # Time step
                wp.abs(sim.gravity[2]),  # Gravity magnitude
                mpm.boundaryPadding
            ],
            device=device
        )
    elif mpm.boundaryCondition == "absorbing":
        # Absorbing (damping) boundary
        wp.launch(
            kernel=mpmRoutines.collideBoundsAbsorbing,
            dim=mpm.gridDims,
            inputs=[
                mpm.grid_v_out,
                mpm.gridDims[0],
                mpm.gridDims[1],
                mpm.gridDims[2],
                mpm.boundaryPadding
            ],
            device=device
        )

    # 7. Grid-to-particle transfer (update x, v, C, F_trial)
    wp.launch(
        kernel=mpmRoutines.g2p,
        dim=sim.nPoints,
        inputs=[
            sim.dt,
            sim.activeLabel,
            sim.materialLabel,
            sim.particle_x,
            sim.particle_v,
            mpm.particle_C,
            mpm.particle_F,
            mpm.particle_F_trial,
            mpm.particle_cov,
            sim.invdx,
            mpm.grid_v_out,
            mpm.update_cov,
            sim.minBounds
        ],
        device=device
    )
    
    # 8. Volumetric locking correction (Itasca MPAC method)
    # Prevents artificial stiffening for near-incompressible materials
    # by averaging the volumetric component of velocity gradient over cells
    if mpm.volumetric_locking_correction:
        # Zero cell arrays
        mpm.cell_div_v_weighted.zero_()
        mpm.cell_vol_sum.zero_()
        
        # Phase 1: Accumulate velocity divergence to cells
        wp.launch(
            kernel=mpmRoutines.accumulate_cell_divergence,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                sim.particle_x,
                mpm.particle_C,
                sim.particle_vol,
                sim.invdx,
                sim.minBounds,
                mpm.cell_div_v_weighted,
                mpm.cell_vol_sum,
            ],
            device=device
        )
        
        # Phase 2: Apply corrected velocity gradient to particles
        wp.launch(
            kernel=mpmRoutines.apply_volumetric_locking_correction,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                sim.particle_x,
                mpm.particle_C,
                sim.invdx,
                sim.minBounds,
                mpm.cell_div_v_weighted,
                mpm.cell_vol_sum,
            ],
            device=device
        )

def xpbdSimulationStep(sim, mpm, xpbd, device, xpbd_only=False):
    """
    Integrate particles using XPBD with gravity and handle collisions, sleeping, and swelling.
    
    Parameters
    ----------
    sim : SimState
    mpm : MPMState (can be None if xpbd_only=True)
    xpbd : XPBDState
    device : str
    xpbd_only : bool
        If True, only XPBD particles are updated (MPM frozen). Skips MPM contact 
        label reset and MPM-XPBD coupling impulses. Used for debris settling phase.
    """

    # Build grid
    xpbd.particle_grid.build(sim.particle_x, sim.dx)

    # Initial integration (gravity only)
    wp.copy(xpbd.particle_x_integrated, sim.particle_x)
    wp.copy(xpbd.particle_v_integrated, sim.particle_v)
    wp.launch(
        kernel=xpbdRoutines.integrateParticlesXPBD,
        dim=sim.nPoints,
        inputs=[
            sim.activeLabel,
            sim.materialLabel,
            xpbd.particle_x_integrated,
            xpbd.particle_v_integrated,
            sim.gravity,
            sim.dtxpbd,
            xpbd.particle_x_integrated,
            xpbd.particle_v_integrated,
            xpbd.particle_v_max
        ],
        device=device,
    )

    # Reset MPM contact labels before contact detection (if transition lock is enabled)
    # This sets materialLabel 0 -> 1 for all MPM particles, allowing transition if they're not in contact
    # The contact kernel will set them back to 0 if they are in contact with XPBD particles
    # Skip this in xpbd_only mode since MPM is frozen
    if xpbd.mpm_contact_transition_lock and not xpbd_only:
        wp.launch(
            kernel=xpbdRoutines.reset_mpm_contact_labels,
            dim=sim.nPoints,
            inputs=[sim.activeLabel, sim.materialLabel],
            device=device,
        )

    # XPBD iterations (contacts)
    for _ in range(xpbd.xpbd_iterations):
        xpbd.particle_delta.zero_()

        # Bound contacts
        wp.launch(
            kernel=xpbdRoutines.my_solve_particle_bound_contacts,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                xpbd.particle_x_integrated,
                xpbd.particle_v_integrated,
                sim.particle_mass,
                sim.particle_radius,
                xpbd.dynamicParticleFriction,
                xpbd.staticVelocityThreshold,
                xpbd.staticParticleFriction,
                xpbd.minBoundsXPBD,
                xpbd.maxBoundsXPBD,
                sim.dtxpbd,
                xpbd.xpbd_relaxation,
            ],
            outputs=[xpbd.particle_delta],
            device=device,
        )

        # Particle-particle contacts
        # In xpbd_only mode, disable mpm_contact_transition_lock since MPM is frozen
        contact_lock = xpbd.mpm_contact_transition_lock if not xpbd_only else 0
        wp.launch(
            kernel=xpbdRoutines.my_solve_particle_particle_contacts,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                xpbd.particle_grid.id,
                xpbd.particle_x_integrated,
                xpbd.particle_v_integrated,
                sim.particle_mass,
                sim.particle_radius,
                xpbd.dynamicParticleFriction,
                xpbd.staticVelocityThreshold,
                xpbd.staticParticleFriction,
                xpbd.particle_cohesion,
                xpbd.max_radius,
                sim.dtxpbd,
                xpbd.xpbd_relaxation,
                contact_lock,
            ],
            outputs=[xpbd.particle_delta],
            device=device,
        )

        # Apply deltas
        wp.copy(xpbd.particle_v_deltaInt, xpbd.particle_v_integrated)
        wp.copy(xpbd.particle_x_deltaInt, xpbd.particle_x_integrated)
        wp.launch(
            kernel=xpbdRoutines.my_apply_particle_deltas,
            dim=sim.nPoints,
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                sim.particle_x,
                xpbd.particle_x_integrated,
                xpbd.particle_delta,
                sim.dtxpbd,
                xpbd.particle_v_max
            ],
            outputs=[
                xpbd.particle_x_deltaInt,
                xpbd.particle_v_deltaInt
            ],
            device=device,
        )
        wp.copy(xpbd.particle_v_integrated, xpbd.particle_v_deltaInt)
        wp.copy(xpbd.particle_x_integrated, xpbd.particle_x_deltaInt)

    # Sleep particles & path integral (also counts sleeping XPBD particles)
    sim.numTotalXPBD.zero_()
    sim.numSleepingXPBD.zero_()
    wp.launch(
        kernel=xpbdRoutines.sleepParticles, 
        dim=sim.nPoints, 
        inputs=[
            sim.activeLabel,
            sim.materialLabel,
            xpbd.sleepThreshold,
            sim.particle_radius,
            sim.particle_x,
            xpbd.particle_x_integrated,
            xpbd.particle_cumDist_xpbd,
            sim.dtxpbd,
            sim.numTotalXPBD,
            sim.numSleepingXPBD
        ], 
        device=device
    )
    
    # Apply MPM contact impulses (always run to reset MPM positions, but skip velocity change in xpbd_only mode)
    # Store velocity before contact impulse for C/F update
    wp.copy(xpbd.particle_v_before_contact, sim.particle_v)
    
    xpbd_only_flag = 1 if xpbd_only else 0
    wp.launch(
        kernel=xpbdRoutines.apply_mpm_contact_impulses,
        dim=sim.nPoints,
        inputs=[
            sim.activeLabel,
            sim.materialLabel,
            sim.particle_x,  # Original MPM positions
            sim.particle_v,
            xpbd.particle_x_integrated,  # Converged XPBD positions
            xpbd.particle_v_integrated,  # Velocities from convergence
            sim.dtxpbd,
            xpbd.particle_v_max,
            xpbd.xpbd_mpm_coupling_strength,  # Coupling parameter
            xpbd_only_flag,  # If 1, only reset position, don't change velocity
        ],
        outputs=[xpbd.particle_v_integrated],  # Update velocities only
        device=device,
    )
    
    # Commit integrated positions/velocities
    sim.particle_x.assign(xpbd.particle_x_integrated)
    sim.particle_v.assign(xpbd.particle_v_integrated)
    
    # CRITICAL: Update C and F to match new velocity
    # Without this, elasticity is broken (v inconsistent with F/C)
    # wp.launch(
    #     kernel=mpmRoutines.update_C_and_F_from_velocity_change,
    #     dim=sim.nPoints,
    #     inputs=[
    #         sim.activeLabel,
    #         sim.materialLabel,
    #         sim.particle_x,
    #         xpbd.particle_v_before_contact,  # Velocity before impulse
    #         sim.particle_v,  # Velocity after impulse
    #         mpm.particle_C,
    #         mpm.particle_F,
    #         mpm.particle_F_trial,
    #         sim.invdx,
    #         sim.minBounds,
    #         sim.dtxpbd,
    #         mpm.grid_v_out,
    #     ],
    #     device=device,
    # )

    # Clip particle velocities to avoid over-explosion
    # wp.launch(
    #     kernel=xpbdRoutines.clipParticleVelocitiesOnPhaseChange, 
    #     dim=nPoints, 
    #     inputs=[
    #         activeLabel,
    #         materialLabel,
    #         particle_cumDist_xpbd, 
    #         particle_v_initial_xpbd, 
    #         particle_v, 
    #         particle_radius
    #     ], 
    #     device=device
    # )

    # Swelling (optional)
    if xpbd.swellingRatio > 0:
        wp.launch(
            kernel=xpbdRoutines.swellParticlesType2, 
            dim=sim.nPoints, 
            inputs=[
                sim.activeLabel,
                sim.materialLabel,
                xpbd.particle_x_initial_xpbd,
                sim.particle_x,
                sim.particle_radius,
                xpbd.swellingActivationFactor,
                xpbd.swellingMaxFactor,
                xpbd.particleMaxRadius,
                xpbd.particleBaseRadius
            ],
            device=device
        )


# save into A the minimum between A and B
@wp.kernel
def arrayScalarMultiply(
    grid_A: wp.array(dtype=float),
    scalar: float):
    x = wp.tid()
    grid_A[x] = grid_A[x] * scalar


@wp.kernel
def velocityConvergence(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    particle_v: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    residual: wp.array(dtype=float),
):
    p = wp.tid()
    if activeLabel[p] == 1: # compute sqrt of the sum of the velocity magnitude over the radius
        particle_v_mag = wp.length(particle_v[p])/particle_radius[p]
        wp.atomic_add(residual, 0, particle_v_mag)

@wp.kernel
def countActiveParticles(
    activeLabel: wp.array(dtype=wp.int32),
    sum: wp.array(dtype=wp.int32)
):
    p = wp.tid()
    if activeLabel[p] == 1:
        wp.atomic_add(sum, 0, 1)

@wp.kernel
def countParticlesByType(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    numActiveMPM: wp.array(dtype=int),
    numActiveXPBD: wp.array(dtype=int),
    numInactiveMPM: wp.array(dtype=int),
    numInactiveXPBD: wp.array(dtype=int),
):
    """Count particles by type (MPM vs XPBD) and status (active vs inactive/sleeping)."""
    p = wp.tid()
    
    is_active = activeLabel[p] == 1
    is_xpbd = materialLabel[p] == 2
    
    if is_xpbd:
        if is_active:
            wp.atomic_add(numActiveXPBD, 0, 1)
        else:
            wp.atomic_add(numInactiveXPBD, 0, 1)
    else:
        # MPM particles (materialLabel 0 or 1)
        if is_active:
            wp.atomic_add(numActiveMPM, 0, 1)
        else:
            wp.atomic_add(numInactiveMPM, 0, 1)


@wp.kernel
def sum_float_array(
    arr: wp.array(dtype=float),
    result: wp.array(dtype=float),
):
    """Sum all elements of a float array using atomic add."""
    p = wp.tid()
    wp.atomic_add(result, 0, arr[p])


@wp.kernel
def sum_float_array_active(
    arr: wp.array(dtype=float),
    activeLabel: wp.array(dtype=wp.int32),
    result: wp.array(dtype=float),
    count: wp.array(dtype=int),
):
    """Sum elements of a float array only for active particles."""
    p = wp.tid()
    if activeLabel[p] == 1:
        wp.atomic_add(result, 0, arr[p])
        wp.atomic_add(count, 0, 1)


@wp.kernel
def sum_float_array_by_material(
    arr: wp.array(dtype=float),
    materialLabel: wp.array(dtype=wp.int32),
    result_mpm: wp.array(dtype=float),
    result_xpbd: wp.array(dtype=float),
    count_mpm: wp.array(dtype=int),
    count_xpbd: wp.array(dtype=int),
):
    """Sum elements by material type (MPM vs XPBD)."""
    p = wp.tid()
    val = arr[p]
    
    if materialLabel[p] == 2:  # XPBD
        wp.atomic_add(result_xpbd, 0, val)
        wp.atomic_add(count_xpbd, 0, 1)
    else:  # MPM (label 0 or 1)
        wp.atomic_add(result_mpm, 0, val)
        wp.atomic_add(count_mpm, 0, 1)


def compute_mean_gpu(arr, sim, device, filtered=True):
    """
    Compute mean of a float array on GPU.
    
    Parameters
    ----------
    arr : wp.array(dtype=float)
        Array to average
    sim : SimState
        Simulation state (for scratch arrays and activeLabel)
    device : str
        Warp device
    filtered : bool
        If True, only average active particles
    
    Returns
    -------
    float
        Mean value
    """
    sim.scratchFloat.zero_()
    sim.scratchInt.zero_()
    
    if filtered:
        wp.launch(
            kernel=sum_float_array_active,
            dim=sim.nPoints,
            inputs=[arr, sim.activeLabel, sim.scratchFloat, sim.scratchInt],
            device=device
        )
        total = sim.scratchFloat.numpy()[0]
        count = sim.scratchInt.numpy()[0]
    else:
        wp.launch(
            kernel=sum_float_array,
            dim=sim.nPoints,
            inputs=[arr, sim.scratchFloat],
            device=device
        )
        total = sim.scratchFloat.numpy()[0]
        count = sim.nPoints
    
    return total / count if count > 0 else 0.0

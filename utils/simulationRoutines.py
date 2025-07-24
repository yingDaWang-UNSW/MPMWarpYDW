import warp as wp
from utils import xpbdRoutines
from utils import mpmRoutines

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

def mpmSimulationStep(
    particle_x,
    particle_v,
    particle_x_initial_xpbd,
    particle_v_initial_xpbd,
    particle_F,
    particle_F_trial,
    particle_stress,
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
    xi,
    softening,
    yMod,
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
):
    """
    Perform one MPM simulation step (stress update, P2G, grid ops, G2P).
    """
    # 1. Zero the grids
    grid_m.zero_()
    grid_v_in.zero_()
    grid_v_out.zero_()
    # 2. Compute stress at particles
    wp.launch(
        kernel=mpmRoutines.compute_stress_from_F_trial,
        dim=nPoints,
        inputs=[
            activeLabel,
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
            eta_shear,
            eta_bulk,
            particle_C,
            particle_stress
        ],
        device=device
    )

    # 3. Particle-to-grid transfer (APIC + stress)
    wp.launch(
        kernel=mpmRoutines.p2g_apic_with_stress,
        dim=nPoints,
        inputs=[
            activeLabel,
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
            dt
        ],
        device=device
    )

    # 4. Grid normalization + gravity
    wp.launch(
        kernel=mpmRoutines.grid_normalization_and_gravity,
        dim=gridDims,
        inputs=[
            grid_m,
            grid_v_in,
            grid_v_out,
            gravity,
            dt
        ],
        device=device
    )

    # 5. Optional grid damping
    if grid_v_damping_scale < 1.0:
        wp.launch(
            kernel=mpmRoutines.add_damping_via_grid,
            dim=gridDims,
            inputs=[
                grid_v_out,
                grid_v_damping_scale
            ],
            device=device
        )

    # 6. Apply boundary conditions on grid
    wp.launch(
        kernel=mpmRoutines.collideBounds,
        dim=gridDims,
        inputs=[
            grid_v_out,
            gridDims[0],
            gridDims[1],
            gridDims[2],
            boundFriction
        ],
        device=device
    )

    # 7. Grid-to-particle transfer (update x, v, C, F_trial)
    wp.launch(
        kernel=mpmRoutines.g2p,
        dim=nPoints,
        inputs=[
            dt,
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
            minBounds
        ],
        device=device
    )

def xpbdSimulationStep(
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
):
    """
    Integrate particles using XPBD with gravity and handle collisions, sleeping, and swelling.
    """

    # Build grid
    particle_grid.build(particle_x, dx)

    # Initial integration (gravity only)
    wp.copy(particle_x_integrated, particle_x)
    wp.copy(particle_v_integrated, particle_v)
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

    # XPBD iterations (contacts)
    for _ in range(xpbd_iterations):
        particle_delta.zero_()

        # Bound contacts
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

        # Particle-particle contacts
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
            device=device,
        )

        # Apply deltas
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

    # Sleep particles & path integral
    wp.launch(
        kernel=xpbdRoutines.sleepParticles, 
        dim=nPoints, 
        inputs=[
            activeLabel,
            materialLabel,
            sleepThreshold,
            particle_radius,
            particle_x,
            particle_x_integrated,
            particle_cumDist_xpbd,
            dtxpbd
        ], 
        device=device
    )

    # Commit integrated positions/velocities
    particle_x.assign(particle_x_integrated)
    particle_v.assign(particle_v_integrated)

    # Clip particle velocities to avoid over-explosion
    wp.launch(
        kernel=xpbdRoutines.clipParticleVelocitiesOnPhaseChange, 
        dim=nPoints, 
        inputs=[
            activeLabel,
            materialLabel,
            particle_cumDist_xpbd, 
            particle_v_initial_xpbd, 
            particle_v, 
            particle_radius
        ], 
        device=device
    )

    # Swelling (optional)
    if swellingRatio > 0:
        wp.launch(
            kernel=xpbdRoutines.swellParticlesType2, 
            dim=nPoints, 
            inputs=[
                activeLabel,
                materialLabel,
                particle_x_initial_xpbd, 
                particle_x, 
                particle_radius, 
                swellingActivationFactor, 
                swellingMaxFactor, 
                particleMaxRadius, 
                particleBaseRadius
            ], 
            device=device
        )

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

    # 3. Particle-to-grid transfer (APIC + stress)
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
    wp.launch(
        kernel=mpmRoutines.collideBounds,
        dim=mpm.gridDims,
        inputs=[
            mpm.grid_v_out,
            mpm.gridDims[0],
            mpm.gridDims[1],
            mpm.gridDims[2],
            mpm.boundFriction,
            mpm.boundaryPadding
        ],
        device=device
    )

    # wp.launch(
    #     kernel=mpmRoutines.collideBoundsAbsorbing,
    #     dim=mpm.gridDims,
    #     inputs=[
    #         mpm.grid_v_out,
    #         mpm.gridDims[0],
    #         mpm.gridDims[1],
    #         mpm.gridDims[2],
    #         mpm.boundaryPadding
    #     ],
    #     device=device
    # )

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

def xpbdSimulationStep(sim, mpm, xpbd, device):
    """
    Integrate particles using XPBD with gravity and handle collisions, sleeping, and swelling.
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

    # Sleep particles & path integral
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
            sim.dtxpbd
        ], 
        device=device
    )
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
            xpbd.particle_v_max
        ],
        outputs=[xpbd.particle_v_integrated],  # Update velocities only
        device=device,
    )
    # Commit integrated positions/velocities
    sim.particle_x.assign(xpbd.particle_x_integrated)
    sim.particle_v.assign(xpbd.particle_v_integrated)

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

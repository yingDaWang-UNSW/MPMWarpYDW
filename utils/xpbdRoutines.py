import warp as wp

@wp.kernel
def reset_mpm_contact_labels(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
):
    """
    Reset all MPM particles (materialLabel 0 or 1) back to materialLabel=1.
    This is called before contact detection so that particles not in contact
    with XPBD can transition if they fail. Particles that ARE in contact will
    be set to 0 in the contact kernel.
    
    materialLabel convention:
    - 0: MPM, in contact with XPBD (cannot transition)
    - 1: MPM, not in contact (can transition to XPBD if damage >= 1)
    - 2: XPBD particle
    """
    tid = wp.tid()
    if activeLabel[tid] == 1 and materialLabel[tid] == 0:
        materialLabel[tid] = 1

@wp.kernel
def integrateParticlesXPBD(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
    v_max: float,
):
    tid = wp.tid()
    if activeLabel[tid] == 1:
        if materialLabel[tid] == 2:  # xpbd particles
            x0 = x[tid]
            v0 = v[tid]
            v1 = v0 + gravity * dt
            v1_mag = wp.length(v1)
            if v1_mag > v_max:
                v1 *= v_max / v1_mag
            x1 = x0 + v1 * dt
            # x[tid] = x1
            # v[tid] = v1
            x_new[tid] = x1
            v_new[tid] = v1
@wp.kernel
def my_solve_particle_bound_contacts(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    mu: float,
    staticGroundVelocityThresholdRatio: float,
    staticGroundFriction: float,
    minbounds: wp.vec3,  
    maxbounds: wp.vec3, 
    dt: float,
    relaxation: float,
    delta: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if activeLabel[tid] != 1:
        return
    if materialLabel[tid] != 2:
        return
    if mass[tid] == 0.0:
        return

    wi = 1.0 / mass[tid]
    x = particle_x[tid]
    v = particle_v[tid]
    r = particle_radius[tid]

    bounds_min = minbounds
    bounds_max = maxbounds

    for i in range(3):  # x, y, z
        # Lower bound face (normal = +i direction)
        n = wp.vec3(0.0)
        if i == 0:
            n = wp.vec3(1.0, 0.0, 0.0)
        elif i == 1:
            n = wp.vec3(0.0, 1.0, 0.0)
        elif i == 2:
            n = wp.vec3(0.0, 0.0, 1.0)

        c = wp.min(wp.dot(n, x - bounds_min) - r, 0.0)
        if c < 0.0:
            lambda_n = c
            delta_n = n * lambda_n
            vn = wp.dot(n, v)

            friction_mu = mu
            if wp.abs(vn) < staticGroundVelocityThresholdRatio * r:
                friction_mu = staticGroundFriction

            vt = v - n * vn
            vt_len = wp.length(vt)
            lambda_f = wp.max(friction_mu * lambda_n, -vt_len * dt)
            delta_f = wp.vec3(0.0)
            if vt_len > 1e-8:
                delta_f = wp.normalize(vt) * lambda_f

            wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation * wi)

        # Upper bound face (normal = -i direction)
        if i == 0:
            n = wp.vec3(-1.0, 0.0, 0.0)
        elif i == 1:
            n = wp.vec3(0.0, -1.0, 0.0)
        elif i == 2:
            n = wp.vec3(0.0, 0.0, -1.0)

        c = wp.min(wp.dot(n, x - bounds_max) - r, 0.0)
        if c < 0.0:
            lambda_n = c
            delta_n = n * lambda_n
            vn = wp.dot(n, v)

            friction_mu = mu
            if wp.abs(vn) < staticGroundVelocityThresholdRatio * r:
                friction_mu = staticGroundFriction

            vt = v - n * vn
            vt_len = wp.length(vt)
            lambda_f = wp.max(friction_mu * lambda_n, -vt_len * dt)
            delta_f = wp.vec3(0.0)
            if vt_len > 1e-8:
                delta_f = wp.normalize(vt) * lambda_f

            wp.atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation * wi)

@wp.kernel
def my_solve_particle_particle_contacts(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    k_mu: float,
    staticParticleVelocityThresholdRatio: float,
    staticParticleFriction: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    mpm_contact_transition_lock: int,
    deltas: wp.array(dtype=wp.vec3),
):
    """
    Solve particle-particle contacts using XPBD.
    
    Also handles MPM-XPBD contact detection (if mpm_contact_transition_lock == 1):
    - If an MPM particle (materialLabel <= 1) contacts an XPBD particle (materialLabel == 2),
      the MPM particle's materialLabel is set to 0 (preventing phase transition).
    - MPM-MPM collisions are skipped (handled by continuum mechanics).
    """
    tid = wp.tid()
    if activeLabel[tid]==1:
        if materialLabel[tid] >= 0:
            i = wp.hash_grid_point_id(grid, tid)
            if i == -1:
                return

            x = particle_x[i]
            v = particle_v[i]
            radius = particle_radius[i]
            if particle_mass[tid] == 0.0:
                return
            w1 = 1.0/particle_mass[i]
            query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion*radius)
            index = int(0)
            delta = wp.vec3(0.0)
            alpha = 0.0 # XPBD compliance factor
            while wp.hash_grid_query_next(query, index):
                if activeLabel[index]==1 and index != i and particle_mass[index] > 0.0:
                    # Skip MPM-MPM collisions (materialLabel<=1 for both)
                    if materialLabel[i] <= 1 and materialLabel[index] <= 1:
                        continue

                    n = x - particle_x[index]
                    d = wp.length(n)
                    err = d - radius - particle_radius[index]
                    w2 = 1.0/particle_mass[index]
                    denom = w1 + w2
                    # if err <= (radius + particle_radius[index])*0.5 and denom > 0.0:
                    if err <= k_cohesion and denom > 0.0:
                        # Check for MPM-XPBD contact and set materialLabel=0 for MPM particle
                        # This prevents the MPM particle from transitioning to XPBD while in contact
                        if mpm_contact_transition_lock == 1:
                            if materialLabel[i] <= 1 and materialLabel[index] == 2:
                                # i is MPM, index is XPBD -> lock i
                                materialLabel[i] = 0
                            if materialLabel[i] == 2 and materialLabel[index] <= 1:
                                # i is XPBD, index is MPM -> lock index
                                materialLabel[index] = 0

                    if err <= k_cohesion and denom > 0.0:
                        n = n / d
                        vrel = v - particle_v[index]
                        lambda_n = err/ (1.0 + (alpha * dt * dt))
                        delta_n = n * lambda_n
                        vn = wp.dot(n, vrel)
                        if wp.abs(vn)<staticParticleVelocityThresholdRatio*radius:
                            k_mu=staticParticleFriction
                        vt = vrel - n * vn
                        lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                        delta_f = wp.normalize(vt) * lambda_f
                        delta += (delta_f - delta_n) / denom * w1
                        # if particle_radius[index] < maxrad*0.95:
                        #     delta = delta * particle_radius[index]/maxrad
            wp.atomic_add(deltas, i, delta * relaxation)

@wp.kernel
def my_apply_particle_deltas(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    x_orig: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
    delta: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if activeLabel[tid]==1:
        x0 = x_orig[tid]
        xp = x_pred[tid]
        d = delta[tid]
        # deal with exchange nans
        if wp.isnan(d):
            d[0]=0.0
            d[1]=0.0
            d[2]=0.0
        
        # suppress deltas from excessive overlaps
        # if wp.length(d)>radius[tid]*0.01:
        #     d=d/wp.length(d)*radius[tid]*0.01
        # delta[tid]=d

        x_new = xp + d
        v_new = (x_new - x0) / dt

        v_new_mag = wp.length(v_new)
        if v_new_mag > v_max:
            v_new *= v_max / v_new_mag
        # if materialLabel[tid] == 2:
        x_out[tid] = x_new
        v_out[tid] = v_new

@wp.kernel
def apply_mpm_contact_impulses(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    x_orig: wp.array(dtype=wp.vec3),
    v_orig: wp.array(dtype=wp.vec3),
    x_final: wp.array(dtype=wp.vec3),
    v_final: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    coupling_strength: float,
    xpbd_only: int,
    v_out: wp.array(dtype=wp.vec3),
):
    """
    Post-convergence treatment for MPM particles:
    Convert the converged XPBD position change into a velocity impulse
    without actually changing the MPM particle position (MPM solver controls position).
    
    Args:
        coupling_strength: Empirically tuned factor (typically 0.2-0.3) that scales
                          XPBD position correction to MPM velocity impulse.
                          Related to soft coupling stiffness and stability.
                          Lower = softer coupling, higher = stiffer (may be unstable).
        xpbd_only: If 1, only reset position but don't change velocity (MPM frozen).
    
    This preserves:
    - XPBD convergence during iterations (all particles move)
    - MPM position authority (grid solver controls position)
    - Contact impulse transfer (velocity correction applied)
    """
    tid = wp.tid()
    if activeLabel[tid] == 1 and materialLabel[tid] <= 1:  # MPM particles only
        # Always reset position - MPM grid controls position
        x_final[tid] = x_orig[tid]
        
        # Only apply velocity impulse if not in xpbd_only mode
        if xpbd_only == 0:
            # MPM velocity comes from grid (already has gravity applied)
            # We only add the contact impulse from XPBD position correction
            
            # v_orig = MPM velocity from grid (includes gravity already)
            # x_final - x_orig = contact-induced position change from XPBD
            
            v_from_grid = v_orig[tid]
            contact_impulse = (x_final[tid] - x_orig[tid]) / dt * coupling_strength
            v_corrected = v_from_grid + contact_impulse
            v_mag = wp.length(v_corrected)
            if v_mag > v_max:
                v_corrected *= v_max / v_mag

            v_out[tid] = v_corrected

# TODO: in fs5, velocities arent zeroed on sleeping particles. in FS6, jury is still out.
@wp.kernel
def sleepParticles(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    sleepThreshold: float,
    radius: wp.array(dtype=float),
    particle_positions_prev: wp.array(dtype=wp.vec3),
    particle_positions_after: wp.array(dtype=wp.vec3),
    particle_distance_total: wp.array(dtype=float),
    dt: float,
    total_xpbd: wp.array(dtype=wp.int32),
    sleeping_xpbd: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if activeLabel[tid]==1:
        if materialLabel[tid] == 2:
            wp.atomic_add(total_xpbd, 0, 1)
            d=wp.length(particle_positions_after[tid]-particle_positions_prev[tid])
            if d/dt<sleepThreshold*radius[tid]:
                particle_positions_after[tid]=particle_positions_prev[tid]
                wp.atomic_add(sleeping_xpbd, 0, 1)
            particle_distance_total[tid]=particle_distance_total[tid]+d

@wp.kernel
def clipParticleVelocitiesOnPhaseChange(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    particle_distance_total: wp.array(dtype=float),
    particleInitialXPBDVelocity: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
):
    tid = wp.tid()
    if activeLabel[tid]==1:
        if materialLabel[tid] == 2:
            # if total distance from genertion less than 2 particles, clip
            if particle_distance_total[tid]<radius[tid]*2.0:
                if wp.length(particle_v[tid])>wp.length(particleInitialXPBDVelocity[tid]):
                    particle_v[tid] = particleInitialXPBDVelocity[tid]

            
@wp.kernel # type 2 swelling simply increments radius based on incremental distance and later, number of neighbours
def swellParticlesType2(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    particle_initial_position: wp.array(dtype=wp.vec3),
    particle_position: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    factor: float,
    factor_max: float,
    swellRadius: wp.array(dtype=float),
    baseRadius: wp.array(dtype=float),
):
    tid = wp.tid()
    if activeLabel[tid]==1:
        if materialLabel[tid] == 2:    # type 2 swelling operates on active, non dump particles (label 0 only, since -1 is inactive, 1, 2, and 5 are v1 swelling, 3 is inactive drawn, 4 is dump)
            dx=wp.length(particle_position[tid]-particle_initial_position[tid])*baseRadius[tid]
            factor=factor*baseRadius[tid]
            factor_max=factor_max*baseRadius[tid]
            
            if dx>factor:# and dx<factor_max:
                newRad=baseRadius[tid]+(swellRadius[tid]-baseRadius[tid])*(dx-factor)/(factor_max-factor)
                newRad=wp.min(wp.max(newRad,particle_radius[tid]),swellRadius[tid])
                particle_radius[tid]=newRad
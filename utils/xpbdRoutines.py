import warp as wp

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
    if activeLabel[tid] == 1: # both mpm and xpbd particles contribute to the grid
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
    deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if activeLabel[tid]==1:
        if materialLabel[tid] > 0:
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
                    n = x - particle_x[index]
                    d = wp.length(n)
                    err = d - radius - particle_radius[index]
                    w2 = 1.0/particle_mass[index]
                    denom = w1 + w2
                    if err <= k_cohesion and denom > 0.0:
                        n = n / d
                        vrel = v - particle_v[index]
                        lambda_n = err/ (1.0 + (alpha * dt * dt))
                        delta_n = n * lambda_n
                        vn = wp.dot(n, vrel)
                        if wp.abs(vn)<staticParticleVelocityThresholdRatio*radius:
                            k_mu=staticParticleFriction
                        vt = v - n * vn
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
        if materialLabel[tid] == 2:
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

            x_out[tid] = x_new
            v_out[tid] = v_new

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
):
    tid = wp.tid()
    if activeLabel[tid]==1:
        if materialLabel[tid] == 2:
            d=wp.length(particle_positions_after[tid]-particle_positions_prev[tid])
            if d/dt<sleepThreshold*radius[tid]:
                particle_positions_after[tid]=particle_positions_prev[tid]
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
            if particle_distance_total[tid]<radius[tid]*4.0:
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
import warp as wp

@wp.kernel
def compute_mu_lam_bulk_from_E_nu(
    E:  wp.array(dtype=float),
    nu:  wp.array(dtype=float),
    mu:  wp.array(dtype=float),
    lam:  wp.array(dtype=float),
    bulk:  wp.array(dtype=float),
    ):
    # Compute the Lame parameters from Young's modulus and Poisson's ratio
    # E: Young's modulus
    # nu: Poisson's ratio
    # mu: Shear modulus
    # lam: First Lame parameter
    # bulk: Bulk modulus
    # Note: This kernel assumes that E and nu are defined for each particle
    #       and that they are of the same length as the output arrays mu, lam,
    #       and bulk.
    p = wp.tid()
    mu[p] = E[p] / (2.0 * (1.0 + nu[p]))
    lam[p] = E[p] * nu[p]  / ((1.0 + nu[p]) * (1.0 - 2.0 * nu[p]))
    bulk[p] = lam[p] + 2./3. * mu[p]


@wp.kernel
def compute_stress_from_F_trial(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    particle_F: wp.array(dtype=wp.mat33),
    particle_F_trial: wp.array(dtype=wp.mat33),
    mu:  wp.array(dtype=float),
    lam:  wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    hardening: wp.array(dtype=wp.int32),
    xi:  wp.array(dtype=float),
    softening: wp.array(dtype=float),
    particle_stress: wp.array(dtype=wp.mat33), 
):
    p = wp.tid()
    # apply return mapping on mpm active particles
    if materialLabel[p] == 1:
        if activeLabel[p] == 1:
            particle_F[p] = von_mises_return_mapping_with_damage_YDW(
                particle_F_trial[p], 
                materialLabel,
                mu,
                lam,
                yield_stress,
                hardening,
                xi,
                softening, 
                p
            )

            # compute stress here
            J = wp.determinant(particle_F[p])
            U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            sig = wp.vec3(0.0)
            stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            wp.svd3(particle_F[p], U, sig, V)

            stress = kirchoff_stress_drucker_prager(particle_F[p], U, V, sig, mu[p], lam[p])

            stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
            particle_stress[p] = stress

@wp.func
def von_mises_return_mapping_with_damage_YDW(
    F_trial: wp.mat33, 
    materialLabel: wp.array(dtype=wp.int32),
    mu:  wp.array(dtype=float),
    lam:  wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    hardening: wp.array(dtype=wp.int32),
    xi:  wp.array(dtype=float),
    softening: wp.array(dtype=float),
    p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0) 
    wp.svd3(F_trial, U, sig_old, V) 

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2])) #log of the trace of the stress tensor
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * mu[p] * epsilon + lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > yield_stress[p]:
        if wp.length(cond) > yield_stress[p]*1.25:
            materialLabel[p] = 2
            # model.mu[p] = model.mu[p]*0.01
            # model.lam[p] = model.lam[p]*0.01
        if yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - yield_stress[p] / (2.0 * mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_stress[p] = yield_stress[p] - softening[p] * wp.length((delta_gamma / epsilon_hat_norm) * epsilon_hat)
        if yield_stress[p] <= 0:
            mu[p] = 0.0
            lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if hardening[p] == 1:
            yield_stress[p] = (
                yield_stress[p] + 2.0 * mu[p] * xi[p] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig0 = wp.max(sig[0], 1e-6)
    sig1 = wp.max(sig[1], 1e-6)
    sig2 = wp.max(sig[2], 1e-6)

    log_sig_sum = wp.log(sig0) + wp.log(sig1) + wp.log(sig2)

    center00 = 2.0 * mu * wp.log(sig0) * (1.0 / sig0) + lam * log_sig_sum * (1.0 / sig0)
    center11 = 2.0 * mu * wp.log(sig1) * (1.0 / sig1) + lam * log_sig_sum * (1.0 / sig1)
    center22 = 2.0 * mu * wp.log(sig2) * (1.0 / sig2) + lam * log_sig_sum * (1.0 / sig2)

    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def compute_dweight(
    inv_dx: float, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * inv_dx

@wp.kernel
def p2g_apic_with_stress(
    activeLabel: wp.array(dtype=wp.int32),
    materialLabel: wp.array(dtype=wp.int32),
    particle_stress: wp.array(dtype=wp.mat33),
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_C: wp.array(dtype=wp.mat33),
    particle_vol: wp.array(dtype=float),
    particle_mass: wp.array(dtype=float),
    dx: float,
    inv_dx: float,
    minBounds: wp.vec3,
    rpic_damping: float,
    grid_m: wp.array(dtype=float, ndim=3),
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3),
    dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    if activeLabel[p] == 1: # all materials contribute to the grid so that materials that rely on the grid can interact with other materials
        
        # xpbd particles contribute to the grid, but their velocities need to be clipped to prevent excessive stress at the interface upon phase change


        material_id = materialLabel[p]  # Get the material ID of the current particle
        stress = particle_stress[p]
        grid_pos = (particle_x[p]-minBounds) * inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(inv_dx, w, dw, i, j, k)
                    C = particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - rpic_damping) * C + rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )
                    if rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    elastic_force = -particle_vol[p] * stress * dweight
                    v_in_add = (
                        weight
                        * particle_mass[p]
                        * (particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )
                    wp.atomic_add(grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        grid_m, ix, iy, iz, weight * particle_mass[p]
                    )


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def set_value_to_int_array(target_array: wp.array(dtype=wp.int32), value: int):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]

# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    grid_m: wp.array(dtype=float, ndim=3),
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3),
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3),
    gravity: wp.vec3,
    dt: float):

    grid_x, grid_y, grid_z = wp.tid()
    if grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = grid_v_in[grid_x, grid_y, grid_z] * (1.0 / grid_m[grid_x, grid_y, grid_z])
        # add gravity
        v_out = v_out + dt * gravity# * wp.min(t/10.0,1.0)
        grid_v_out[grid_x, grid_y, grid_z] = v_out

@wp.kernel
def add_damping_via_grid(    
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3),
    scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    grid_v_out[grid_x, grid_y, grid_z] = (grid_v_out[grid_x, grid_y, grid_z] * scale)



@wp.kernel
def g2p(dt: float,
        activeLabel: wp.array(dtype=wp.int32),
        materialLabel: wp.array(dtype=wp.int32),
        particle_x: wp.array(dtype=wp.vec3),
        particle_v: wp.array(dtype=wp.vec3),
        particle_C: wp.array(dtype=wp.mat33),
        particle_F: wp.array(dtype=wp.mat33),
        particle_F_trial: wp.array(dtype=wp.mat33),
        particle_cov: wp.array(dtype=float),
        inv_dx: float,
        grid_v_out: wp.array(dtype=wp.vec3, ndim=3),
        update_cov_with_F: bool,
        minBounds: wp.vec3,
        ):
    p = wp.tid()
    if activeLabel[p] == 1: # both mpm and xpbd particles contribute to the grid
        if materialLabel[p] == 1:  # only mpm particles need grid information
            grid_pos = (particle_x[p]-minBounds) * inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
            new_v = wp.vec3(0.0, 0.0, 0.0)
            new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                        grid_v = grid_v_out[ix, iy, iz]
                        new_v = new_v + grid_v * weight
                        new_C = new_C + wp.outer(grid_v, dpos) * (
                            weight * inv_dx * 4.0
                        )
                        dweight = compute_dweight(inv_dx, w, dw, i, j, k)
                        new_F = new_F + wp.outer(grid_v, dweight)

            particle_v[p] = new_v
            particle_x[p] = particle_x[p] + dt * new_v
            particle_C[p] = new_C
            I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            F_tmp = (I33 + new_F * dt) * particle_F[p]
            particle_F_trial[p] = F_tmp

            if update_cov_with_F:
                update_cov(particle_cov, p, new_F, dt)


@wp.func
def update_cov(particle_cov: wp.array(dtype=float),
               p: int, 
               grad_v: wp.mat33, 
               dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = particle_cov[p * 6]
    cov_n[0, 1] = particle_cov[p * 6 + 1]
    cov_n[0, 2] = particle_cov[p * 6 + 2]
    cov_n[1, 0] = particle_cov[p * 6 + 1]
    cov_n[1, 1] = particle_cov[p * 6 + 3]
    cov_n[1, 2] = particle_cov[p * 6 + 4]
    cov_n[2, 0] = particle_cov[p * 6 + 2]
    cov_n[2, 1] = particle_cov[p * 6 + 4]
    cov_n[2, 2] = particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    particle_cov[p * 6] = cov_np1[0, 0]
    particle_cov[p * 6 + 1] = cov_np1[0, 1]
    particle_cov[p * 6 + 2] = cov_np1[0, 2]
    particle_cov[p * 6 + 3] = cov_np1[1, 1]
    particle_cov[p * 6 + 4] = cov_np1[1, 2]
    particle_cov[p * 6 + 5] = cov_np1[2, 2]

# the domain is bounded by a box
@wp.func
def apply_coulomb_friction(v: float, friction_force: float) -> float:
    s = wp.sign(v)
    return s * wp.max(wp.abs(v) - friction_force, 0.0)

@wp.kernel
def collideBounds(
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3),
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    friction_force: float,  # e.g., mu * g * dt
):
    grid_x, grid_y, grid_z = wp.tid()
    padding = 3
    v = grid_v_out[grid_x, grid_y, grid_z]

    if grid_x < padding and v[0] < 0.0:
        grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
            0.0,
            apply_coulomb_friction(v[1], friction_force),
            apply_coulomb_friction(v[2], friction_force),
        )
    if grid_x >= grid_dim_x - padding and v[0] > 0.0:
        grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
            0.0,
            apply_coulomb_friction(v[1], friction_force),
            apply_coulomb_friction(v[2], friction_force),
        )

    if grid_y < padding and v[1] < 0.0:
        grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
            apply_coulomb_friction(v[0], friction_force),
            0.0,
            apply_coulomb_friction(v[2], friction_force),
        )
    if grid_y >= grid_dim_y - padding and v[1] > 0.0:
        grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
            apply_coulomb_friction(v[0], friction_force),
            0.0,
            apply_coulomb_friction(v[2], friction_force),
        )

    if grid_z < padding and v[2] < 0.0:
        grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
            apply_coulomb_friction(v[0], friction_force),
            apply_coulomb_friction(v[1], friction_force),
            0.0,
        )
    if grid_z >= grid_dim_z - padding and v[2] > 0.0:
        grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
            apply_coulomb_friction(v[0], friction_force),
            apply_coulomb_friction(v[1], friction_force),
            0.0,
        )

# routines to extract rotation and covariance from deformation gradient
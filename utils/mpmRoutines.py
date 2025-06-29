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
    if materialLabel[p] == 1 and activeLabel[p] == 1:
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

        stress = kirchoff_stress_drucker_prager(
            particle_F[p], U, V, sig, mu[p], lam[p]
        )

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
    # This function implements the return mapping algorithm for von Mises plasticity with damage.
    # It computes the trial stress, checks if it exceeds the yield stress,
    # and if so, applies the return mapping to compute the elastic deformation gradient.
    # If the yield stress is exceeded, it also applies damage to the material properties.
    # The function returns the elastic deformation gradient if plasticity occurs, otherwise returns the trial deformation gradient.
    # Note: This function assumes that the material properties (mu, lam, yield_stress) are already set in the model.
    # The function also assumes that the model has a 'softening' parameter that controls the damage.
    # If the yield stress is exceeded, the yield stress is reduced by a softening factor.
    # If the yield stress becomes zero or negative, the material is considered fully damaged and mu and lam are set to zero.

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
    sig0 = wp.max(sig[0], 1e-16)
    sig1 = wp.max(sig[1], 1e-16)
    sig2 = wp.max(sig[2], 1e-16)

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
    if activeLabel[p] == 1:
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
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]
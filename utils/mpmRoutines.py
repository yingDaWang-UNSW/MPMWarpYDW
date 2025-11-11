import warp as wp

@wp.kernel
def creep_by_damage_with_baseline(
    materialLabel: wp.array(dtype=wp.int32),
    damage: wp.array(dtype=float),
    ys: wp.array(dtype=float),             # current yield stress
    ys_base: wp.array(dtype=float),        # reference yield stress
    dt: float,
    base_creep: float,                     # A_base: baseline rate
    damage_creep: float,                   # A_damage: damage-amplified rate
    damage_exponent: float                 # beta: exponent on damage
):
    p = wp.tid()
    if materialLabel[p] != 1:
        return

    D = damage[p]
    creep_rate = base_creep + damage_creep * wp.pow(D, damage_exponent)
    decay = creep_rate * ys_base[p] * dt
    ys[p] = wp.max(ys_base[p] * 0.2, ys[p] - decay)  # floor at 20% of original

@wp.kernel
def initialize_geostatic_stress(
    particle_x: wp.array(dtype=wp.vec3),
    particle_stress: wp.array(dtype=wp.mat33),
    density: wp.array(dtype=float),
    gravity: float,
    z_top: float,
    K0: float
):
    """
    DEPRECATED: This initializes stress but it gets overwritten by compute_stress_from_F_trial.
    Use initialize_geostatic_F instead to modify the deformation gradient.
    """
    tid = wp.tid()
    z = particle_x[tid][2]
    depth = z_top - z  # distance below the top surface
    sig_zz = density[tid] * gravity * depth
    sig_xx = K0 * sig_zz
    sig_yy = K0 * sig_zz

    # Fill stress tensor (compressive is positive here)
    particle_stress[tid] = wp.mat33(
        sig_xx, 0.0, 0.0,
        0.0, sig_yy, 0.0,
        0.0, 0.0, sig_zz
    )


@wp.kernel
def initialize_geostatic_F(
    particle_x: wp.array(dtype=wp.vec3),
    particle_F: wp.array(dtype=wp.mat33),
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    density: wp.array(dtype=float),
    gravity: float,
    z_top: float,
    K0: float
):
    """
    Initialize deformation gradient F to represent geostatic stress state.
    
    For an isotropic elastic material under geostatic stress:
    σ_v = ρgh (vertical)
    σ_h = K0 * σ_v (horizontal)
    
    We compute the strain that would produce this stress, then F = exp(ε).
    This way, the material "remembers" it's prestressed.
    """
    tid = wp.tid()
    z = particle_x[tid][2]
    depth = z_top - z
    
    # Target stresses (Kirchhoff stress for small strain ≈ Cauchy stress)
    # Compression = negative, so σ_v = -ρgh (negative at depth)
    sigma_v = -density[tid] * gravity * depth
    sigma_h = K0 * sigma_v
    
    # For linear elasticity: σ = λ*tr(ε)*I + 2μ*ε
    # Solve for strains given target stresses (σ_xx = σ_yy = σ_h, σ_zz = σ_v)
    # σ_h = λ*(ε_xx + ε_yy + ε_zz) + 2μ*ε_xx
    # σ_h = λ*(ε_xx + ε_yy + ε_zz) + 2μ*ε_yy
    # σ_v = λ*(ε_xx + ε_yy + ε_zz) + 2μ*ε_zz
    
    # Assuming ε_xx = ε_yy (horizontal isotropy):
    # σ_h = λ*(2ε_h + ε_v) + 2μ*ε_h
    # σ_v = λ*(2ε_h + ε_v) + 2μ*ε_v
    
    # Solve this 2x2 system:
    mu_p = mu[tid]
    lam_p = lam[tid]
    
    # Coefficient matrix for [ε_h, ε_v]:
    # [2λ + 2μ,    λ   ] [ε_h]   [σ_h]
    # [2λ,       λ + 2μ] [ε_v] = [σ_v]
    
    a11 = 2.0 * lam_p + 2.0 * mu_p
    a12 = lam_p
    a21 = 2.0 * lam_p
    a22 = lam_p + 2.0 * mu_p
    
    det = a11 * a22 - a12 * a21
    
    if wp.abs(det) > 1e-12:
        # Cramer's rule
        eps_h = (sigma_h * a22 - sigma_v * a12) / det
        eps_v = (a11 * sigma_v - a21 * sigma_h) / det
    else:
        eps_h = 0.0
        eps_v = 0.0
    
    # Logarithmic strain (since we use multiplicative elastoplasticity)
    # F = exp(ε) for small strain, or for diagonal ε:
    F_xx = wp.exp(eps_h)
    F_yy = wp.exp(eps_h)
    F_zz = wp.exp(eps_v)
    
    particle_F[tid] = wp.mat33(
        F_xx, 0.0, 0.0,
        0.0, F_yy, 0.0,
        0.0, 0.0, F_zz
    )


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
    particlePosition: wp.array(dtype=wp.vec3),
    particleVelocity: wp.array(dtype=wp.vec3),
    initialPhaseChangePosition: wp.array(dtype=wp.vec3),
    initialPhaseChangeVelocity: wp.array(dtype=wp.vec3),
    particle_F: wp.array(dtype=wp.mat33),
    particle_F_trial: wp.array(dtype=wp.mat33),
    mu:  wp.array(dtype=float),
    lam:  wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    hardening:  wp.array(dtype=float),
    softening: wp.array(dtype=float),
    particle_density: wp.array(dtype=float),
    strainCriteria: wp.array(dtype=float),
    efficiency: float,
    eta_shear: wp.array(dtype=float),
    eta_bulk: wp.array(dtype=float),
    particle_C: wp.array(dtype=wp.mat33),
    particle_stress: wp.array(dtype=wp.mat33), 
    particle_accumulated_strain: wp.array(dtype=float),
    particle_damage: wp.array(dtype=float),
    constitutive_model: int,
    alpha: wp.array(dtype=float),
):
    p = wp.tid()
    # apply return mapping on mpm active particles
    if materialLabel[p] == 1:
        if activeLabel[p] == 1:
            # Choose constitutive model: 0=Von Mises, 1=Drucker-Prager
            if constitutive_model == 0:
                particle_F[p] = von_mises_return_mapping_with_damage_YDW(
                    particle_F_trial[p], 
                    particlePosition,
                    particleVelocity,
                    initialPhaseChangePosition,
                    initialPhaseChangeVelocity,
                    materialLabel,
                    particle_accumulated_strain,
                    particle_damage,
                    mu,
                    lam,
                    yield_stress,
                    hardening,
                    softening, 
                    particle_density,
                    strainCriteria,
                    efficiency,
                    p
                )
            else:  # Drucker-Prager
                particle_F[p] = drucker_prager_return_mapping(
                    particle_F_trial[p], 
                    particlePosition,
                    particleVelocity,
                    initialPhaseChangePosition,
                    initialPhaseChangeVelocity,
                    materialLabel,
                    particle_accumulated_strain,
                    particle_damage,
                    mu,
                    lam,
                    yield_stress,
                    alpha,
                    hardening,
                    softening, 
                    particle_density,
                    strainCriteria,
                    efficiency,
                    p
                )

            # compute stress here
            J = wp.determinant(particle_F[p])
            U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            sig = wp.vec3(0.0)
            stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            wp.svd3(particle_F[p], U, sig, V)

            stress = kirchoff_stress(particle_F[p], U, V, sig, mu[p], lam[p])

            stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry

            # NOTE: Gravitational stress is NOT added here because:
            # 1. It would not be persistent (F tracks deformation from reference, not absolute stress)
            # 2. Instead, we initialize geostatic stress at t=0 (via initialize_geostatic_stress)
            # 3. And apply gravity as body force in grid update to maintain equilibrium
            
            # === Kelvin-Voigt damping ===
            #TODO: eta should scale with damage, can add rayleigh high and low frequency damping
            # Get symmetric part of velocity gradient
            grad_v = particle_C[p]  # particle_C already stores gradv (from G2P)
            strain_rate = 0.5 * (grad_v + wp.transpose(grad_v))
            volumetric_rate = (strain_rate[0, 0] + strain_rate[1, 1] + strain_rate[2, 2])

            # Deviatoric part (trace-free)
            identity = wp.mat33(1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0)
            strain_rate_dev = strain_rate - (volumetric_rate / 3.0) * identity

            viscous_stress = (2.0 * eta_shear[p]) * strain_rate_dev + eta_bulk[p] * volumetric_rate * identity

            # Add to Kirchhoff stress
            stress = stress + viscous_stress

            particle_stress[p] = stress


@wp.func
def von_mises_return_mapping_with_damage_YDW(
    F_trial: wp.mat33, 
    particlePosition: wp.array(dtype=wp.vec3),
    particleVelocity: wp.array(dtype=wp.vec3),
    initialPhaseChangePosition: wp.array(dtype=wp.vec3),
    initialPhaseChangeVelocity: wp.array(dtype=wp.vec3),
    materialLabel: wp.array(dtype=wp.int32),
    particle_accumulated_strain: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    mu:  wp.array(dtype=float),
    lam:  wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    hardening:  wp.array(dtype=float),
    softening: wp.array(dtype=float),
    density: wp.array(dtype=float),
    strainCriteria: wp.array(dtype=float),
    efficiency: float,
    p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0) 
    wp.svd3(F_trial, U, sig_old, V) 

    sig = wp.vec3(wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01))  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    mean_eps = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0
    epsilon_dev = epsilon - wp.vec3(mean_eps)

    # Trial deviatoric stress (effective)
    tau = 2.0 * mu[p] * epsilon + lam[p] * (epsilon[0] + epsilon[1] + epsilon[2]) * wp.vec3(1.0)
    p_stress = (tau[0] + tau[1] + tau[2]) / 3.0
    tau_dev = tau - p_stress * wp.vec3(1.0)

    # Modified yield stress due to damage
    D = damage[p]
    sigma_eq = wp.length(tau_dev)
    yield_eff = (1.0 - D) * yield_stress[p]

    if sigma_eq > yield_eff:
        epsilon_dev_norm = wp.length(epsilon_dev) + 1e-6

        delta_gamma = epsilon_dev_norm - yield_eff / (2.0 * mu[p])
        # Accumulate plastic strain
        plastic_strain_increment = wp.sqrt(2.0 / 3.0) * delta_gamma
        particle_accumulated_strain[p] += plastic_strain_increment

        # === Damage evolution ===
        if strainCriteria[p] > 0.0:
            dD = plastic_strain_increment / strainCriteria[p]
            damage[p] = wp.min(1.0, damage[p] + dD)
            D = damage[p]  # Update effective value
            
        # === Phase transition ===
        if damage[p] >= 1.0:
            materialLabel[p] = 2
            initialPhaseChangePosition[p] = particlePosition[p]
            # estimate the velocity from energy release
            # Convert deviatoric stress magnitude to velocity via elastic energy estimate #TODO: this can be computed from components directly rather than estimated
            rho = density[p]

            # sigma_vm = wp.length(cond)  # von Mises-like
            # E = youngs_modulus[p]
            # v_expected = wp.sqrt(efficiency * sigma_vm * sigma_vm / (rho * E))

            # Compute elastic strain energy density: u = 0.5 * sigma : epsilon
            u = 0.5 * (tau[0]*epsilon[0] + tau[1]*epsilon[1] + tau[2]*epsilon[2])

            # Energy-to-velocity conversion
            v_expected = wp.sqrt(2.0 * efficiency * u / rho)
            
            # Add in the direction of current motion
            v_dir = wp.normalize(particleVelocity[p] + wp.vec3(1e-12))  # avoid divide by zero
            particleVelocity[p] = particleVelocity[p] + v_expected * v_dir
            initialPhaseChangeVelocity[p] = particleVelocity[p]
            # initialPhaseChangeVelocity[p][0]=0.0
            # initialPhaseChangeVelocity[p][1]=0.0
            # initialPhaseChangeVelocity[p][2]=0.0
            
            return F_trial
        
        # Plastic correction
        epsilon = epsilon - (delta_gamma / epsilon_dev_norm) * epsilon_dev

        # Update yield stress (softening or hardening)
        # dimensionless knobs: hardening[p] = \bar H, softening[p] = \bar S
        dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
        yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)

        # fail the material
        if yield_stress[p] < 0.0:
            yield_stress[p] = 0.0
            mu[p] = 0.0
            lam[p] = 0.0

        # Reconstruct elastic part
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]), 0.0, 0.0,
            0.0, wp.exp(epsilon[1]), 0.0,
            0.0, 0.0, wp.exp(epsilon[2])
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic

    else:
        return F_trial


@wp.func
def drucker_prager_return_mapping(
    F_trial: wp.mat33, 
    particlePosition: wp.array(dtype=wp.vec3),
    particleVelocity: wp.array(dtype=wp.vec3),
    initialPhaseChangePosition: wp.array(dtype=wp.vec3),
    initialPhaseChangeVelocity: wp.array(dtype=wp.vec3),
    materialLabel: wp.array(dtype=wp.int32),
    particle_accumulated_strain: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    mu:  wp.array(dtype=float),
    lam:  wp.array(dtype=float),
    yield_stress: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    hardening:  wp.array(dtype=float),
    softening: wp.array(dtype=float),
    density: wp.array(dtype=float),
    strainCriteria: wp.array(dtype=float),
    efficiency: float,
    p: int
):
    
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0) 
    wp.svd3(F_trial, U, sig_old, V) 

    sig = wp.vec3(wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01))  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    mean_eps = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0
    epsilon_dev = epsilon - wp.vec3(mean_eps)

    # Trial deviatoric stress (effective)
    tau = 2.0 * mu[p] * epsilon + lam[p] * (epsilon[0] + epsilon[1] + epsilon[2]) * wp.vec3(1.0)
    p_stress = (tau[0] + tau[1] + tau[2]) / 3.0
    tau_dev = tau - p_stress * wp.vec3(1.0)

    # Modified yield stress due to damage
    D = damage[p]
    sigma_eq = wp.length(tau_dev)
    # Add pressure effect to yield stress (Drucker-Prager-like)
    # α controls pressure sensitivity (friction angle effect)
    # Sign convention: compression = negative, so use MINUS to increase strength
    # - Compression (p < 0): -α·p > 0 → increases yield stress
    # - Tension (p > 0):     -α·p < 0 → decreases yield stress
    # If α = 0, behaves like standard Von Mises
    yield_eff = (1.0 - D) * (yield_stress[p] - alpha[p] * p_stress)

    if sigma_eq > yield_eff:
        epsilon_dev_norm = wp.length(epsilon_dev) + 1e-6

        delta_gamma = epsilon_dev_norm - yield_eff / (2.0 * mu[p])
        # Accumulate plastic strain
        plastic_strain_increment = wp.sqrt(2.0 / 3.0) * delta_gamma
        particle_accumulated_strain[p] += plastic_strain_increment

        # === Damage evolution ===
        if strainCriteria[p] > 0.0:
            dD = plastic_strain_increment / strainCriteria[p]
            damage[p] = wp.min(1.0, damage[p] + dD)
            D = damage[p]  # Update effective value
            
        # === Phase transition ===
        if damage[p] >= 1.0:
            materialLabel[p] = 2
            initialPhaseChangePosition[p] = particlePosition[p]
            # estimate the velocity from energy release
            # Convert deviatoric stress magnitude to velocity via elastic energy estimate #TODO: this can be computed from components directly rather than estimated
            rho = density[p]

            # sigma_vm = wp.length(cond)  # von Mises-like
            # E = youngs_modulus[p]
            # v_expected = wp.sqrt(efficiency * sigma_vm * sigma_vm / (rho * E))

            # Compute elastic strain energy density: u = 0.5 * sigma : epsilon
            u = 0.5 * (tau[0]*epsilon[0] + tau[1]*epsilon[1] + tau[2]*epsilon[2])

            # Energy-to-velocity conversion
            v_expected = wp.sqrt(2.0 * efficiency * u / rho)
            
            # Add in the direction of current motion
            v_dir = wp.normalize(particleVelocity[p] + wp.vec3(1e-12))  # avoid divide by zero
            particleVelocity[p] = particleVelocity[p] + v_expected * v_dir
            initialPhaseChangeVelocity[p] = particleVelocity[p]
            # initialPhaseChangeVelocity[p][0]=0.0
            # initialPhaseChangeVelocity[p][1]=0.0
            # initialPhaseChangeVelocity[p][2]=0.0
            
            return F_trial
        
        # Plastic correction
        epsilon = epsilon - (delta_gamma / epsilon_dev_norm) * epsilon_dev

        # Update yield stress (softening or hardening)
        # dimensionless knobs: hardening[p] = \bar H, softening[p] = \bar S
        dsy = 2.0 * mu[p] * (hardening[p] - softening[p]) * delta_gamma
        yield_stress[p] = wp.max(0.0, yield_stress[p] + dsy)

        # fail the material
        if yield_stress[p] < 0.0:
            yield_stress[p] = 0.0
            mu[p] = 0.0
            lam[p] = 0.0

        # Reconstruct elastic part
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]), 0.0, 0.0,
            0.0, wp.exp(epsilon[1]), 0.0,
            0.0, 0.0, wp.exp(epsilon[2])
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic

    else:
        return F_trial




@wp.func
def drucker_prager_return_mapping_old(
    F_trial: wp.mat33,
    particlePosition: wp.array(dtype=wp.vec3),
    particleVelocity: wp.array(dtype=wp.vec3),
    initialPhaseChangePosition: wp.array(dtype=wp.vec3),
    initialPhaseChangeVelocity: wp.array(dtype=wp.vec3),
    materialLabel: wp.array(dtype=wp.int32),
    particle_accumulated_strain: wp.array(dtype=float),
    damage: wp.array(dtype=float),
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    cohesion: wp.array(dtype=float),
    friction_angle: wp.array(dtype=float),
    dilation_angle: wp.array(dtype=float),
    tension_cutoff: wp.array(dtype=float),
    hardening: wp.array(dtype=float),
    softening: wp.array(dtype=float),
    density: wp.array(dtype=float),
    strainCriteria: wp.array(dtype=float),
    efficiency: float,
    p: int
):
    """
    Simplified Drucker-Prager for rock caving:
    - Pressure-dependent yielding (cohesion + friction)
    - Non-associated flow (dilatancy)
    - Tension cutoff
    - Cohesion degradation with damage
    """
    # SVD decomposition
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)
    
    sig = wp.vec3(wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01))
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    
    # Trial stress
    tau = 2.0 * mu[p] * epsilon + lam[p] * (epsilon[0] + epsilon[1] + epsilon[2]) * wp.vec3(1.0)
    
    # Stress invariants (positive = compression)
    p_stress = (tau[0] + tau[1] + tau[2]) / 3.0
    tau_dev = tau - p_stress * wp.vec3(1.0)
    q = wp.sqrt(1.5) * wp.length(tau_dev)
    
    # Damage-dependent cohesion (degrades exponentially)
    D = damage[p]
    c_eff = cohesion[p] * (1.0 - D)  # Simple linear degradation
    
    # Drucker-Prager parameters
    phi = friction_angle[p]  # Already in radians
    psi = dilation_angle[p]  # Already in radians
    
    sin_phi = wp.sin(phi)
    cos_phi = wp.cos(phi)
    sin_psi = wp.sin(psi)
    
    # Plane strain approximation
    alpha = (2.0 * sin_phi) / (wp.sqrt(3.0) * (3.0 - sin_phi))
    k = (6.0 * c_eff * cos_phi) / (wp.sqrt(3.0) * (3.0 - sin_phi))
    beta = (2.0 * sin_psi) / (wp.sqrt(3.0) * (3.0 - sin_psi))
    
    # Tension cutoff
    sigma_t = tension_cutoff[p]
    if p_stress < -sigma_t:
        # Tensile failure
        damage[p] = 1.0
        materialLabel[p] = 2
        initialPhaseChangePosition[p] = particlePosition[p]
        
        rho = density[p]
        u = 0.5 * (tau[0] * epsilon[0] + tau[1] * epsilon[1] + tau[2] * epsilon[2])
        u = wp.max(u, 0.0)
        v_expected = wp.sqrt(2.0 * efficiency * u / rho)
        
        v_dir = wp.normalize(particleVelocity[p] + wp.vec3(1e-12))
        particleVelocity[p] = particleVelocity[p] + v_expected * v_dir
        initialPhaseChangeVelocity[p] = particleVelocity[p]
        
        return F_trial
    
    # Yield function: F = q + alpha*p - k
    F_yield = q + alpha * p_stress - k
    
    if F_yield > 0.0:
        # Return mapping
        bulk_modulus = lam[p] + 2.0 * mu[p] / 3.0
        denom = 3.0 * mu[p] + 9.0 * bulk_modulus * alpha * beta
        
        delta_gamma = F_yield / denom if denom > 1e-12 else 0.0
        
        # Plastic strain
        plastic_increment = delta_gamma * wp.sqrt(1.0 + beta * beta / 3.0)
        particle_accumulated_strain[p] += plastic_increment
        
        # Damage evolution
        if strainCriteria[p] > 0.0:
            dD = plastic_increment / strainCriteria[p]
            damage[p] = wp.min(1.0, damage[p] + dD)
        
        # Phase transition check
        if damage[p] >= 1.0:
            materialLabel[p] = 2
            initialPhaseChangePosition[p] = particlePosition[p]
            
            rho = density[p]
            u = 0.5 * (tau[0] * epsilon[0] + tau[1] * epsilon[1] + tau[2] * epsilon[2])
            u = wp.max(u, 0.0)
            v_expected = wp.sqrt(2.0 * efficiency * u / rho)
            
            v_dir = wp.normalize(particleVelocity[p] + wp.vec3(1e-12))
            particleVelocity[p] = particleVelocity[p] + v_expected * v_dir
            initialPhaseChangeVelocity[p] = particleVelocity[p]
            
            return F_trial
        
        # Update stresses
        p_stress_new = p_stress - 3.0 * bulk_modulus * beta * delta_gamma
        q_new = wp.max(q - 3.0 * mu[p] * delta_gamma, 0.0)
        
        # Reconstruct stress
        if q > 1e-12:
            tau_new = (q_new / q) * tau_dev + p_stress_new * wp.vec3(1.0)
        else:
            tau_new = p_stress_new * wp.vec3(1.0)
        
        # Back to strain
        trace_tau = tau_new[0] + tau_new[1] + tau_new[2]
        epsilon_new = (tau_new - lam[p] * trace_tau * wp.vec3(1.0/3.0)) / (2.0 * mu[p])
        
        # Reconstruct F
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]), 0.0, 0.0,
            0.0, wp.exp(epsilon_new[1]), 0.0,
            0.0, 0.0, wp.exp(epsilon_new[2])
        )
        return U * sig_elastic * wp.transpose(V)
    
    return F_trial


@wp.func
def kirchoff_stress(
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
def set_mat33_to_copy(source_array: wp.array(dtype=wp.mat33), target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = source_array[tid]


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
    if activeLabel[p] == 1:
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


@wp.kernel
def collideBoundsAbsorbing(
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3),
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int
):
    grid_x, grid_y, grid_z = wp.tid()
    padding = wp.float32(3.0)
    damping_inner = wp.float32(1)
    damping_outer = wp.float32(0.0)
    one = wp.float32(1.0)

    v = grid_v_out[grid_x, grid_y, grid_z]

    # -X boundary
    if grid_x < 3:
        layer_factor = (padding - wp.float32(grid_x)) / padding
        damping = damping_outer * layer_factor + damping_inner * (one - layer_factor)
        grid_v_out[grid_x, grid_y, grid_z] = v * damping

    # +X boundary
    if grid_x >= grid_dim_x - 3:
        layer_factor = (padding - wp.float32(grid_dim_x - 1 - grid_x)) / padding
        damping = damping_outer * layer_factor + damping_inner * (one - layer_factor)
        grid_v_out[grid_x, grid_y, grid_z] = v * damping

    # -Y boundary
    if grid_y < 3:
        layer_factor = (padding - wp.float32(grid_y)) / padding
        damping = damping_outer * layer_factor + damping_inner * (one - layer_factor)
        grid_v_out[grid_x, grid_y, grid_z] = v * damping

    # +Y boundary
    if grid_y >= grid_dim_y - 3:
        layer_factor = (padding - wp.float32(grid_dim_y - 1 - grid_y)) / padding
        damping = damping_outer * layer_factor + damping_inner * (one - layer_factor)
        grid_v_out[grid_x, grid_y, grid_z] = v * damping

    # -Z boundary
    if grid_z < 3:
        layer_factor = (padding - wp.float32(grid_z)) / padding
        damping = damping_outer * layer_factor + damping_inner * (one - layer_factor)
        grid_v_out[grid_x, grid_y, grid_z] = v * damping

    # +Z boundary
    if grid_z >= grid_dim_z - 3:
        layer_factor = (padding - wp.float32(grid_dim_z - 1 - grid_z)) / padding
        damping = damping_outer * layer_factor + damping_inner * (one - layer_factor)
        grid_v_out[grid_x, grid_y, grid_z] = v * damping

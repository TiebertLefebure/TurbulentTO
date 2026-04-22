from dolfin import (
    Constant,
    DOLFIN_EPS,
    dot,
    grad,
    inner,
    nabla_grad,
    skew,
    sqrt,
)


def _smooth_abs(value, smooth_abs_eps):
    return sqrt(value**2 + Constant(smooth_abs_eps))


def _smooth_positive(value, floor=0.0, smooth_abs_eps=None):
    floor_constant = Constant(float(floor))
    shifted_value = value - floor_constant
    if smooth_abs_eps is None:
        return 0.5 * (shifted_value + abs(shifted_value)) + floor_constant
    return 0.5 * (shifted_value + _smooth_abs(shifted_value, smooth_abs_eps)) + floor_constant


def sa_min(a, b, smooth_abs_eps=None):
    """UFL-safe min helper; optionally smoothed for differentiability."""
    if smooth_abs_eps is None:
        return (a + b - abs(a - b)) / Constant(2.0)
    return (a + b - _smooth_abs(a - b, smooth_abs_eps)) / Constant(2.0)


def sa_turbulent_viscosity(nu_tilde, nu_laminar, smooth_abs_eps=None):
    """SA eddy viscosity from nu_tilde."""
    chi = nu_tilde / (nu_laminar + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1) ** 3)
    nu_t_raw = nu_tilde * f_v1
    if smooth_abs_eps is None:
        return nu_t_raw
    return 0.5 * (nu_t_raw + _smooth_abs(nu_t_raw, smooth_abs_eps))


def sa_transport_terms(
    external_velocity,
    nu_tilde,
    nu_laminar,
    wall_distance,
    smooth_abs_eps=None,
    wall_distance_floor=0.0,
    nu_tilde_floor=0.0,
):
    """Build SA transport-model terms for a monolithic state residual."""
    sigma = Constant(2.0 / 3.0)
    cb1 = Constant(0.1355)
    cb2 = Constant(0.622)
    kappa = Constant(0.41)
    cw2 = Constant(0.3)
    cw3 = Constant(2.0)
    cw1 = cb1 / kappa**2 + (Constant(1.0) + cb2) / sigma

    nu_tilde_safe = _smooth_positive(
        nu_tilde,
        floor=nu_tilde_floor,
        smooth_abs_eps=smooth_abs_eps,
    )

    chi = nu_tilde_safe / (nu_laminar + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1) ** 3)
    f_v2 = Constant(1.0) - chi / (Constant(1.0) + chi * f_v1)
    # Yoon 2016 uses the no-ft2 SA form in Eqs. (10)-(15).
    f_t2 = Constant(0.0)

    omega_sq = Constant(2.0) * inner(
        skew(nabla_grad(external_velocity)),
        skew(nabla_grad(external_velocity)),
    )
    s_value = sqrt(omega_sq + DOLFIN_EPS)

    y_safe = wall_distance + Constant(max(float(wall_distance_floor), float(DOLFIN_EPS)))
    s_tilde = s_value + nu_tilde_safe / (kappa**2 * y_safe**2) * f_v2

    r_arg = nu_tilde_safe / (s_tilde * kappa**2 * y_safe**2 + DOLFIN_EPS)
    r_value = sa_min(r_arg, Constant(10.0), smooth_abs_eps=smooth_abs_eps)

    g_value = r_value + cw2 * (r_value**6 - r_value)
    f_w = g_value * (
        (Constant(1.0) + cw3**6) / (g_value**6 + cw3**6)
    ) ** (Constant(1.0) / Constant(6.0))

    prod_nt = cb1 * (Constant(1.0) - f_t2) * s_tilde * nu_tilde_safe
    react_nt = cw1 * f_w * (nu_tilde_safe / y_safe**2)
    cross_diff_nt = (cb2 / sigma) * inner(nabla_grad(nu_tilde), nabla_grad(nu_tilde))
    source_nt = prod_nt + cross_diff_nt
    return sigma, react_nt, source_nt, nu_tilde_safe


def build_spalart_allmaras_residual(
    external_velocity,
    nu_tilde,
    test_nu_tilde,
    nu_laminar,
    wall_distance,
    custom_dx,
    nu_tilde_penalty_reaction=None,
    smooth_abs_eps=None,
    wall_distance_floor=0.0,
    nu_tilde_floor=0.0,
):
    """Return the steady SA weak residual for the monolithic full solver."""
    sigma, react_nt, source_nt, nu_tilde_safe = sa_transport_terms(
        external_velocity,
        nu_tilde,
        nu_laminar,
        wall_distance,
        smooth_abs_eps=smooth_abs_eps,
        wall_distance_floor=wall_distance_floor,
        nu_tilde_floor=nu_tilde_floor,
    )
    penalty_react = (
        nu_tilde_penalty_reaction
        if nu_tilde_penalty_reaction is not None
        else Constant(0.0)
    )

    return (
        dot(external_velocity, nabla_grad(nu_tilde)) * test_nu_tilde * custom_dx
        + inner((nu_laminar + nu_tilde_safe) / sigma * grad(nu_tilde), grad(test_nu_tilde)) * custom_dx
        + (react_nt + penalty_react) * nu_tilde_safe * test_nu_tilde * custom_dx
        - source_nt * test_nu_tilde * custom_dx
    )

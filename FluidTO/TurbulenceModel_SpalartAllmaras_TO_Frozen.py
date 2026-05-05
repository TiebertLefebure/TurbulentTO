from dolfin import (
    CellDiameter, Constant, DOLFIN_EPS, Function, TestFunction, TrialFunction,
    assemble, dot, grad, inner, interpolate, lhs, nabla_grad, rhs, skew, solve, sqrt,
)


# -------------------------------------------------------------------------- #
# Spalart-Allmaras turbulence model  (adjustments for Topology Optimization) #
# -------------------------------------------------------------------------- #

def _smooth_abs(value, smooth_abs_eps):
    return sqrt(value**2 + Constant(smooth_abs_eps))


def _smooth_positive(value, floor=0.0, smooth_abs_eps=None):
    """Smooth floor for SA coefficients while leaving the unknown itself raw."""
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


def sa_turbulent_viscosity(nu_tilde, nu_laminar, smooth_abs_eps=None, nu_tilde_floor=0.0):
    """SA eddy viscosity from nu_tilde."""
    # Negative SA iterates are allowed numerically, but coefficients use the
    # floored value so the eddy-viscosity law stays physical.
    nu_tilde_safe = _smooth_positive(
        nu_tilde,
        floor=nu_tilde_floor,
        smooth_abs_eps=smooth_abs_eps,
    )
    chi = nu_tilde_safe / (nu_laminar + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1)**3)
    nu_t_raw = nu_tilde_safe * f_v1
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
    """Build SA transport-model terms shared across solvers."""
    sigma = Constant(2.0 / 3.0)
    cb1 = Constant(0.1355)
    cb2 = Constant(0.622)
    kappa = Constant(0.41)
    cw2 = Constant(0.3)
    cw3 = Constant(2.0)
    cw1 = cb1 / kappa**2 + (Constant(1.0) + cb2) / sigma

    # Production/destruction coefficients use nu_tilde_safe; the linear solve
    # still advances nu_tilde as the unknown.
    nu_tilde_safe = _smooth_positive(
        nu_tilde,
        floor=nu_tilde_floor,
        smooth_abs_eps=smooth_abs_eps,
    )

    chi = nu_tilde_safe / (nu_laminar + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1)**3)
    f_v2 = Constant(1.0) - chi / (Constant(1.0) + chi * f_v1)
    # Yoon 2016 uses the no-ft2 SA form in Eqs. (10)-(15).
    f_t2 = Constant(0.0)

    omega_sq = Constant(2.0) * inner(skew(nabla_grad(external_velocity)), skew(nabla_grad(external_velocity)))
    S = sqrt(omega_sq + DOLFIN_EPS)

    y_safe = wall_distance + Constant(max(float(wall_distance_floor), float(DOLFIN_EPS)))
    S_tilde = S + nu_tilde_safe / (kappa**2 * y_safe**2) * f_v2

    r_arg = nu_tilde_safe / (S_tilde * kappa**2 * y_safe**2 + DOLFIN_EPS)
    r = sa_min(r_arg, Constant(10.0), smooth_abs_eps=smooth_abs_eps)

    g = r + cw2 * (r**6 - r)
    f_w = g * ((Constant(1.0) + cw3**6) / (g**6 + cw3**6))**(Constant(1.0) / Constant(6.0))

    prod_nt = cb1 * (Constant(1.0) - f_t2) * S_tilde * nu_tilde_safe
    react_nt = cw1 * f_w * (nu_tilde_safe / y_safe**2)
    cross_diff_nt = (cb2 / sigma) * inner(nabla_grad(nu_tilde), nabla_grad(nu_tilde))
    source_nt = prod_nt + cross_diff_nt
    return sigma, react_nt, source_nt, nu_tilde_safe


class SpalartAllmarasSteadyState:
    def __init__(
        self,
        space,
        bcs,
        nu_tilde_init,
        nu_laminar,
        custom_dx,
        wall_distance,
        nu_tilde_penalty_reaction=None,
        wall_distance_floor=0.0,
        smooth_abs_eps=None,
        nu_tilde_floor=0.0,
        supg_stabilization=False,
        supg_tau_scale=1.0,
        pseudo_time_stabilization=False,
        pseudo_dt=1.0,
    ):
        self._space = space
        self._bcs = bcs
        self._nu_laminar = nu_laminar
        self._dx = custom_dx
        self._wall_distance = wall_distance
        self._nu_tilde_penalty_reaction = nu_tilde_penalty_reaction
        self._wall_distance_floor = float(wall_distance_floor)
        self._smooth_abs_eps = smooth_abs_eps
        self._nu_tilde_floor = float(nu_tilde_floor)
        self._supg_stabilization = bool(supg_stabilization)
        self._supg_tau_scale = float(supg_tau_scale)
        self._pseudo_time_stabilization = bool(pseudo_time_stabilization)
        self._pseudo_dt = float(pseudo_dt)
        if self._pseudo_time_stabilization and self._pseudo_dt <= 0.0:
            raise ValueError("SA pseudo-time step must be positive.")

        self._nu_tilde = TrialFunction(space)
        self._xi = TestFunction(space)
        self._nu_tilde1 = Function(space)
        self._nu_tilde0 = interpolate(Constant(nu_tilde_init), space)

    def construct_forms(self, external_velocity):
        sigma, react_nt, source_nt, nu_tilde_safe = sa_transport_terms(
            external_velocity,
            self._nu_tilde0,
            self._nu_laminar,
            self._wall_distance,
            smooth_abs_eps=self._smooth_abs_eps,
            wall_distance_floor=self._wall_distance_floor,
            nu_tilde_floor=self._nu_tilde_floor,
        )
        penalty_react = self._nu_tilde_penalty_reaction if self._nu_tilde_penalty_reaction is not None else Constant(0.0)

        # Steady-state SA transport equation (Yoon 2016 Eq. 27 for TO penalization).
        # Nonlinear coefficients are frozen from nu_tilde_safe.
        FNT = (
            dot(external_velocity, nabla_grad(self._nu_tilde)) * self._xi * self._dx
            + inner((self._nu_laminar + nu_tilde_safe) / sigma * grad(self._nu_tilde), grad(self._xi)) * self._dx
            + (react_nt + penalty_react) * self._nu_tilde * self._xi * self._dx
            - source_nt * self._xi * self._dx
        )
        if self._pseudo_time_stabilization:
            FNT += (
                (self._nu_tilde - self._nu_tilde0)
                / Constant(self._pseudo_dt)
                * self._xi
                * self._dx
            )
        if self._supg_stabilization:
            # Streamline stabilization for advection-dominated SA transport.
            h_cell = CellDiameter(self._space.mesh())
            speed = sqrt(inner(external_velocity, external_velocity) + DOLFIN_EPS)
            tau = Constant(self._supg_tau_scale) * h_cell / (
                Constant(2.0) * speed + Constant(float(DOLFIN_EPS))
            )
            streamline_test = dot(external_velocity, nabla_grad(self._xi))
            strong_residual = (
                dot(external_velocity, nabla_grad(self._nu_tilde))
                + (react_nt + penalty_react) * self._nu_tilde
                - source_nt
            )
            FNT += tau * streamline_test * strong_residual * self._dx
        self._a_nt = lhs(FNT)
        self._l_nt = rhs(FNT)

    def solve_turbulence_model(self):
        A_NT = assemble(self._a_nt)
        b_nt = assemble(self._l_nt)
        for bc in self._bcs:
            bc.apply(A_NT, b_nt)
        solve(A_NT, self._nu_tilde1.vector(), b_nt)

    def update_variables(self, relaxation=1.0):
        self._nu_tilde0.assign(relaxation * self._nu_tilde1 + (1.0 - relaxation) * self._nu_tilde0)

    @property
    def nu_tilde0(self):
        return self._nu_tilde0

    @property
    def nu_tilde1(self):
        return self._nu_tilde1

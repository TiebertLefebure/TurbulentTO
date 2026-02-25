from dolfin import *
from Utilities import *

# -------------------------------------------------------------------------- #
# Spalart-Allmaras turbulence model  (adjustments for Topology Optimization) #
# -------------------------------------------------------------------------- #

def _smooth_abs(value, smooth_abs_eps):
    return sqrt(value**2 + Constant(smooth_abs_eps))


def sa_min(a, b, smooth_abs_eps=None):
    """UFL-safe min helper; optionally smoothed for differentiability."""
    if smooth_abs_eps is None:
        return (a + b - abs(a - b)) / Constant(2.0)
    return (a + b - _smooth_abs(a - b, smooth_abs_eps)) / Constant(2.0)


def sa_turbulent_viscosity(nu_tilde, nu_laminar, smooth_abs_eps=None):
    """
    SA eddy viscosity from nu_tilde.
    If smooth_abs_eps is provided, returns a positive smoothed viscosity.
    """
    chi = nu_tilde / (nu_laminar + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1) ** 3)
    nu_t_raw = nu_tilde * f_v1
    if smooth_abs_eps is None:
        return nu_t_raw
    return 0.5 * (nu_t_raw + _smooth_abs(nu_t_raw, smooth_abs_eps))


def sa_transport_terms(external_velocity, nu_tilde, nu_laminar, wall_distance, smooth_abs_eps=None):
    """Build SA transport-model terms shared across solvers."""
    sigma = Constant(2.0 / 3.0)
    cb1 = Constant(0.1355)
    cb2 = Constant(0.622)
    kappa = Constant(0.41)
    cw2 = Constant(0.3)
    cw3 = Constant(2.0)
    cw1 = cb1 / kappa**2 + (Constant(1.0) + cb2) / sigma

    chi = nu_tilde / (nu_laminar + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1) ** 3)
    f_v2 = Constant(1.0) - chi / (Constant(1.0) + chi * f_v1)
    f_t2 = Constant(1.2) * exp(Constant(-0.5) * chi**2)

    # SA model uses vorticity magnitude in S_tilde.
    omega_sq = Constant(2.0) * inner(skew(nabla_grad(external_velocity)), skew(nabla_grad(external_velocity)))
    S = sqrt(omega_sq + DOLFIN_EPS)

    y_safe = wall_distance + DOLFIN_EPS
    S_tilde = S + nu_tilde / (kappa**2 * y_safe**2) * f_v2

    r_arg = nu_tilde / (S_tilde * kappa**2 * y_safe**2 + DOLFIN_EPS)
    r = sa_min(r_arg, Constant(10.0), smooth_abs_eps=smooth_abs_eps)

    g = r + cw2 * (r**6 - r)
    f_w = g * ((Constant(1.0) + cw3**6) / (g**6 + cw3**6)) ** (Constant(1.0) / Constant(6.0))

    prod_nt = cb1 * (Constant(1.0) - f_t2) * S_tilde * nu_tilde
    react_nt = cw1 * f_w * (nu_tilde / y_safe**2)
    cross_diff_nt = (cb2 / sigma) * inner(nabla_grad(nu_tilde), nabla_grad(nu_tilde))
    source_nt = prod_nt + cross_diff_nt
    return sigma, react_nt, source_nt


class SpalartAllmarasGeneral:
    def __init__(
        self,
        N,
        bcn,
        nu_tilde_init,
        nu,
        force,
        custom_dx,
        custom_ds,
        distance_field,
        nu_tilde_penalty_reaction=None,
    ):
        """Base class for the Spalart-Allmaras one-equation turbulence model."""
        self._N = N
        self._bcn = bcn
        self._nu_tilde_init = nu_tilde_init

        self._nu = nu
        self._force = force
        self._dx = custom_dx
        self._ds = custom_ds
        self._y = distance_field
        self._nu_tilde_penalty_reaction = nu_tilde_penalty_reaction

        self._construct_functions()

    def _construct_functions(self):
        """Construct model functions for the modified turbulent viscosity."""
        self._nu_tilde, self._xi, self._nu_tilde1, self._nu_tilde0 = initialize_functions(self._N, Constant(self._nu_tilde_init))

    def construct_forms(self):
        """Constructs the variational forms. Must be implemented in subclasses."""
        raise NotImplementedError("This method must be implemented in subclasses.")

    def solve_turbulence_model(self):
        """Solves the transport equation for nu_tilde."""
        A_NT = assemble(self._a_nt); b_nt = assemble(self._l_nt)
        [bc.apply(A_NT,b_nt) for bc in self._bcn]
        solve(A_NT, self._nu_tilde1.vector(), b_nt)

        # Enforce positivity
        self._nu_tilde1 = bound_from_bellow(self._nu_tilde1, 1e-16)

    def update_variables(self, relaxation = 1.0):
        """Update nu_tilde variable with relaxation."""
        # The assign method can handle linear combinations of Functions.
        # This is more readable and consistent with the k-epsilon implementation.
        self._nu_tilde0.assign(relaxation * self._nu_tilde1 + (1.0 - relaxation) * self._nu_tilde0)

    def _construct_turbulent_quantities(self, external_u1):
        """
        Constructs the various terms (production, destruction, etc.) for the
        Spalart-Allmaras transport equation.
        This uses a stable formulation where production is an explicit source
        and destruction is an implicit sink.
        """
        self._nu_t = sa_turbulent_viscosity(self._nu_tilde0, self._nu)
        self._sigma, self._react_nt, self._source_nt = sa_transport_terms(
            external_u1,
            self._nu_tilde0,
            self._nu,
            self._y,
        )

    @property
    def nu_t(self):
        """Turbulent eddy viscosity."""
        return self._nu_t

    @property
    def nu_tilde0(self):
        """Value of nu_tilde from previous iteration."""
        return self._nu_tilde0

    @property
    def nu_tilde1(self):
        """Value of nu_tilde for current iteration."""
        return self._nu_tilde1


class SpalartAllmarasSteadyState(SpalartAllmarasGeneral):
    def __init__(
        self,
        N,
        bcn,
        nu_tilde_init,
        nu,
        force,
        custom_dx,
        custom_ds,
        distance_field,
        nu_tilde_penalty_reaction=None,
    ):
        super().__init__(
            N,
            bcn,
            nu_tilde_init,
            nu,
            force,
            custom_dx,
            custom_ds,
            distance_field,
            nu_tilde_penalty_reaction=nu_tilde_penalty_reaction,
        )

    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)
        penalty_react = self._nu_tilde_penalty_reaction
        if penalty_react is None:
            penalty_react = Constant(0.0)

        # Weak form for steady-state SA model
        FNT  = dot(dot(external_u1, nabla_grad(self._nu_tilde)), self._xi)*self._dx \
            + inner((self._nu + self._nu_tilde0) / self._sigma * grad(self._nu_tilde), grad(self._xi))*self._dx \
            + dot((self._react_nt + penalty_react) * self._nu_tilde, self._xi)*self._dx \
            - dot(self._source_nt, self._xi)*self._dx
        self._a_nt = lhs(FNT); self._l_nt = rhs(FNT)
        # added penalization term to the Spalart-Allmaras transport equation -> Yoon 2016 Eq.(27)

class SpalartAllmarasTransient(SpalartAllmarasGeneral):
    def __init__(
        self,
        N,
        bcn,
        nu_tilde_init,
        nu,
        force,
        custom_dx,
        custom_ds,
        dt,
        distance_field,
        nu_tilde_penalty_reaction=None,
    ):
        self._dt = dt
        super().__init__(
            N,
            bcn,
            nu_tilde_init,
            nu,
            force,
            custom_dx,
            custom_ds,
            distance_field,
            nu_tilde_penalty_reaction=nu_tilde_penalty_reaction,
        )

    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)
        penalty_react = self._nu_tilde_penalty_reaction
        if penalty_react is None:
            penalty_react = Constant(0.0)
        mesh = self._nu_tilde.function_space().mesh()
        h = CellDiameter(mesh)
        u_mag = sqrt(dot(external_u1, external_u1) + 1e-10)
        tau = h / (2.0 * u_mag)

        # Residual used in SUPG stabilization (same style as k-epsilon model)
        res_nt = (self._nu_tilde - self._nu_tilde0) / self._dt \
               + dot(external_u1, nabla_grad(self._nu_tilde)) \
               + (self._react_nt + penalty_react) * self._nu_tilde - self._source_nt
        F_supg_nt = inner(tau * dot(external_u1, nabla_grad(self._xi)), res_nt) * self._dx

        # Weak form for transient SA model
        FNT  = dot((self._nu_tilde - self._nu_tilde0) / self._dt, self._xi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._nu_tilde)), self._xi)*self._dx \
            + inner((self._nu + self._nu_tilde0) / self._sigma * grad(self._nu_tilde), grad(self._xi))*self._dx \
            + dot((self._react_nt + penalty_react) * self._nu_tilde, self._xi)*self._dx \
            - dot(self._source_nt, self._xi)*self._dx + F_supg_nt
        self._a_nt = lhs(FNT); self._l_nt = rhs(FNT)
        # added penalty_react to the Spalart-Allmaras transport equation -> Yoon 2016 Eq.(27)

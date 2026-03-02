from dolfin import *
from Utilities import *

# --------------------------------- #
# Spalart-Allmaras turbulence model #
# --------------------------------- #

class SpalartAllmarasGeneral:
    def __init__(self, N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field, sa_options=None):
        """Base class for the Spalart-Allmaras one-equation turbulence model."""
        self._N = N
        self._bcn = bcn
        self._nu_tilde_init = nu_tilde_init

        self._nu = nu
        self._force = force
        self._dx = custom_dx
        self._ds = custom_ds
        self._y = distance_field 
        # This Spalart-Allmaras implementation (without TO) uses a relaxed wall Eikonal equation for the wall-distance field (in Utilities.py)
        # According to Yoon et al. (2016)
        self._sa_options = {} if sa_options is None else dict(sa_options)

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
        # Helpers for min/max functions in UFL
        def Min(a, b): return (a+b-abs(a-b))/Constant(2)
        def Max(a, b): return (a+b+abs(a-b))/Constant(2)

        # Non-dimensional viscosity ratio
        chi = self._nu_tilde0 / self._nu
        
        # Damping functions
        f_v1 = chi**3 / (chi**3 + Constant(7.1)**3)
        f_v2 = 1 - chi / (1 + chi * f_v1)
        # Yoon 2016 Eq. (11)-(15) uses the standard SA internal-flow terms
        # without the trip-term correction in production/destruction.
        # Keep f_t2 disabled (f_t2 = 0.0) so the model matches that formulation.
        f_t2 = Constant(0.0)

        # SA model uses vorticity magnitude in S_tilde.
        # On some meshes in this repo (e.g. U-bend), the geometry is stored as an
        # embedded 2D surface in XYZ coordinates. For SA, using skew(nabla_grad(u))
        # on such meshes can under-predict the in-plane vorticity magnitude and
        # suppress turbulence production. For 2D domains, compute the in-plane curl
        # directly from (u_x, u_y) so XY and XYZ planar meshes behave consistently.
        mesh = self._N.mesh()
        if mesh.topology().dim() == 2 and external_u1.ufl_shape[0] >= 2:
            omega = Dx(external_u1[1], 0) - Dx(external_u1[0], 1)
            S = sqrt(omega**2 + DOLFIN_EPS)
        else:
            omega_sq = 2 * inner(skew(nabla_grad(external_u1)), skew(nabla_grad(external_u1)))
            S = sqrt(omega_sq + DOLFIN_EPS) # Add epsilon for robustness

        # Wall distance with safety epsilon
        y_safe = self._y + DOLFIN_EPS
        # calculate_Distance_field in Utilities.py solves the Eikonal equation for the wall-distance function
        kappa = 0.41

        # Modified strain rate S_tilde (standard SA piecewise definition).
        # The negative-S_bar branch is usually inactive in simple channel flow,
        # but it can matter in curved/adverse-gradient regions (e.g. a U-bend).
        S_bar = self._nu_tilde0 / (kappa**2 * y_safe**2) * f_v2
        cv2 = Constant(0.7)
        cv3 = Constant(0.9)
        S_tilde_pos = S + S_bar
        S_tilde_neg = S + S * (cv2**2 * S + cv3 * S_bar) / (
            (cv3 - Constant(2.0) * cv2) * S - S_bar + DOLFIN_EPS
        )
        S_tilde = conditional(ge(S_bar, -cv2 * S), S_tilde_pos, S_tilde_neg)

        # Argument for f_w function
        r_arg = self._nu_tilde0 / (S_tilde * kappa**2 * y_safe**2 + DOLFIN_EPS)
        r = Min(r_arg, Constant(10.0)) # Cap r as in original model to prevent singularity
        if self._sa_options.get('R_CLIP_NONNEGATIVE', False):
            r = Max(r, Constant(0.0))

        # Wall function f_w
        g = r + 0.3 * (r**6 - r) # Note: c_w2 = 0.3
        cw3 = 2.0
        f_w = g * ((1 + cw3**6) / (g**6 + cw3**6))**(1.0/6.0)

        # Turbulent eddy viscosity for the RANS equations
        self._nu_t = self._nu_tilde0 * f_v1


        # ----------------------------------------------------
        # --- Terms for the SA nu_tilde transport equation ---
        # ----------------------------------------------------
        
        # Model constants
        sigma = 2.0/3.0
        cb1 = 0.1355
        cb2 = 0.622
        cw1 = cb1/kappa**2 + (1 + cb2)/sigma

        # Production term (explicit source)
        # P = cb1 * S_tilde * nu_tilde
        # Yoon 2016 has f_t2 = 0
        prod_nt = cb1 * (1 - f_t2) * S_tilde * self._nu_tilde0
        
        # Destruction term (linearized for implicit sink)
        # D = cw1 * f_w * (nu_tilde/y)^2 ~= (cw1 * f_w * nu_tilde_0 / y^2) * nu_tilde
        self._react_nt = cw1 * f_w * (self._nu_tilde0 / y_safe**2)
        
        # Cross-diffusion term (explicit source)
        # This is the second SA diffusion contribution (cb2/sigma * |grad(nu_tilde)|^2).
        # The first diffusion contribution is the divergence term that appears in the
        # weak form as inner(((nu + nu_tilde0)/sigma) * grad(nu_tilde), grad(test)).
        cross_diff_nt = (cb2/sigma) * inner(nabla_grad(self._nu_tilde0), nabla_grad(self._nu_tilde0))

        # Combine all explicit source terms
        self._source_nt = prod_nt + cross_diff_nt

        # SA debug expressions (all scalar UFL expressions, evaluated/projected on demand).
        self._sa_debug_expressions = {
            'S': S,
            'S_bar': S_bar,
            'S_tilde': S_tilde,
            'r_arg': r_arg,
            'r': r,
            'f_w': f_w,
            'f_v1': f_v1,
            'f_v2': f_v2,
            'nu_t': self._nu_t,
            'prod_nt': prod_nt,
            'cross_diff_nt': cross_diff_nt,
            'destroy_nt': self._react_nt * self._nu_tilde0,
            'react_nt': self._react_nt,
            'source_nt': self._source_nt,
            'neg_sbar_branch': conditional(lt(S_bar, -cv2 * S), Constant(1.0), Constant(0.0)),
            'r_arg_lt0': conditional(lt(r_arg, Constant(0.0)), Constant(1.0), Constant(0.0)),
            'r_arg_gt10': conditional(gt(r_arg, Constant(10.0)), Constant(1.0), Constant(0.0)),
            'S_tilde_lt0': conditional(lt(S_tilde, Constant(0.0)), Constant(1.0), Constant(0.0)),
        }

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

    @property
    def sa_debug_expressions(self):
        """Scalar UFL expressions for inspecting SA production/destruction balance."""
        return getattr(self, '_sa_debug_expressions', {})


class SpalartAllmarasSteadyState(SpalartAllmarasGeneral):
    def __init__(self, N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field, sa_options=None):
        super().__init__(N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field, sa_options=sa_options)

    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        sigma = 2.0/3.0

        # Weak form for steady-state SA model
        FNT  = dot(dot(external_u1, nabla_grad(self._nu_tilde)), self._xi)*self._dx \
            + inner((self._nu + self._nu_tilde0) / sigma * grad(self._nu_tilde), grad(self._xi))*self._dx \
            + dot(self._react_nt * self._nu_tilde, self._xi)*self._dx \
            - dot(self._source_nt, self._xi)*self._dx
        self._a_nt = lhs(FNT); self._l_nt = rhs(FNT)


class SpalartAllmarasTransient(SpalartAllmarasGeneral):
    def __init__(self, N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, dt, distance_field, sa_options=None):
        self._dt = dt
        super().__init__(N, bcn, nu_tilde_init, nu, force, custom_dx, custom_ds, distance_field, sa_options=sa_options)

    def construct_forms(self, external_u1):
        self._construct_turbulent_quantities(external_u1)

        sigma = 2.0/3.0
        mesh = self._nu_tilde.function_space().mesh()
        h = CellDiameter(mesh)
        u_mag = sqrt(dot(external_u1, external_u1) + 1e-10)
        tau = h / (2.0 * u_mag)
        supg_factor = Constant(float(self._sa_options.get('SUPG_FACTOR', 1.0)))

        # Residual used in SUPG stabilization (same style as k-epsilon model)
        res_nt = (self._nu_tilde - self._nu_tilde0) / self._dt \
               + dot(external_u1, nabla_grad(self._nu_tilde)) \
               + self._react_nt * self._nu_tilde - self._source_nt
        F_supg_nt = supg_factor * inner(tau * dot(external_u1, nabla_grad(self._xi)), res_nt) * self._dx

        # Weak form for transient SA model
        FNT  = dot((self._nu_tilde - self._nu_tilde0) / self._dt, self._xi)*self._dx \
            + dot(dot(external_u1, nabla_grad(self._nu_tilde)), self._xi)*self._dx \
            + inner((self._nu + self._nu_tilde0) / sigma * grad(self._nu_tilde), grad(self._xi))*self._dx \
            + dot(self._react_nt * self._nu_tilde, self._xi)*self._dx \
            - dot(self._source_nt, self._xi)*self._dx + F_supg_nt
        self._a_nt = lhs(FNT); self._l_nt = rhs(FNT)

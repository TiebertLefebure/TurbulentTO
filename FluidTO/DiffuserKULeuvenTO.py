from dolfin import *
import numpy as np
import os
import shutil
from time import strftime, localtime
try:
    from ufl import tanh
except ModuleNotFoundError:
    from ufl_legacy import tanh
from mma import mmasub

# =================================================
# Original KU Leuven file for Borrvall diffuser
# =================================================


# -----------------------------------
# Borrvall Diffuser case (Laminar)
# -----------------------------------


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(THIS_DIR, 'Results_DiffuserKULeuvenTO')
MESH_DIR = os.path.join(RESULTS_ROOT, 'mesh')
RHO_DIR = os.path.join(RESULTS_ROOT, 'rho_')
RHOP_DIR = os.path.join(RESULTS_ROOT, 'rho_p')
U_DIR = os.path.join(RESULTS_ROOT, 'u_')
P_DIR = os.path.join(RESULTS_ROOT, 'p_')
DESIGN_DIR = os.path.join(RESULTS_ROOT, 'design_')
LOG_FILE = os.path.join(RESULTS_ROOT, 'OptimizationLogDiffuser.txt')

def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

tol = DOLFIN_EPS
mufluid = Constant(1.0)
rhofluid = Constant(1.0)

alphafluid = Constant(2.5 * mufluid / 100**2.0)     # parameter for \alpha
alphasolid = Constant(2.5 * mufluid / 0.01**2.0)     # parameter for \alpha

qpenal = Constant(0.1) # q value that controls difficulty/discrete-valuedness of solution

L = 1.0                 # length of the edge of the domain
Height = 1.0            # Height of the domain

N = 120 #384
r_filter = L*2.0/(N)
r = r_filter/(2*3**0.5)

betaproj = Constant(0.1)
eta_i = 0.50

mmma = 1 # number of constraints
vol_frac = 0.5 # Target for volume fraction constraint

# qpen = [0.001, 0.01, 0.1, 1.0]
qpen = [0.1]

max_inner_iterations = 100

def projection(rho, etaproj):
    return (tanh(betaproj*Constant(etaproj)) + tanh(betaproj*(rho - Constant(etaproj)))) / (tanh(betaproj*Constant(etaproj)) + tanh(betaproj*(Constant(1.0) - Constant(etaproj))))

def alpha(rho):
    return (alphasolid + (alphafluid - alphasolid) * rho * (1 + qpenal) / (rho + qpenal))

def my_between(x, range, eps=DOLFIN_EPS):
	return (range[0]- eps <= x) and (x <= range[1] + eps)

#########################################################
#          MESH AND FUNCTION SPACES                     #
#########################################################

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0)\
    , Point(L, L), int(N/L), int(N), 'crossed'))
ensure_clean_dir(RESULTS_ROOT)
ensure_clean_dir(MESH_DIR)
File(os.path.join(MESH_DIR, 'mesh.pvd')) << mesh

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A = FiniteElement("DG", mesh.ufl_cell(), 0)        # control function space

FlowSpace = FunctionSpace(mesh, U_h*P_h)
FlowSpaceAdj = FunctionSpace(mesh, U_h*P_h)
DensitySpace = FunctionSpace(mesh, A)

w_fwd = Function(FlowSpace)
(u, p) = split(w_fwd)
w_adj = Function(FlowSpaceAdj)
(v, q) = split(w_adj)

rho = Function(DensitySpace)
rho_f = Function(DensitySpace)

fplot = Function(DensitySpace)

unfilteredGradient = Function(DensitySpace)
filteredGradient = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)

#########################################################
#          FORWARD BOUNDARY CONDITIONS                  #
#########################################################

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], L, tol) or near(x[1], 0.0, tol)  \
            or (near(x[0], L, tol) and my_between(x[1], (0.0, 1/3*L), tol)) \
            or (near(x[0], L, tol) and my_between(x[1], (2/3*L, L), tol)))

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol) and my_between(x[1], (0.0, L), tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol) and my_between(x[1], (1/3*L, 2/3*L), tol)

mark = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundaries.set_all(mark["generic"])
walls = Walls()
inlet = Inlet()
outlet = Outlet()
walls.mark(boundaries, mark['walls'])
inlet.mark(boundaries, mark['inlet'])
outlet.mark(boundaries, mark['outlet'])

# Inlet
u_max_in = 1.0          # [m/s]
u_max_inlet = Constant(u_max_in)
width_inlet = Constant(1.0)
y_inlet = (1/2) * Height
y_local_inlet = Expression('x[1]-y_inlet', degree=1, y_inlet=y_inlet)
u_inlet = Expression(('u_max_inlet * (1 - pow(2*y_local_inlet/width_inlet, 2))', '0.0'), degree=2, u_max_inlet=u_max_inlet, y_local_inlet=y_local_inlet, width_inlet=width_inlet)

# Outlet
u_max_out = 3.0          # [m/s]
u_max_outlet = Constant(u_max_out)
width_outlet = Constant(1/3.)
y_outlet = (1/2) * Height
y_local_outlet = Expression('x[1]-y_outlet', degree=1, y_outlet=y_outlet)
u_outlet = Expression(('u_max_outlet * (1 - pow(2*y_local_outlet/width_outlet, 2))', '0.0'), degree=2, u_max_outlet=u_max_outlet, y_local_outlet=y_local_outlet, width_outlet=width_outlet)

# Walls
u_noslip = Constant((0.0, 0.0))

bcu_walls = DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet = DirichletBC(FlowSpace.sub(0), u_inlet, boundaries, mark["inlet"])
bcu_outlet = DirichletBC(FlowSpace.sub(0), u_outlet, boundaries, mark["outlet"])

bc_NS = [bcu_walls, bcu_inlet, bcu_outlet]

bcu_wallsAdj = DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inletAdj = DirichletBC(FlowSpaceAdj.sub(0), u_inlet, boundaries, mark["inlet"])
bcu_outletAdj = DirichletBC(FlowSpaceAdj.sub(0), u_outlet, boundaries, mark["outlet"])

bc_NS_adj = [bcu_wallsAdj, bcu_inletAdj, bcu_outletAdj]

#####################################
#   Active design space
#########################
AreaOfInterest = Function(DensitySpace)
assign(AreaOfInterest, interpolate(Expression('(x[0] >= 0.0 && x[0] <= L) ? 1.0 : 0.0', L = L, degree = 0), DensitySpace))

#####################################
#   Filter Setup
#########################
uFilter = TrialFunction(DensitySpace)
vFilter = TestFunction(DensitySpace)
PDEFilterIn = Function(DensitySpace)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

def pdefilter(filterin, filterout):#, uFilter, vFilter, PDEFilterIn, n, h, h_avg, boundaries, ds):

    alphaDG = 4.0

    Helmholtz = r**2 * (alphaDG/h_avg*dot(jump(vFilter, n), jump(uFilter, n)))*dS \
        + uFilter * vFilter * dx - PDEFilterIn * vFilter * dx

    assign(PDEFilterIn, filterin)
    solve(lhs(Helmholtz) == rhs(Helmholtz), filterout)

    return filterout

#########################################################
#########################################################
#          Optimization Problem                         #
#########################################################
#########################################################

ObjFunctional =  AreaOfInterest * (1/2 * mufluid * inner((nabla_grad(u) + nabla_grad(u).T), (nabla_grad(u) + nabla_grad(u).T)) + alpha(projection(rho_f, eta_i)) * inner(u, u)) * dx        # Eq1

volume = AreaOfInterest * dx
volume = assemble(volume)
vol_constraint = AreaOfInterest * projection(rho_f, eta_i) * dx - AreaOfInterest * vol_frac * dx # Old formulation. The value of the volume is not from 0 to 1
# vol_constraint = (1/vol_target) * AreaOfInterest * projection(rho_f, eta_i) * dx

sensitivities_vol_constraint = derivative(vol_constraint, rho_f)

# NSR only for fluid flow
NSR = rhofluid * inner(dot(u, nabla_grad(u)), v) * dx \
    + mufluid * inner(grad(u), grad(v)) * dx \
    + inner(grad(p), v) * dx + inner(div(u), q) * dx + alpha(projection(rho_f, eta_i)) * inner(u, v) * dx

UFL_State = NSR

UFL_AL = ObjFunctional + UFL_State

NSR_fwd = derivative(UFL_State, w_adj, TestFunction(FlowSpace))
NSR_adj = derivative(UFL_AL, w_fwd, TestFunction(FlowSpaceAdj))
ddx = derivative(UFL_AL, rho_f)

##################################################################
#                   Create Directories                            
##################################################################

ensure_clean_dir(RHO_DIR)
ensure_clean_dir(RHOP_DIR)
ensure_clean_dir(U_DIR)
ensure_clean_dir(P_DIR)
ensure_clean_dir(DESIGN_DIR)

rho_out = File(os.path.join(RHO_DIR, "plot_rho.pvd"))
rhop_out = File(os.path.join(RHOP_DIR, "plot_rho_p.pvd"))
u_out = File(os.path.join(U_DIR, "plot_u.pvd"))
p_out = File(os.path.join(P_DIR, "plot_p.pvd"))

txtout = open(LOG_FILE,"w")
header1 = "Iteration     "
header2 = "Obj.Function  "
header21 = "Obj. Conv.    "
header6 = "Vol_frac      "

txtout.write("{} {} {} {} {}\r\n".format(header1, header2, header21, header6, \
              strftime("%a, %d %b %Y %H:%M:%S", localtime())))
txtout.close()

###########################
#     Optimization
###########################

assign(rho, interpolate(Expression('(x[0] >= 0.0 && x[0] <= L)  ? 0.5 : 1.0', L = L, degree = 0), DensitySpace))

iter_count = 0
inner_count = 0
previousObjective = 0.0

#MMA Parameters
nmma = mesh.num_cells()
xval = np.zeros((nmma, 1))
xval[:,0] = rho.vector()
xold1 = np.zeros((nmma, 1))
xold2 = np.zeros((nmma, 1))
low = np.zeros((nmma, 1))
upp = np.zeros((nmma, 1))

a0 = 1.0
a = np.zeros((mmma, 1))
c = 1.0e4*np.ones((mmma, 1))
d = np.ones((mmma, 1))

xmin = np.zeros((nmma, 1))
xmax = np.ones((nmma, 1))

move = 0.2

xmma = np.zeros((nmma, 1))
df0dx = np.zeros((nmma, 1))
fval = np.zeros((mmma, 1))
dfdx = np.zeros((mmma, nmma))

convergenceTolerance = 1e-5

for jj in range(len(qpen)):

    inner_count = 0
    convergenceHistory = 0

    qpenal.assign(qpen[jj])
    # betaproj.assign(b_proj[jj])

    objectiveConvergence = False

    #OPTIMIZATION LOOP
    while inner_count <= max_inner_iterations and objectiveConvergence == False:

        ###########################
        #           FWD
        ###########################

        rho_f = pdefilter(rho, rho_f)

        fplot.vector()[:] = project(projection(rho_f, eta_i), DensitySpace).vector()[:]
        rho_out << rho
        rhop_out <<  fplot
        u_out << w_fwd.sub(0)
        p_out << w_fwd.sub(1)

        ###### SOLVE NSR_fwd
        Jac_NSR_fwd = derivative(NSR_fwd, w_fwd)
        problem_NSR_fwd = NonlinearVariationalProblem(NSR_fwd, w_fwd, bc_NS, Jac_NSR_fwd)
        solver_NSR_fwd = NonlinearVariationalSolver(problem_NSR_fwd)
        solver_NSR_fwd.parameters['nonlinear_solver'] = 'snes'
        #solver_NSR_fwd.parameters["snes_solver"]["report"] = False
        solver_NSR_fwd.parameters['snes_solver']['linear_solver'] = 'lu'
        solver_NSR_fwd.solve()
        
        ########################################
        #        ADJOINT
        ########################################

        ###### SOLVE NSR_adj
        Jac_NSR_adj = derivative(NSR_adj, w_adj)
        problem_NSR_adj = NonlinearVariationalProblem(NSR_adj, w_adj, bc_NS_adj, Jac_NSR_adj)
        solver_NSR_adj = NonlinearVariationalSolver(problem_NSR_adj)
        solver_NSR_adj.parameters['nonlinear_solver'] = 'snes'
        solver_NSR_adj.parameters["snes_solver"]["report"] = True
        solver_NSR_adj.parameters['snes_solver']['linear_solver'] = 'lu'
        solver_NSR_adj.parameters['snes_solver']['relative_tolerance'] = 1.0e-4
        solver_NSR_adj.solve()

        f0val = assemble(ObjFunctional)

        objConvergenceCheck = abs((f0val-previousObjective)/f0val)
        if objConvergenceCheck < convergenceTolerance:
            convergenceHistory+= 1
            if convergenceHistory == 5:
                objectiveConvergence = True
        else:
            convergenceHistory = 0

        previousObjective = f0val

        unfilteredGradient.vector()[:] = assemble(ddx)[:]
        filteredGradient = pdefilter(unfilteredGradient, filteredGradient)
        np.savetxt(os.path.join(DESIGN_DIR, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])

        ########################################
        #           MMA
        ########################################

        df0dx[:,0] = filteredGradient.vector()

        fval = assemble(vol_constraint)
        unfiltered_s_vol.vector()[:] = assemble(sensitivities_vol_constraint)[:]
        filtered_s_vol = pdefilter(unfiltered_s_vol, filtered_s_vol)

        dfdx[0,:] = filtered_s_vol.vector()

                
        xmma,_ymma,_zmma,_lam,_xsi,_eta,_mu_mma,_zet,_s,low,upp = \
            mmasub(mmma,nmma,iter_count,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        rho.vector()[:] = xmma[:,0].copy()
        
        txtout = open(LOG_FILE,"a")
        txtout.write("{:02d}.{:03d}         {:.12}   {:.12}   {:.12}   {}\r\n".format(jj, inner_count, str(round(f0val, 10)).ljust(12), str(round(objConvergenceCheck, 10)).ljust(12),\
              str(round(assemble(rho * dx) / volume, 10)).ljust(12), \
              strftime("%a, %d %b %Y %H:%M:%S", localtime())))
        txtout.close()
        print("The volume fraction is: ", assemble(rho * dx) / volume)
        inner_count += 1
        iter_count += 1

from dolfin import *
import numpy as np
import os
import pyipopt
import shutil
from IPython.core.debugger import set_trace
from petsc4py import PETSc
import logging
from time import gmtime, strftime, localtime
from ufl import tanh
from scipy.sparse import csr_matrix, lil_matrix
from mma import mmasub
from mshr import *

tol = DOLFIN_EPS
mufluid = Constant(1.0)
rhofluid = Constant(1.0)

alphafluid = Constant(0.0)     # parameter for \alpha
alphasolid = Constant(1.0E4)     # parameter for \alpha

# alphafluid = Constant(2.5 * mufluid / 100**2.0)     # parameter for \alpha
# alphasolid = Constant(2.5 * mufluid / 0.01**2.0)     # parameter for \alpha

qpenal = Constant(0.1) # q value that controls difficulty/discrete-valuedness of solution

Length = 8.0                 # length of the edge of the domain
Height = 8.0            # Height of the domain

x_min = 0.0; x_max = Length # x dimensions
y_min = 0.0; y_max = Height # y dimensions

N = 100 #384
r_filter = Length*2.0/(N)
r = r_filter/(2*3**0.5)

betaproj = Constant(0.1)
eta_d = 0.25
eta_i = 0.50
eta_e = 0.75

vol_frac = 0.4 # Target for volume fraction constraint

def projection(rho, etaproj):
    return (tanh(betaproj*Constant(etaproj)) + tanh(betaproj*(rho - Constant(etaproj)))) / (tanh(betaproj*Constant(etaproj)) + tanh(betaproj*(Constant(1.0) - Constant(etaproj))))

def alpha(rho):
    return (alphasolid + (alphafluid - alphasolid) * rho * (1 + qpenal) / (rho + qpenal))

def my_between(x, range, eps=DOLFIN_EPS):
	return (range[0]- eps <= x) and (x <= range[1] + eps)

def Array2PETScVector(x):
    # x is an numpy.ndarray
    # returns the same array in a Vector or #PETScVector format
    out = Vector(MPI.comm_self, len(x))
    out.set_local(x)
    return out 
    # return as_backend_type(out)

#########################################################
#          MESH AND FUNCTION SPACES                     #
#########################################################

# parameters["ghost_mode"] = "shared_facet"     # For parallel computing with mpirun -n n_cores python3 file.py

delta_x = x_max - x_min; delta_y = y_max - y_min

mesh = Mesh()
design_domain = Rectangle(Point(x_min, y_min), Point(x_max, y_max))
inlet_domain = Rectangle(Point(6.0, -3.0), Point(7.0, 0.0))
outlet_domain_1 = Rectangle(Point(-1.0, 1.0), Point(0.0, 2.5))
outlet_domain_2 = Rectangle(Point(-1.0, 5.5), Point(0.0, 7.0))
domain = design_domain + inlet_domain + outlet_domain_1 + outlet_domain_2
domain.set_subdomain(1, design_domain)

# Polygon can be used for more complex meshes
# design_domain = Polygon([Point(x_min, y_min),
#                         Point(x_max, y_min),
#                         Point(x_max, y_max),
#                         Point(x_min, y_max)])

mesh = generate_mesh(domain, N)
File('mesh/mesh.pvd') << mesh

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A = FiniteElement("DG", mesh.ufl_cell(), 0)        # control function space

FlowSpace = FunctionSpace(mesh, U_h*P_h)
FlowSpaceAdj = FunctionSpace(mesh, U_h*P_h)
FlowSpaceAdj_mc1 = FunctionSpace(mesh, U_h*P_h)
FlowSpaceAdj_mc2 = FunctionSpace(mesh, U_h*P_h)
DensitySpace = FunctionSpace(mesh, A)

w_fwd = Function(FlowSpace)
(u, p) = split(w_fwd)
w_adj = Function(FlowSpaceAdj)
(v, q) = split(w_adj)
w_adj_mc1 = Function(FlowSpaceAdj_mc1)
(v2, q2) = split(w_adj_mc1)
w_adj_mc2 = Function(FlowSpaceAdj_mc2)
(v3, q3) = split(w_adj_mc2)

rho = Function(DensitySpace)
rho_f = Function(DensitySpace)
rho_p = Function(DensitySpace)

splot = Function(DensitySpace)
fplot = Function(DensitySpace)

unfilteredGradient = Function(DensitySpace)
filteredGradient = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)
unfiltered_s_out1 = Function(DensitySpace)
filtered_s_out1 = Function(DensitySpace)
unfiltered_s_out2 = Function(DensitySpace)
filtered_s_out2 = Function(DensitySpace)

#########################################################
#          FORWARD BOUNDARY CONDITIONS                  #
#########################################################

class All(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
        
# class Walls(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and (near(x[1], Length, tol) or near(x[1], 0.0, tol)  \
#             or (near(x[0], Length, tol) and my_between(x[1], (0.0, 1/3*Length), tol)) \
#             or (near(x[0], Length, tol) and my_between(x[1], (2/3*Length, Length), tol)))

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -3.0, tol) and my_between(x[0], (6.0, 7.0), tol)

class Outlet1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -1.0, tol) and my_between(x[1], (1.0, 2.5), tol)

class Outlet2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -1.0, tol) and my_between(x[1], (5.5, 7.0), tol)

mark = {"generic": 2, "walls": 3, "inlet": 4, "outlet1": 5, "outlet2": 6}
# markers = MeshFunction('size_t', mesh, 2, mesh.domains())
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundaries.set_all(mark["generic"])
walls = All()
inlet = Inlet()
outlet1 = Outlet1()
outlet2 = Outlet2()
walls.mark(boundaries, mark['walls'])
inlet.mark(boundaries, mark['inlet'])
outlet1.mark(boundaries, mark['outlet1'])
outlet2.mark(boundaries, mark['outlet2'])
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
File('mesh/markers.pvd') << boundaries

# Inlet
u_max_in = 1.0          # [m/s]
u_max_inlet = Constant(u_max_in)
width_inlet = Constant(1.0)
x_inlet = Constant(6.5)
x_local_inlet = Expression('x[0]-x_inlet', degree=1, x_inlet=x_inlet)
u_inlet = Expression(('0.0', 'u_max_inlet * (1 - pow(2*x_local_inlet/width_inlet, 2))'), degree=2, u_max_inlet=u_max_inlet, x_local_inlet=x_local_inlet, width_inlet=width_inlet)

# Outlet
p_out = 0.0           # [Pa]
p_outlet = Constant(p_out)

# Walls
u_noslip = Constant((0.0, 0.0))

bcu_walls = DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet = DirichletBC(FlowSpace.sub(0), u_inlet, boundaries, mark["inlet"])
bcp_outlet1 = DirichletBC(FlowSpace.sub(1), p_outlet, boundaries, mark["outlet1"])
bcp_outlet2 = DirichletBC(FlowSpace.sub(1), p_outlet, boundaries, mark["outlet2"])

bc_NS = [bcu_walls, bcu_inlet, bcp_outlet1, bcp_outlet2]

bcu_wallsAdj = DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inletAdj = DirichletBC(FlowSpaceAdj.sub(0), u_inlet, boundaries, mark["inlet"])
bcp_outletAdj1 = DirichletBC(FlowSpaceAdj.sub(1), p_outlet, boundaries, mark["outlet1"])
bcp_outletAdj2 = DirichletBC(FlowSpaceAdj.sub(1), p_outlet, boundaries, mark["outlet2"])

bc_NS_adj = [bcu_wallsAdj, bcu_inletAdj, bcp_outletAdj1, bcp_outletAdj2]

bcu_wallsAdj_mc1 = DirichletBC(FlowSpaceAdj_mc1.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inletAdj_mc1 = DirichletBC(FlowSpaceAdj_mc1.sub(0), u_inlet, boundaries, mark["inlet"])
bcp_outletAdj_mc1_1 = DirichletBC(FlowSpaceAdj_mc1.sub(1), p_outlet, boundaries, mark["outlet1"])
bcp_outletAdj_mc1_2 = DirichletBC(FlowSpaceAdj_mc1.sub(1), p_outlet, boundaries, mark["outlet2"])

bc_NS_adj_mc1 = [bcu_wallsAdj_mc1, bcu_inletAdj_mc1, bcp_outletAdj_mc1_1, bcp_outletAdj_mc1_2]

bcu_wallsAdj_mc2 = DirichletBC(FlowSpaceAdj_mc2.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inletAdj_mc2 = DirichletBC(FlowSpaceAdj_mc2.sub(0), u_inlet, boundaries, mark["inlet"])
bcp_outletAdj_mc2_1 = DirichletBC(FlowSpaceAdj_mc2.sub(1), p_outlet, boundaries, mark["outlet1"])
bcp_outletAdj_mc2_2 = DirichletBC(FlowSpaceAdj_mc2.sub(1), p_outlet, boundaries, mark["outlet2"])

bc_NS_adj_mc2 = [bcu_wallsAdj_mc2, bcu_inletAdj_mc2, bcp_outletAdj_mc2_1, bcp_outletAdj_mc2_2]

#------ For independent x and y components on Dirichlet boundary conditions

# bcux_inlet = DirichletBC(FlowSpace.sub(0).sub(0), ux_inlet, boundaries, mark["inlet"])
# bcuy_inlet = DirichletBC(FlowSpace.sub(0).sub(1), uy_zero, boundaries, mark["inlet"])
# bc_NS = [bcu_walls, bcux_inlet, bcuy_inlet, bcp_outlet, bcuy_outlet]
# bcux_inletAdj = DirichletBC(FlowSpaceAdj.sub(0).sub(0), ux_inlet, boundaries, mark["inlet"])
# bcuy_inletAdj = DirichletBC(FlowSpaceAdj.sub(0).sub(1), uy_zero, boundaries, mark["inlet"])
# bc_NS_adj = [bcu_wallsAdj, bcux_inletAdj, bcuy_inletAdj, bcp_outletAdj, bcuy_outletAdj]

#####################################
#   Active design space
#########################
AreaOfInterest = Function(DensitySpace)
assign(AreaOfInterest, interpolate(Expression('(x[0] >= 0.0 && x[0] <= Length && x[1] >= 0.0) ? 1.0 : 0.0', Length = Length, degree = 0), DensitySpace))

#########################################################
#         Active Design Variables         #
#########################################################
nodes = mesh.coordinates()
connectivities = mesh.cells()
ActiveDV = []
PassiveDV = []
for ii in range(connectivities.shape[0]):
    xcenter = (nodes[connectivities[ii,0], 0] + nodes[connectivities[ii,1], 0] + nodes[connectivities[ii,2], 0])/3
    ycenter = (nodes[connectivities[ii,0], 1] + nodes[connectivities[ii,1], 1] + nodes[connectivities[ii,2], 1])/3
    if not ((xcenter <= x_min or ycenter <= y_min)):
        ActiveDV.append(ii)
    else:
        PassiveDV.append(ii)
        
# set_trace()

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
    gammaDG = 8.0

    # Helmholtz = r**2 * (alphaDG/h_avg*dot(jump(vFilter, n), jump(uFilter, n)))*dS \
        # + uFilter * vFilter * dx - PDEFilterIn * vFilter * dx

    Helmholtz = r**2 * (alphaDG/h_avg*dot(jump(vFilter, n), jump(uFilter, n))*dS \
        + (gammaDG/h)*vFilter*uFilter*ds(2)) \
        + uFilter * vFilter * dx - PDEFilterIn * vFilter * dx

    assign(PDEFilterIn, filterin)
    solve(lhs(Helmholtz) == rhs(Helmholtz), filterout)
    
    # For parallel computing with mpirun -n n_cores python3 file.py
    # problem_Helmholtz = LinearVariationalProblem(lhs(Helmholtz), rhs(Helmholtz), filterout)
    # solver_Helmholtz = LinearVariationalSolver(problem_Helmholtz)
    # solver_Helmholtz.parameters['linear_solver'] = 'gmres'
    # solver_Helmholtz.solve()
    # Or    
    # solve(lhs(Helmholtz) == rhs(Helmholtz), filterout, solver_parameters={"linear_solver": "gmres"})

    return filterout

#########################################################
#########################################################
#          Optimization Problem                         #
#########################################################
#########################################################

# Objective function
# norm_J = 100 * mufluid * (u_max_inlet/width_inlet)**2 # Normalization for energy dissipation
norm_J = 1.0
ObjFunctional = (1.0/norm_J) * AreaOfInterest * (1/2 * mufluid * inner((nabla_grad(u) + nabla_grad(u).T), (nabla_grad(u) + nabla_grad(u).T)) + alpha(projection(rho_f, eta_i)) * inner(u, u)) * dx        # Eq1

# Mass flow constraints
bcu_inletAdj.apply(w_fwd.vector())
inflow = assemble((dot(u, n)) * ds(mark["inlet"]))
outflow_outlet1 = 1/2  
outflow_outlet2 = 1/2

# set_trace()
m_constraint_1 = (1/(-inflow*outflow_outlet1)) * dot(u, n) * ds(mark["outlet1"])
m_constraint_2 = (1/(-inflow*outflow_outlet2)) * dot(u, n) * ds(mark["outlet2"])

# Volume constraint
volume = AreaOfInterest * dx
volume = assemble(volume)
vol_target = vol_frac*(volume)
vol_constraint = AreaOfInterest * projection(rho_f, eta_i) * dx - AreaOfInterest * vol_frac * dx
sensitivities_vol_constraint = derivative(vol_constraint, rho_f)

# NSR only for fluid flow
NSR = rhofluid * inner(dot(u, nabla_grad(u)), v) * dx \
    + mufluid * inner(grad(u), grad(v)) * dx \
    + inner(grad(p), v) * dx + inner(div(u), q) * dx + alpha(projection(rho_f, eta_i)) * inner(u, v) * dx

NSR_mc1 = rhofluid * inner(dot(u, nabla_grad(u)), v2) * dx \
    + mufluid * inner(grad(u), grad(v2)) * dx \
    + inner(grad(p), v2) * dx + inner(div(u), q2) * dx + alpha(projection(rho_f, eta_i)) * inner(u, v2) * dx
    
NSR_mc2 = rhofluid * inner(dot(u, nabla_grad(u)), v3) * dx \
    + mufluid * inner(grad(u), grad(v3)) * dx \
    + inner(grad(p), v3) * dx + inner(div(u), q3) * dx + alpha(projection(rho_f, eta_i)) * inner(u, v3) * dx

UFL_State = NSR
UFL_State_mc1 = NSR_mc1
UFL_State_mc2 = NSR_mc2

UFL_AL = ObjFunctional + UFL_State
UFL_AL_mc1 = m_constraint_1 + UFL_State_mc1
UFL_AL_mc2 = m_constraint_2 + UFL_State_mc2

NSR_fwd = derivative(UFL_State, w_adj, TestFunction(FlowSpace))
NSR_adj = derivative(UFL_AL, w_fwd, TestFunction(FlowSpaceAdj))
NSR_adj_mc1 = derivative(UFL_AL_mc1, w_fwd, TestFunction(FlowSpaceAdj_mc1))
NSR_adj_mc2 = derivative(UFL_AL_mc2, w_fwd, TestFunction(FlowSpaceAdj_mc2))

ddx = derivative(UFL_AL, rho_f)
sensitivities_out1_constraint = derivative(UFL_AL_mc1, rho_f)
sensitivities_out2_constraint = derivative(UFL_AL_mc2, rho_f)

##################################################################
#                   Create Directories                            
##################################################################

if os.path.exists('rho_') and os.path.isdir('rho_'):
    shutil.rmtree('rho_')
os.mkdir('rho_')
if os.path.exists('rho_f') and os.path.isdir('rho_f'):
    shutil.rmtree('rho_f')
os.mkdir('rho_f')
if os.path.exists('rho_p') and os.path.isdir('rho_p'):
    shutil.rmtree('rho_p')
os.mkdir('rho_p')
if os.path.exists('u_') and os.path.isdir('u_'):
    shutil.rmtree('u_')
os.mkdir('u_')
if os.path.exists('p_') and os.path.isdir('p_'):
    shutil.rmtree('p_')
os.mkdir('p_')
if os.path.exists('s_') and os.path.isdir('s_'):
    shutil.rmtree('s_')
os.mkdir('s_')
if os.path.exists('design_') and os.path.isdir('design_'):
    shutil.rmtree('design_')
os.mkdir('design_')

rho_out = File("rho_/plot_rho.pvd")
rhof_out = File("rho_f/plot_rho_f.pvd")
rhop_out = File("rho_p/plot_rho_p.pvd")
u_out = File("u_/plot_u.pvd")
p_out = File("p_/plot_p.pvd") 
s_out = File("s_/plot_s.pvd")

txtout = open("OptimizationLog.txt","w")
header1 = "Iteration     "
header2 = "Obj.Function  "
header21 = "Obj. Conv.    "
header6 = "Vol_frac      "

txtout.write("{} {} {} {} {}\r\n".format(header1, header2, header21, header6, \
              strftime("%a, %d %b %Y %H:%M:%S", localtime())))
txtout.close()

#############
#     Optimization
#############

# qpen = [0.01, 0.1, 0.1, 1.0]
qpen = [0.1]
b_proj = [1.0]
# qpen = [0.01, 0.03, 0.1, 0.3, 1.0, 1.0, 1.0]
# b_proj = [1.0, 10.0**0.25, 10.0**0.5, 10.0**0.75, 10.0, 10.0, 10.0]


assign(rho, interpolate(Expression('(x[0] >= 0.0 && x[0] <= Length && x[1] >= 0.0) ? 0.5 : 1.0', Length = Length, degree = 0), DensitySpace))

iter_count = 0
inner_count = 0
updateStage = 0
previousObjective = 0.0

#MMA Parameters
mmma = 3
# nmma = mesh.num_cells() activefix
nmma = len(ActiveDV)

xval = np.zeros((nmma, 1))
# xval[:,0] = rho.vector() activefix
xval[:,0] = rho.vector()[ActiveDV]
xold1 = np.zeros((nmma, 1))
xold2 = np.zeros((nmma, 1))
low = np.zeros((nmma, 1))
upp = np.zeros((nmma, 1))

a0 = 1.0
a = np.zeros((mmma, 1))
c = 1.0e5*np.ones((mmma, 1))
d = np.zeros((mmma, 1))

xmin = np.zeros((nmma, 1))
xmax = np.ones((nmma, 1))

move = 0.5

xmma = np.zeros((nmma, 1))
df0dx = np.zeros((nmma, 1))
fval = np.zeros((mmma, 1))
dfdx = np.zeros((mmma, nmma))

convergenceTolerance = 1e-5

for jj in range(len(qpen)):

    inner_count = 0
    convergenceHistory = 0

    qpenal.assign(qpen[jj])
    betaproj.assign(b_proj[jj])

    objectiveConvergence = False

    #OPTIMIZATION LOOP
    while inner_count <= 100 and objectiveConvergence == False:

        ###########################
        #           FWD
        #########################
        # set_trace()
        rho_f = pdefilter(rho, rho_f)
        rho_p.vector()[:] = project(projection(rho_f, eta_i), DensitySpace).vector()[:]
        rho_out << rho
        rhof_out << rho_f
        rhop_out << rho_p
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

        ##############################
        #        ADJ
        ########################################

        ####### SOLVE NSR_adj
        Jac_NSR_adj = derivative(NSR_adj, w_adj)
        problem_NSR_adj = NonlinearVariationalProblem(NSR_adj, w_adj, bc_NS_adj, Jac_NSR_adj)
        solver_NSR_adj = NonlinearVariationalSolver(problem_NSR_adj)
        solver_NSR_adj.parameters['nonlinear_solver'] = 'snes'
        solver_NSR_adj.parameters["snes_solver"]["report"] = True
        solver_NSR_adj.parameters['snes_solver']['linear_solver'] = 'lu'
        solver_NSR_adj.parameters['snes_solver']['relative_tolerance'] = 1.0e-4
        solver_NSR_adj.solve()

        ####### SOLVE NSR_adj_mc1
        Jac_NSR_adj_mc1 = derivative(NSR_adj_mc1, w_adj_mc1)
        problem_NSR_adj_mc1 = NonlinearVariationalProblem(NSR_adj_mc1, w_adj_mc1, bc_NS_adj_mc1, Jac_NSR_adj_mc1)
        solver_NSR_adj_mc1 = NonlinearVariationalSolver(problem_NSR_adj_mc1)
        solver_NSR_adj_mc1.parameters['nonlinear_solver'] = 'snes'
        solver_NSR_adj_mc1.parameters["snes_solver"]["report"] = True
        solver_NSR_adj_mc1.parameters['snes_solver']['linear_solver'] = 'lu'
        solver_NSR_adj_mc1.parameters['snes_solver']['relative_tolerance'] = 1.0e-4
        solver_NSR_adj_mc1.solve()
        
        ####### SOLVE NSR_adj_mc2
        Jac_NSR_adj_mc2 = derivative(NSR_adj_mc2, w_adj_mc2)
        problem_NSR_adj_mc2 = NonlinearVariationalProblem(NSR_adj_mc2, w_adj_mc2, bc_NS_adj_mc2, Jac_NSR_adj_mc2)
        solver_NSR_adj_mc2 = NonlinearVariationalSolver(problem_NSR_adj_mc2)
        solver_NSR_adj_mc2.parameters['nonlinear_solver'] = 'snes'
        solver_NSR_adj_mc2.parameters["snes_solver"]["report"] = True
        solver_NSR_adj_mc2.parameters['snes_solver']['linear_solver'] = 'lu'
        solver_NSR_adj_mc2.parameters['snes_solver']['relative_tolerance'] = 1.0e-4
        solver_NSR_adj_mc2.solve()

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
        #s_out << filteredGradient
        np.savetxt("design_/rho_{:03}.txt".format(iter_count), rho.vector()[:])

        ##############################
        #           MMA
        ###########################

        # df0dx[:,0] = filteredGradient.vector() activefix
        df0dx[:,0] = filteredGradient.vector()[ActiveDV]

        fval[0,:] = assemble(vol_constraint)
        unfiltered_s_vol.vector()[:] = assemble(sensitivities_vol_constraint)[:]
        filtered_s_vol = pdefilter(unfiltered_s_vol, filtered_s_vol)
        dfdx[0,:] = filtered_s_vol.vector()[ActiveDV]
            
        fval[1,:] = assemble(m_constraint_1) - 1
        unfiltered_s_out1.vector()[:] = assemble(sensitivities_out1_constraint)
        filtered_s_out1 = pdefilter(unfiltered_s_out1, filtered_s_out1)
        dfdx[1,:] = filtered_s_out1.vector()[ActiveDV]
        
        fval[2,:] = assemble(m_constraint_2) - 1
        unfiltered_s_out2.vector()[:] = assemble(sensitivities_out2_constraint)
        filtered_s_out2 = pdefilter(unfiltered_s_out2, filtered_s_out2)
        dfdx[2,:] = filtered_s_out2.vector()[ActiveDV]

        # if iter_count%5 == 0:
        #     set_trace()
                                      
        xmma,ymma,zmma,lam,xsi,eta,mufluid,zet,s,low,upp = \
            mmasub(mmma,nmma,iter_count,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        # rho.vector()[:] = xmma[:,0].copy() activefix
        rho.vector()[ActiveDV] = xmma[:,0].copy()
        
        txtout = open("OptimizationLog.txt","a")
        txtout.write("{:02d}.{:03d}         {:.12}   {:.12}   {:.12}   {}\r\n".format(jj, inner_count, str(round(f0val, 10)).ljust(12), str(round(objConvergenceCheck, 10)).ljust(12),\
              str(round(assemble(AreaOfInterest * rho * dx) / volume, 10)).ljust(12), \
              strftime("%a, %d %b %Y %H:%M:%S", localtime())))
        txtout.close()
        print("The volume fraction is: ", assemble(AreaOfInterest * rho * dx) / volume)
        print("The mass flow fraction 1 is: ", outflow_outlet1 * assemble(m_constraint_1))
        print("The mass flow fraction 2 is: ", outflow_outlet2 * assemble(m_constraint_2))
        
        inner_count += 1
        iter_count += 1
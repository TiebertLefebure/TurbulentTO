## TurbulentTO: Topology Optimization for Turbulent Flows

**TurbulentTO** is a custom finite-element CFD and topology optimization framework built in [FEniCS](https://fenicsproject.org/). It is designed to optimize fluid manifolds in the turbulent regime to minimize pressure drop. It utilizes a highly stabilized **Spalart-Allmaras (SA) one-equation turbulence model** coupled with a continuous adjoint sensitivity solver.

## Repository Structure

* **`TurbulenceModels/`**: The standalone Spalart-Allmaras fluid dynamics engine. Includes the IPCS fractional-step solver, SUPG stabilization, and Yoon's PDE-based Eikonal wall-distance solver.
* **`FluidTO/`**: The topology optimization architecture. Includes the continuous adjoint solver (via the Frozen Turbulence assumption), Brinkman penalization (Fictitious Domain), Helmholtz density filtering, and MMA optimizer integration.

## Key Features

* **Wall-Resolved Turbulence:** Integrates through the viscous sublayer ($y^+ \approx 1$) without empirical wall functions, ensuring stability as boundaries evolve.
* **PDE-Based Wall Distance:** Replaces non-differentiable geometric searches with a relaxed Eikonal equation for stable adjoint derivation.
* **Rigorous Verification:** The SA fluid solver is quantitatively verified against **Ansys Fluent** (Validation Case VMFL048) for predicting flow separation in $De \approx 10,000$ curvature.

## Dependencies
* **FEniCS** (dolfin, UFL)
* **Gmsh** (Meshing)
* **SciPy / NumPy / Matplotlib**
* **MPI4Py** (Parallel execution)

## Academic Use
Developed as part of a Master's thesis on thermal-fluid topology optimization. Please cite this repository if you utilize this codebase or the modified Eikonal formulations in your research.






# TurbulentTO

Research code for density-based topology optimization of incompressible internal flows with a wall-resolved Spalart-Allmaras turbulence model in FEniCS.

The repository currently supports the thesis workflow around turbulent topology optimization, not the older Borrvall-style benchmark-only setup. The active optimization cases are the Alexandersen/Bayat pipe bend and U-bend and the Yoon diffuser, with laminar reference cases and standalone turbulence-solver verification scripts kept alongside them.

## Repository layout

- `FluidTO/`: topology-optimization code.
  - `TurbulentTO_Frozen.py`: turbulent optimization driver with a frozen-turbulence adjoint.
  - `LaminarTO.py`: laminar reference optimization driver.
  - `Configs_Frozen/`: turbulent frozen-SA case definitions.
  - `Configs_Laminar/`: laminar reference case definitions.
  - `Meshes/`: XDMF/HDF5 meshes and mesh-generation scripts for the optimization cases.
  - `Results_Frozen/` and `Results_Laminar/`: generated optimization output folders.
- `TurbulenceModels/`: standalone forward solvers used to verify the FEniCS turbulence implementation before embedding it in topology optimization.
  - Steady and transient Spalart-Allmaras drivers.
  - Laminar and low-Re k-epsilon comparison drivers.
  - Channel, backward-facing-step, and U-bend validation meshes/configs.
- `Master-Thesis/`: LaTeX thesis sources, figures, bibliography, and compiled thesis artifacts.
- `compose/frozen/`: minimal Ubuntu/FEniCS Docker image for running the frozen optimization driver.
- `scripts/`: convenience wrappers for launching frozen-SA optimization cases.

## Main capabilities

- Wall-resolved Spalart-Allmaras forward solver with Picard coupling and IPCS/SNES flow solves.
- Frozen-turbulence continuous adjoint for topology optimization: the adjoint differentiates the final flow residual while holding the converged SA working variable and wall-distance field fixed.
- Brinkman density penalization with MMA updates.
- DG0 Helmholtz filtering, Heaviside projection, continuation schedules, passive-fluid regions, and per-cell design bounds.
- Topology-dependent SA and reciprocal wall-distance penalties so optimizer-created solids act as walls.
- Sensitivity diagnostics, optimization checkpoints, VTK/PVD output, and thesis-oriented postprocessing fields.

## Active optimization cases

Turbulent frozen-SA cases:

- `FluidTO/Configs_Frozen/Config_PipeBendAlexandersen_Frozen.py`
- `FluidTO/Configs_Frozen/Config_UBendAlexandersen_Frozen.py`
- `FluidTO/Configs_Frozen/Config_DiffuserYoon_Frozen.py`
- `FluidTO/Configs_Frozen/*_Sensitivity.py` for finite-difference/Taylor sensitivity checks.

Laminar reference cases:

- `FluidTO/Configs_Laminar/Config_PipeBendAlexandersen_Laminar.py`
- `FluidTO/Configs_Laminar/Config_UBendAlexandersen_Laminar.py`
- `FluidTO/Configs_Laminar/Config_DiffuserYoon_Laminar.py`

The legacy Borrvall diffuser implementation remains in `FluidTO/DiffuserKULeuvenTO.py`, but it is not the primary entry point for the current thesis results.

## Dependencies

This code targets the legacy FEniCS stack used by `dolfin`/UFL, not FEniCSx.

Core runtime:

- FEniCS/dolfin with PETSc, MPI, and MUMPS support
- `mpi4py`
- `numpy`
- `scipy`
- `matplotlib`

Mesh generation:

- `gmsh`
- `meshio`

The Dockerfile in `compose/frozen/Dockerfile` installs FEniCS from the Ubuntu PPA. Extra Python packages such as `meshio` are only needed when regenerating meshes, not for running checked-in meshes.

## Running optimization cases

From the repository root, run a frozen turbulent optimization case through the helper:

```bash
./scripts/run-frozen.sh Configs_Frozen/Config_DiffuserYoon_Frozen.py
```

Equivalent direct command:

```bash
cd FluidTO
python3 TurbulentTO_Frozen.py --config Configs_Frozen/Config_DiffuserYoon_Frozen.py
```

Laminar references use `LaminarTO.py`:

```bash
cd FluidTO
python3 LaminarTO.py --config Configs_Laminar/Config_PipeBendAlexandersen_Laminar.py
```

For MPI runs, launch the same command with `mpirun`:

```bash
cd FluidTO
mpirun -np 4 python3 TurbulentTO_Frozen.py --config Configs_Frozen/Config_PipeBendAlexandersen_Frozen.py
```

Resume behavior is controlled in each config with `RESUME_OPTIMIZATION`; it can also be overridden with `FLUIDTO_RESUME` or `TURBULENTTO_RESUME`.

## Running with Docker

Build the FEniCS image:

```bash
docker build -t turbulentto-fenics -f compose/frozen/Dockerfile .
```

Run a case with the repository mounted into the container:

```bash
docker run --rm -it \
  -v "$PWD":/work \
  -w /work/FluidTO \
  turbulentto-fenics \
  TurbulentTO_Frozen.py --config Configs_Frozen/Config_DiffuserYoon_Frozen.py
```

FEniCS JIT and temporary files are redirected to `FluidTO/.runtime/` by `Runtime_Setup.py` so shared mounts do not become the cache location.

## Standalone turbulence-model verification

The scripts in `TurbulenceModels/` are separate forward-flow verification runs. Run them from inside the `TurbulenceModels` directory because their config paths are relative to that folder:

```bash
cd TurbulenceModels
python3 ChannelSimulation_SpalartAllmaras_Steady.py
python3 UBendSimulation_SpalartAllmaras_Steady.py
python3 BackStepSimulation_SpalartAllmaras_Steady.py
```

Outputs are written under `TurbulenceModels/Results/`.

## Meshes

Checked-in XDMF/HDF5 meshes are available for the active cases. Regenerate only when changing geometry or wall resolution:

```bash
python3 FluidTO/Meshes/DiffuserYoon/generate_diffuser_yoon_yplus1.py
python3 FluidTO/Meshes/PipeBendAlexandersen/generate_pipe_bend_alexandersen_sa_yplus1.py
python3 FluidTO/Meshes/UBendAlexandersen/generate_u_bend_alexandersen_sa_yplus1.py
```

The standalone solver meshes have their own generators under `TurbulenceModels/Meshes/`.

## Outputs and checkpoints

Generated result folders are intentionally ignored by Git:

- `FluidTO/Results_Frozen/`
- `FluidTO/Results_Laminar/`
- `TurbulenceModels/Results/`
- `FluidTO/.runtime/`

Optimization runs write design fields, projected densities, state fields, logs, objective histories, checkpoint files, and selected sensitivity diagnostics depending on the case config.

## Academic context

This repository accompanies a master's thesis on turbulent internal-flow topology optimization. The thesis sources in `Master-Thesis/` document the numerical method, solver verification, benchmark definitions, sensitivity checks, and independent Ansys Fluent validation of extracted geometries.

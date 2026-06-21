# TurbulentTO

Topology optimization of incompressible internal flows using legacy FEniCS (`dolfin`). The repository includes a frozen-Spalart-Allmaras turbulent solver, a laminar solver, and steady Spalart-Allmaras forward-solver verification cases.

## Main optimization files

All topology-optimization files are in `FluidTO/`.

- `TurbulentTO_Frozen.py`: turbulent topology-optimization driver using a frozen-turbulence adjoint.
- `LaminarTO.py`: laminar topology-optimization driver.
- `TurbulenceModel_SpalartAllmaras_TO_Frozen.py`: Spalart-Allmaras model used by the turbulent driver.
- `Utilities_TurbulentTO_Frozen.py`: wall-distance and turbulent penalization helpers.
- `Utilities_SharedTO.py`: mesh, boundary, logging, checkpoint, and output helpers shared by both drivers.
- `mma.py`: Method of Moving Asymptotes optimizer.

## Optimization cases

The following cases have configurations and meshes in the repository.

### Turbulent cases

Use `TurbulentTO_Frozen.py` for all three cases.

- Yoon diffuser: `Configs_Frozen/Config_DiffuserYoon_Frozen.py` with `Meshes/DiffuserYoon/`.
- Alexandersen pipe bend: `Configs_Frozen/Config_PipeBendAlexandersen_Frozen.py` with `Meshes/PipeBendAlexandersen/`.
- Alexandersen U-bend: `Configs_Frozen/Config_UBendAlexandersen_Frozen.py` with `Meshes/UBendAlexandersen/`.

### Laminar cases

Use `LaminarTO.py` for all three cases.

- Yoon diffuser: `Configs_Laminar/Config_DiffuserYoon_Laminar.py` with `Meshes/DiffuserYoon/`.
- Alexandersen pipe bend: `Configs_Laminar/Config_PipeBendAlexandersen_Laminar.py` with `Meshes/PipeBendAlexandersen/`.
- Alexandersen U-bend: `Configs_Laminar/Config_UBendAlexandersen_Laminar.py` with `Meshes/UBendAlexandersen/`.

Finite-difference and Taylor sensitivity checks use these configurations:

- Yoon diffuser: `Configs_Frozen/Config_DiffuserYoon_Frozen_Sensitivity.py`.
- Alexandersen pipe bend: `Configs_Frozen/Config_PipeBendAlexandersen_Frozen_Sensitivity.py`.

Each sensitivity configuration imports its corresponding base configuration, so both files are required.

## Required mesh files

Keep each `.xdmf` file together with its matching `.h5` file.

- Yoon diffuser: `mesh_yoon_yplus1.xdmf` + `.h5`, `cell_yoon_yplus1.xdmf` + `.h5`, and `facet_yoon_yplus1.xdmf` + `.h5`.
- Alexandersen pipe bend: `mesh_yplus1.xdmf` + `.h5` and `cell_yplus1.xdmf` + `.h5`.
- Alexandersen U-bend: `mesh_yplus1.xdmf` + `.h5` and `cell_yplus1.xdmf` + `.h5`.

The mesh-generation scripts are stored in the same geometry folders.

## Running a case

Use the solver and configuration from the case list:

```bash
cd FluidTO
python3 <solver-file> --config <configuration-file>
```

Prefix the command with `mpirun -np <process-count>` for an MPI run.

## Steady Spalart-Allmaras forward-solver cases

`TurbulenceModels/` contains two steady Spalart-Allmaras verification cases separate from topology optimization.

- Channel: `ChannelSimulation_SpalartAllmaras_Steady.py`, `Configs/ConfigChannel_SpalartAllmaras_Steady.py`, and `Meshes/Channel/`.
- U-bend: `UBendSimulation_SpalartAllmaras_Steady.py`, `Configs/ConfigUBend_SpalartAllmaras_Steady.py`, and `Meshes/U-Bend/`.

Both use `TurbulenceModel_SpalartAllmaras.py`, `SA_Steady_IPCS_Picard_Solver.py`, and `Utilities.py`.

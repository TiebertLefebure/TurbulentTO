# TopologyOptimization 

Laminar & Turbulent Topology Optimization

# --------------------------------------------------------------------------------------------------------------------- #
### Turbulent Topology Optimization uses the "frozen turbulence" assumption and the Spalart-Allmaras turbulence model ###
# --------------------------------------------------------------------------------------------------------------------- #


Paper: 
Borrvall 2003 Topology optimization of fluids in Stokes flow.pdf


1. Borrvall, 2003: Diffuser case -> Laminar 

- DiffuserBorrvallTO.py (original KU Leuven FEniCS code)
- mma.py

`python3 DiffuserBorrvallTO.py`




2. Borrvall, 2003: Pipe Bend case -> Laminar & Turbulent

Laminar:
- Config_PipeBendBorrvall_LaminarTO.py
- Borrvall_LaminarTO.py
- mma.py

`python3 Borrvall_LaminarTO.py --config Config_PipeBendBorrvall_LaminarTO`

Turbulent:
- Config_PipeBendBorrvall_TurbulentTO.py
- Borrvall_TurbulentTO.py
- TurbulenceModel_SpalartAllmaras_TO.py
- mma.py

`python3 Borrvall_TurbulentTO.py --config Config_PipeBendBorrvall_TurbulentTO`




3. Borrvall, 2003: Double Pipe case (wide domain, delta=1.5) -> Laminar & Turbulent

Laminar:
- Config_DoublePipeBorrvall_LaminarTO.py
- Borrvall_LaminarTO.py
- mma.py

`python3 Borrvall_LaminarTO.py --config Config_DoublePipeBorrvall_LaminarTO`

Turbulent:
- Config_DoublePipeBorrvall_TurbulentTO.py
- Borrvall_TurbulentTO.py
- mma.py

`python3 Borrvall_TurbulentTO.py --config Config_DoublePipeBorrvall_TurbulentTO`






#!/usr/bin/env python3
"""Generate defense-preparation PDFs from the thesis source."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = ROOT / "Master-Thesis"
ANNOTATION_DIR = THESIS_DIR / "DefensePrep"


INCLUDED_TEX_FILES = [
    "chapter-1-introduction",
    "chapter-2",
    "chapter-3",
    "chapter-4",
    "chapter-5",
    "chapter-6-results",
    "chapter-7-conclusion",
    "appendix-A-SA-Validity-Cases-Details",
    "appendix-B-SA-FEniCS-Solver-Support",
    "appendix-B-SA-Verification-Benchmarks-Details",
    "appendix-C-TO-Methodology-Support",
    "appendix-D-TO-Benchmarks-Details",
]


def latex_text(text: str) -> str:
    """Escape plain-text Q&A material for LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


ANNOTATIONS = [
    {
        "file": "chapter-1-introduction",
        "target": (
            "The broader motivation is thermal-management design, but the model problem treated here is intentionally hydraulic. "
            "The thesis does not solve the energy equation, does not perform conjugate heat-transfer optimization, and does not minimize temperature. "
            "It instead considers flow-only bends, U-bends, and diffusers, where the relevant behavior is pressure loss, viscous dissipation, separation, and the response of the turbulence closure. "
            "Establishing that hydraulic baseline is necessary before the same optimization framework can be trusted for turbulent heat-transfer problems."
        ),
        "question": "Why did you leave out heat transfer if the motivation is thermal management?",
        "answer": (
            "Because the hydraulic turbulent TO problem is already the unstable part of the later CHT problem: the solver must handle RANS-SA coupling, evolving walls, Brinkman penalization, and adjoint sensitivities. "
            "Adding the energy equation before verifying that baseline would mix thermal errors with unresolved momentum-model errors."
        ),
    },
    {
        "file": "chapter-1-introduction",
        "target": (
            "To mitigate this, gradient-based turbulent optimizations frequently rely on the ``frozen turbulence'' assumption, where the turbulent viscosity field is updated in the forward solve but treated as fixed during sensitivity assembly."
        ),
        "question": "Is frozen turbulence just a convenience that makes the gradients physically wrong?",
        "answer": (
            "It is a controlled approximation, not a claim of full RANS sensitivity. The forward solve still updates turbulence for every design; the adjoint omits the infinitesimal derivative of nu_t and wall distance. "
            "The thesis therefore verifies the frozen derivative internally and validates extracted designs independently instead of presenting the gradient as a fully coupled turbulent adjoint."
        ),
    },
    {
        "file": "chapter-2",
        "target": (
            "Equation~\\ref{eq:boussinesq} also defines an important limitation for the present work. "
            "Since $\\mu_t$ is scalar, the modeled stress is aligned with the mean strain-rate tensor and cannot independently represent Reynolds-stress anisotropy. "
            "This is acceptable for many attached shear flows, but it becomes weaker for flows with strong curvature, rotation, or separation. "
            "Curved ducts and bends are therefore critical test cases because centrifugal effects redistribute momentum across the cross-section and can create secondary motion that a scalar eddy-viscosity closure cannot reproduce exactly \\cite{Rumsey2000,Rumsey2001}."
        ),
        "question": "If scalar eddy viscosity is known to fail in anisotropic flows, why use it at all?",
        "answer": (
            "Because the thesis uses SA as a practical optimization model inside a stated envelope, not as closure-independent truth. "
            "A Reynolds-stress model or LES would be harder to stabilize inside density-based TO and much more expensive. The defense is the evidence chain: identify the limitation, verify the implementation, and cross-check final geometries with SST k-omega."
        ),
    },
    {
        "file": "chapter-2",
        "target": (
            "The Dean number is academically relevant here because it separates curvature severity from Reynolds number alone. "
            "However, it should not be interpreted as a universal threshold for SA validity. "
            "The breakdown of an eddy-viscosity model depends on geometry, separation, mesh resolution, boundary conditions, and the validation quantity being compared. "
            "In this thesis, the Dean number is used only as a rough physical descriptor of the bend cases; the final judgment still comes from profile comparisons and independent CFD validation."
        ),
        "question": "Can you really defend the De approximately 15000 and 18000 distinction?",
        "answer": (
            "Only as case-specific evidence. The thesis does not claim a universal Dean-number cutoff. "
            "The point is that two similar-Re Ansys verification cases show different SA behavior when curvature changes, which warns that curvature severity matters and must be checked by profiles and validation metrics."
        ),
    },
    {
        "file": "chapter-3",
        "target": (
            "The solver instead applies \\textbf{Picard iteration}, or successive substitution, a standard segregated strategy for RANS calculations \\cite{Ferziger2012,Versteeg2007}. "
            "At iteration $k$, the velocity-pressure solve uses the eddy viscosity assembled from the previous SA field. "
            "The updated velocity is then used in the next SA solve, after which the eddy viscosity is recomputed. "
            "Repeating this sequence forms the outer fixed-point loop for the coupled RANS-SA state."
        ),
        "question": "Why not solve the complete RANS-SA system monolithically?",
        "answer": (
            "A monolithic solve is cleaner mathematically but harder to make robust while the topology changes and intermediate designs create near-solid regions. "
            "Picard decoupling gives a stable forward state for the design loop. Where the final flow residual is differentiated, the code can use a steady SNES solve so the adjoint is assembled from a consistent final velocity-pressure residual."
        ),
    },
    {
        "file": "chapter-3",
        "target": (
            "For convection-dominated internal-flow cases, the implementation may augment the SA transport solve with SUPG stabilization \\cite{Brooks1982}. "
            "This stabilization is applied to the transported working variable $\\tilde{\\nu}$ only; it is not a modification of the RANS momentum equation and it is not an additional turbulence-model source term."
        ),
        "question": "Does SUPG change the turbulence model and bias your optimized shapes?",
        "answer": (
            "It changes the numerical transport discretization of nu_tilde, not the physical SA closure. "
            "The purpose is to suppress element-scale oscillations in convection-dominated SA transport. It is still a numerical modeling choice, so it belongs in the solver setup and should remain fixed when comparing designs."
        ),
    },
    {
        "file": "chapter-3",
        "target": (
            "However, to ensure this verification solver translates into the topology optimization code, where solid boundaries evolve continuously, the implementation uses a PDE-based approach."
        ),
        "question": "Why solve a PDE for wall distance instead of computing exact geometric distance?",
        "answer": (
            "Exact distance is simple for a fixed wall but becomes awkward and non-smooth when the wall is created by a density field. "
            "The relaxed reciprocal-distance PDE gives a differentiable field that can be penalized in topology-created solids, which is more compatible with FEniCS/UFL and the optimization loop."
        ),
    },
    {
        "file": "chapter-4",
        "target": (
            "Because the channel is driven by a prescribed pressure difference, $\\Delta p=2.0$ Pa is identical by construction and is not an independent comparison metric."
        ),
        "question": "Why do you report bulk velocity instead of pressure drop for the channel verification?",
        "answer": (
            "The pressure drop is imposed, so matching it would not verify the solver. "
            "The meaningful response is the flow rate or bulk velocity produced by that pressure gradient, plus the velocity and nu_tilde profiles."
        ),
    },
    {
        "file": "chapter-4",
        "target": (
            "Since the FEniCS domain is mathematically planar, it cannot resolve true 3D Dean secondary vortices. "
            "The Dean number is therefore retained only as an inherited descriptor of the corresponding 3D curved-pipe operating point, not as evidence that the 2D model contains Dean-vortex physics."
        ),
        "question": "Does the 2D U-bend verification prove your solver captures Dean-vortex physics?",
        "answer": (
            "No. It verifies the 2D wall-resolved SA implementation against a matched 2D Ansys baseline. "
            "Dean number is retained to connect the setup to the original physical case, but secondary vortices and 3D separation are explicitly outside the present model scope."
        ),
    },
    {
        "file": "chapter-5",
        "target": (
            "The fields $\\gamma_{\\mathrm{lower}}$ and $\\gamma_{\\mathrm{upper}}$ are not additional optimizer variables. "
            "They are cellwise lower and upper bounds used to impose prescribed non-design regions: freely optimized cells use bounds $0$ and $1$, fixed-fluid cells use bounds $1$ and $1$, and fixed-solid cells use bounds $0$ and $0$. "
            "This keeps inlet ducts, outlet ducts, separator bars, and other case-specific regions fixed during the optimization. "
            "The same masking convention is used for objective and volume integrations; the relevant regions are shown with each benchmark setup in Chapter~\\ref{ch:benchmark_results}."
        ),
        "question": "Could the optimizer cheat by changing inlet, outlet, or separator regions?",
        "answer": (
            "No. Those cells are removed from the active design through lower and upper density bounds and through the objective/volume masks. "
            "The optimizer only updates active DG0 design variables; fixed fluid or fixed solid regions are restored after filtering/projection."
        ),
    },
    {
        "file": "chapter-5",
        "target": (
            "Topology-created solid regions should not retain turbulent working-variable content. "
            "A sink term is therefore added to the SA transport equation \\cite{Yoon2016,Dilgen2018}:"
        ),
        "question": "Why add a topology penalty to nu_tilde instead of relying only on Brinkman damping?",
        "answer": (
            "Brinkman damping suppresses velocity, but it does not automatically make the turbulence working variable behave like it is adjacent to a wall. "
            "Without a nu_tilde sink, solid-like regions could retain artificial eddy viscosity. The penalty makes topology-created solids more wall-like for the SA model."
        ),
    },
    {
        "file": "chapter-5",
        "target": (
            "Thus, forcing $G$ toward $G_0$ collapses the recovered distance toward zero in topology-created solid. "
            "As with the SA working-variable penalty, $\\alpha_G$ can be continued by schedule, while $N_G$ is kept fixed in the reported cases."
        ),
        "question": "What is the physical meaning of forcing the reciprocal wall-distance variable in solid regions?",
        "answer": (
            "It is not a new physical wall law; it is a topology-consistent way to tell the SA model that a density-created obstacle should act like a nearby wall. "
            "Driving G to G0 makes the recovered wall distance approach zero, increasing wall-destruction behavior around emerging solids."
        ),
    },
    {
        "file": "chapter-5",
        "target": (
            "The two objectives target the same physical idea--reducing hydraulic losses--but they expose it differently to the optimizer. "
            "The dissipation objective $J_D$ measures losses as a volume integral of viscous dissipation and Brinkman drag in the objective region. "
            "The pressure objective $J_p$ measures the mean inlet pressure directly; when the inlet flow rate and outlet reference pressure are fixed, minimizing $J_p$ is equivalent to minimizing the mean pressure drop, and the corresponding hydraulic power loss is proportional to $\\Delta p\\,Q$. "
            "In the final turbulent benchmark set, the pipe-bend and U-bend use $J_p$, while the diffuser uses $J_D$."
        ),
        "question": "Is it inconsistent to use pressure objective for bends and dissipation for the diffuser?",
        "answer": (
            "The objectives are both hydraulic-loss objectives, but they match different reference benchmarks and boundary-value problems. "
            "The thesis avoids hiding this by reporting common sharp-geometry validation metrics, pressure drop and viscous dissipation, for all final designs."
        ),
    },
    {
        "file": "chapter-5",
        "target": (
            "The implementation therefore uses a frozen-turbulence adjoint of the final steady velocity-pressure residual. "
            "The forward loop still solves the topology-dependent wall-distance and SA equations, but their infinitesimal design derivatives are omitted during gradient assembly."
        ),
        "question": "What exactly is frozen in the frozen-turbulence adjoint?",
        "answer": (
            "The converged SA working variable, eddy viscosity, and wall-distance fields are treated as fixed coefficients while differentiating the final velocity-pressure residual with respect to density. "
            "The velocity and pressure response to Brinkman changes is still included through the adjoint."
        ),
    },
    {
        "file": "chapter-5",
        "target": (
            "Because Eq.~\\ref{eq:frozen_design_gradient} is built under the frozen-turbulence assumption, the finite-difference checks in Appendix~\\ref{app:C_topology_optimization_benchmarks} verify the implemented frozen derivative rather than a fully coupled turbulence derivative."
        ),
        "question": "What do your finite-difference and Taylor checks actually prove?",
        "answer": (
            "They prove that the derivative supplied to MMA is consistent with the implemented frozen model and the selected solver path. "
            "They do not prove that the omitted coupled turbulence terms are negligible, and the thesis says that explicitly."
        ),
    },
    {
        "file": "chapter-6-results",
        "target": (
            "All extracted geometries are then evaluated as sharp-wall Ansys Fluent models, where the Brinkman penalty is removed and the validation metrics are recomputed on the exported flow domain."
        ),
        "question": "Why validate sharp geometries if the optimizer solved a diffuse Brinkman problem?",
        "answer": (
            "Because a manufactured design would be a sharp-wall flow path, not a porous density field. "
            "The diffuse field is an optimization device. Sharp-geometry validation tests whether the design transfers after thresholding, smoothing, and removing the artificial Brinkman resistance."
        ),
    },
    {
        "file": "chapter-6-results",
        "target": (
            "The reported FEniCS value is the endpoint of a short $70$-iteration run without continuation. "
            "It is used here as a provisional design state for comparing topology and validation metrics, not as a converged optimum. "
            "The convergence history in Figure~\\ref{fig:alexandersen_pipebend_histories} shows that the normalized pressure objective is still monotonically decreasing, so the run is promising but should be extended with more MMA iterations and continuation."
        ),
        "question": "Can you claim the pipe-bend result is successful if the optimization was not converged?",
        "answer": (
            "The success claim is about transfer and robustness of the extracted design candidate, not final optimality. "
            "The thesis clearly labels it as provisional. The important evidence is that the candidate remains strong under Ansys-SA and SST k-omega and strongly beats the laminar reference under identical turbulent conditions."
        ),
    },
    {
        "file": "chapter-6-results",
        "target": (
            "Using the Ansys-SA result as the reference, the SST $k-\\omega$ evaluation increases the pressure drop by approximately $4.5\\%$ and the viscous dissipation by approximately $8.5\\%$. "
            "The SST $k-\\omega$ model therefore predicts slightly higher losses, consistent with a closure that is more sensitive to adverse pressure gradients, but the changes remain modest. "
            "This suggests that the extracted pipe-bend geometry is not narrowly tuned to the SA model. "
            "It does not, however, establish that the underlying FEniCS optimization has converged."
        ),
        "question": "Why is the pipe bend your strongest validation case?",
        "answer": (
            "The closure shift is modest and the laminar-vs-turbulent ranking is decisive under SST k-omega. "
            "That combination suggests the geometry is not merely exploiting an SA-specific separation prediction, even though the optimization endpoint itself is not claimed to be the final optimum."
        ),
    },
    {
        "file": "chapter-6-results",
        "target": (
            "Using the Ansys-SA result as the reference, the SST $k-\\omega$ validation predicts larger losses for the same extracted turbulent geometry: $\\Phi_D$ increases by approximately $29.4\\%$ and $\\Delta p$ by approximately $48.3\\%$. "
            "This trend is physically plausible because the fixed separator forces a compact return flow, where SST $k-\\omega$ is more responsive to adverse pressure gradients and local recirculation downstream of the separator. "
            "The U-bend validation is therefore interpreted as a closure-sensitivity result for the forced $180^\\circ$ return flow rather than as closure-independent agreement."
        ),
        "question": "Does the U-bend result undermine the method?",
        "answer": (
            "It narrows the method's validated envelope. Under SA the turbulent topology improves on the laminar reference, but SST k-omega shows strong closure sensitivity. "
            "That is exactly why the thesis frames the U-bend as a cautionary case rather than a clean success."
        ),
    },
    {
        "file": "chapter-6-results",
        "target": (
            "The FEniCS-SA values in Table~\\ref{tab:yoon_diffuser_validation_metrics} are approximately $2.21$ times larger in $\\Phi_D$ and $2.25$ times larger in $\\Delta p_s$ than the Ansys-SA sharp-geometry values, so this case cannot be claimed as quantitative cross-code agreement. "
            "The result is interpreted as a transfer limitation between the diffuse Brinkman-domain optimization diagnostic and the sharp-wall extracted geometry."
        ),
        "question": "Is the diffuser validation a failure?",
        "answer": (
            "It is a failed quantitative transfer case, but still useful evidence. "
            "It shows where diffuse-to-sharp extraction, adverse pressure gradients, and closure model form dominate the interpretation. The thesis does not claim quantitative agreement for the diffuser."
        ),
    },
    {
        "file": "chapter-6-results",
        "target": (
            "Under Ansys-SA, the turbulent topology reduces the pressure drop by $17.4\\%$ and the viscous dissipation by $11.9\\%$ relative to the pressure-outlet laminar topology. "
            "Under SST $k-\\omega$, the ranking reverses: the turbulent topology gives a $30.5\\%$ higher pressure drop and a $13.7\\%$ higher dissipation."
        ),
        "question": "How should you defend a ranking reversal in the diffuser?",
        "answer": (
            "Do not soften it. The reversal is evidence that the SA-optimized diffuser is model-form sensitive in adverse-pressure-gradient flow. "
            "The honest conclusion is that this case needs a stronger turbulence model, better extraction checks, or experimental validation before making design claims."
        ),
    },
    {
        "file": "chapter-7-conclusion",
        "target": (
            "The central conclusion is balanced. "
            "A frozen-SA density-based topology optimization framework can produce credible low-loss turbulent internal-flow design candidates when the flow remains within the closure's practical range and when the extracted geometry is independently validated. "
            "It is not a substitute for final CFD validation, experimental testing, or a higher-fidelity turbulence model in separated, strongly three-dimensional flows. "
            "Independent sharp-geometry CFD validation is the mandatory engineering step between an optimized density field and anything that could be manufactured."
        ),
        "question": "What is the one-sentence contribution of the thesis?",
        "answer": (
            "It builds and verifies an open FEniCS frozen-SA topology-optimization workflow and, more importantly, defines when its optimized turbulent internal-flow candidates should and should not be trusted."
        ),
    },
]


QA_SECTIONS = [
    (
        "Core Thesis Claims",
        [
            (
                "What is the main thesis claim?",
                "Frozen-SA density-based topology optimization can generate useful turbulent internal-flow design candidates, but only inside a validated model envelope. The pipe bend is the cleanest success; the U-bend and diffuser show closure sensitivity.",
            ),
            (
                "What is the main contribution beyond producing optimized shapes?",
                "The contribution is the evidence chain: SA model-envelope analysis, FEniCS-SA verification, frozen-gradient implementation and checks, sharp-geometry Ansys validation, and conservative interpretation of closure sensitivity.",
            ),
            (
                "Why is this not a heat-transfer thesis?",
                "The work isolates the hydraulic turbulent-flow problem. CHT would add thermal transport and thermal adjoints on top of a turbulent momentum model that first needed to be stabilized and validated.",
            ),
            (
                "What should you say if the jury asks whether the method is ready for industrial design?",
                "It is ready as an early-stage research/design-candidate generator for low-loss hydraulic layouts, not as a final certification tool. Final CFD validation and preferably experiments remain mandatory.",
            ),
            (
                "What is the strongest evidence in the thesis?",
                "The pipe-bend case: the extracted turbulent design remains robust under Ansys-SA and SST k-omega and strongly outperforms the laminar design under identical turbulent operating conditions.",
            ),
        ],
    ),
    (
        "Turbulence Modeling",
        [
            (
                "Why choose Spalart-Allmaras instead of SST k-omega or k-epsilon?",
                "SA is a one-equation wall-resolved model, which reduces nonlinear coupling and adjoint complexity. It avoids wall functions in the density-based FEniCS formulation. The trade-off is model-form sensitivity in separated or strongly curved flows.",
            ),
            (
                "What is the biggest physical limitation of SA here?",
                "It is a scalar eddy-viscosity closure. It cannot represent Reynolds-stress anisotropy, secondary Dean vortices, or separated-flow recovery with closure-independent accuracy.",
            ),
            (
                "Does the Dean number provide a hard validity threshold?",
                "No. It is a physical scale for curvature severity. The thesis uses it to organize evidence, but model validity depends on geometry, separation, mesh, boundary conditions, and the validation quantity.",
            ),
            (
                "Why compare with SST k-omega?",
                "SST k-omega is more sensitive to adverse pressure gradients and separation than SA. It is a useful independent closure stress test for extracted geometries.",
            ),
            (
                "Why not DNS or LES?",
                "They are not practical inside thousands of optimization iterations with evolving topology. RANS is the engineering compromise; the thesis then validates how far that compromise can be trusted.",
            ),
        ],
    ),
    (
        "Numerical Solver",
        [
            (
                "Why use FEniCS?",
                "FEniCS exposes weak forms and UFL derivatives directly, which is valuable for implementing the SA residual, Brinkman terms, filters, and frozen adjoint consistently.",
            ),
            (
                "Why use Picard coupling?",
                "It gives a robust segregated fixed-point solve: velocity-pressure uses the previous eddy viscosity, then SA is updated with the new velocity. That is more stable for difficult intermediate density fields than a fully monolithic RANS-SA solve.",
            ),
            (
                "Why use IPCS in the forward solver?",
                "IPCS separates tentative velocity, pressure correction, and velocity update, which is robust for incompressible flow. In the active turbulent driver it can be used for Picard updates, while SNES can polish the final residual for adjoint assembly.",
            ),
            (
                "Does SUPG change the physics?",
                "It is numerical stabilization for the transported nu_tilde equation. It suppresses element-scale oscillations in convection-dominated cases but is not an extra turbulence-production model.",
            ),
            (
                "Why is wall distance solved with a relaxed reciprocal PDE?",
                "Because topology-created walls evolve continuously. A PDE wall-distance field is smoother and can be penalized in solid-like regions, while exact geometric distance would be awkward during density optimization.",
            ),
        ],
    ),
    (
        "Topology Optimization",
        [
            (
                "Explain the density chain.",
                "MMA updates raw DG0 density gamma. The code filters it, projects it with a Heaviside map, then applies lower/upper bounds to produce physical density gamma_bar for the PDEs.",
            ),
            (
                "Why use Brinkman penalization?",
                "It converts low-density regions into porous resistance inside a fixed mesh, so the optimizer can move solid-fluid boundaries without remeshing every iteration.",
            ),
            (
                "Why add topology penalties to SA and wall distance?",
                "Brinkman resistance suppresses velocity but does not automatically make emerging solids act like walls for turbulence. The nu_tilde sink and reciprocal wall-distance penalty impose wall-like turbulence behavior near density-created solids.",
            ),
            (
                "Why use MMA?",
                "The problem has many bounded design variables and a volume constraint. MMA is a standard robust gradient-based optimizer for large constrained topology optimization problems.",
            ),
            (
                "Why use continuation?",
                "Continuation avoids jumping immediately to a sharp, difficult binary design. It gradually sharpens projection, changes RAMP behavior, adjusts move limits, and can increase topology penalties.",
            ),
            (
                "Why are J_p and J_D both used?",
                "They are both hydraulic-loss objectives. J_p directly targets mean inlet pressure under fixed flow rate; J_D targets integrated viscous and Brinkman dissipation. The chosen objective follows the benchmark setup, while validation reports common metrics for comparison.",
            ),
        ],
    ),
    (
        "Adjoint And Sensitivities",
        [
            (
                "What does frozen turbulence omit?",
                "It omits design derivatives through nu_tilde, eddy viscosity, and wall distance. The velocity-pressure response and direct Brinkman density effects are still differentiated.",
            ),
            (
                "Are the gradients validated?",
                "They are verified against finite differences and Taylor tests for the implemented frozen model. The checks do not validate a fully coupled RANS-SA topology derivative.",
            ),
            (
                "What would a fully coupled turbulent adjoint require?",
                "Adjoint equations or a discrete adjoint for velocity, pressure, nu_tilde, wall distance, clipping/relaxation, and possibly the full Picard/SNES update sequence.",
            ),
            (
                "Why not use finite differences for optimization?",
                "There are too many active DG0 cells. A central finite difference would require two forward solves per design variable per objective evaluation.",
            ),
            (
                "What is the correct defense wording for the derivative?",
                "The derivative is internally consistent with the frozen-turbulence optimization model. It is not claimed to be the exact derivative of the fully coupled turbulent physical system.",
            ),
        ],
    ),
    (
        "Results And Validation",
        [
            (
                "Why is sharp-geometry validation necessary?",
                "The optimizer solves a diffuse Brinkman problem, but a physical design is a sharp-wall geometry. Validation checks whether the extracted design still performs after removing the porous model.",
            ),
            (
                "Why is the pipe-bend result strong even though the optimization endpoint is provisional?",
                "Because the extracted geometry is robust under closure change and strongly beats the laminar design. The thesis does not claim final optimality, only credible transfer of a candidate design.",
            ),
            (
                "What does the U-bend teach?",
                "A forced 180-degree return flow is more closure-sensitive. The turbulent design helps under SA, but SST k-omega predicts much higher losses and makes the laminar comparison nearly neutral.",
            ),
            (
                "What does the diffuser teach?",
                "Outlet boundary condition and turbulence closure can dominate the ranking. The pressure-outlet diffuser improves under SA but reverses under SST k-omega, so it is a cautionary case.",
            ),
            (
                "How do you answer if someone says the diffuser invalidates the thesis?",
                "It invalidates any universal performance claim, which the thesis does not make. It strengthens the central conclusion: SA-based TO must be validated and has a limited envelope.",
            ),
            (
                "Why compare laminar-optimized designs at turbulent operating conditions?",
                "That tests whether using turbulent physics in the optimizer changes the design in a way that matters at the target operating point. It is not a validation of laminar optimization itself.",
            ),
        ],
    ),
    (
        "Python Implementation",
        [
            (
                "What are the active entry points?",
                "The active optimization drivers are FluidTO/TurbulentTO_Frozen.py and FluidTO/LaminarTO.py. Standalone verification runs live under TurbulenceModels/.",
            ),
            (
                "How are cases configured?",
                "Each case is a Python config module loaded through --config. The config supplies mesh paths, boundary markers, objective flags, continuation schedules, solver tolerances, output toggles, and resume behavior.",
            ),
            (
                "How are fixed regions protected in code?",
                "Configs load cell markers and build density lower/upper bound Functions. Active variables are only the free DG0 cells; fixed fluid and fixed solid cells are restored through bounds after filtering/projection.",
            ),
            (
                "What is the role of Utilities_SharedTO.py?",
                "It holds shared config loading, mesh loading, masks, checkpoint/resume helpers, VTK output resilience, pressure-drop utilities, and optimization log formatting used by both laminar and turbulent drivers.",
            ),
            (
                "What is the role of Utilities_TurbulentTO_Frozen.py?",
                "It contains SA-specific inlet nu_tilde conversion, positivity helpers, Poisson/reciprocal wall-distance solvers, and topology-penalized wall-distance update callbacks used by the frozen turbulent driver.",
            ),
            (
                "How does resume work?",
                "RESUME_OPTIMIZATION can be set in the config and overridden with FLUIDTO_RESUME or TURBULENTTO_RESUME. Checkpoints store optimization state so long runs can continue without restarting.",
            ),
            (
                "Which code was cleaned for GitLab?",
                "Unused non-frozen turbulent TO helper modules and unreferenced shared pressure-drop helpers were removed, stale commented config alternatives were deleted, and launch-script comments were normalized while preserving active flags and meshes.",
            ),
        ],
    ),
    (
        "Future Work",
        [
            (
                "What is the next technical improvement?",
                "Build an adjoint hierarchy: frozen SA as baseline, selected SA transport sensitivities as an intermediate model, and stronger closures for curvature/separation-sensitive cases.",
            ),
            (
                "What is needed for CHT?",
                "Add the energy equation, turbulent heat-flux modeling through turbulent Prandtl assumptions or richer closures, and thermal adjoint sensitivities for temperature or heat-transfer objectives.",
            ),
            (
                "What is needed for 3D?",
                "Parallel mesh generation, scalable solvers/preconditioners, wall-resolved near-wall resolution in 3D, and robust extraction/validation of three-dimensional sharp geometries.",
            ),
        ],
    ),
]


def prepare_annotation_sources() -> None:
    ANNOTATION_DIR.mkdir(exist_ok=True)
    macro_file = ANNOTATION_DIR / "defense-qa-macros.tex"
    macro_file.write_text(
        r"""
\newcommand{\DefenseAnnotated}[3]{%
  \par\noindent
  \begingroup
  \setlength{\fboxsep}{3pt}%
  \colorbox{yellow!35}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{#1}}%
  \endgroup
  \par\vspace{0.25em}
  \noindent\begingroup\small
  \textbf{Jury question:} #2\par
  \textbf{Answer:} #3
  \par\endgroup\vspace{0.70em}
}
""".lstrip(),
        encoding="utf-8",
    )

    annotations_by_file: dict[str, list[dict[str, str]]] = {}
    for item in ANNOTATIONS:
        annotations_by_file.setdefault(item["file"], []).append(item)

    for stem in INCLUDED_TEX_FILES:
        source_path = THESIS_DIR / f"{stem}.tex"
        text = source_path.read_text(encoding="utf-8")
        for item in annotations_by_file.get(stem, []):
            target = item["target"]
            replacement = (
                "\\DefenseAnnotated{"
                + target
                + "}{"
                + latex_text(item["question"])
                + "}{"
                + latex_text(item["answer"])
                + "}"
            )
            occurrences = text.count(target)
            if occurrences != 1:
                raise RuntimeError(
                    f"Expected one occurrence of annotation target in {stem}, found {occurrences}: "
                    f"{target[:100]!r}"
                )
            text = text.replace(target, replacement)
        (ANNOTATION_DIR / f"{stem}-annotations.tex").write_text(text, encoding="utf-8")

    main_text = (THESIS_DIR / "thesis.tex").read_text(encoding="utf-8")
    if r"\input{DefensePrep/defense-qa-macros.tex}" not in main_text:
        main_text = main_text.replace(
            "\\usepackage{longtable}\n",
            "\\usepackage{longtable}\n\\input{DefensePrep/defense-qa-macros.tex}\n",
            1,
        )
    for stem in INCLUDED_TEX_FILES:
        main_text = main_text.replace(
            f"\\include{{{stem}}}",
            f"\\include{{DefensePrep/{stem}-annotations}}",
        )
    (THESIS_DIR / "thesis-annotations.tex").write_text(main_text, encoding="utf-8")


def build_annotated_thesis_pdf() -> Path:
    prepare_annotation_sources()
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-jobname=Master-Thesis-Annotations",
            "thesis-annotations.tex",
        ],
        cwd=THESIS_DIR,
        check=True,
    )
    built_pdf = THESIS_DIR / "Master-Thesis-Annotations.pdf"
    final_pdf = ROOT / "Master-Thesis-Annotations.pdf"
    shutil.copy2(built_pdf, final_pdf)
    cleanup_annotation_sources()
    return final_pdf


def cleanup_annotation_sources() -> None:
    """Remove generated LaTeX source copies after a successful PDF build."""
    (THESIS_DIR / "thesis-annotations.tex").unlink(missing_ok=True)
    (ANNOTATION_DIR / "defense-qa-macros.tex").unlink(missing_ok=True)
    for stem in INCLUDED_TEX_FILES:
        (ANNOTATION_DIR / f"{stem}-annotations.tex").unlink(missing_ok=True)


def pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(200 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_qa_pdf() -> Path:
    output = ROOT / "Final-Presentation-Q&A.pdf"
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#1F2933"),
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#12355B"),
        spaceBefore=8,
        spaceAfter=8,
    )
    q_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.4,
        leading=12,
        textColor=colors.HexColor("#111827"),
    )
    a_style = ParagraphStyle(
        "Answer",
        parent=styles["Normal"],
        fontSize=8.9,
        leading=11.5,
        textColor=colors.HexColor("#273444"),
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        fontSize=9.2,
        leading=12.5,
        textColor=colors.HexColor("#334E68"),
        leftIndent=4,
    )

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
    )
    story = [
        Paragraph("Final Presentation Q&A", title),
        Paragraph(
            "Defense preparation for the master's thesis on frozen-SA turbulent topology optimization. "
            "Answers are phrased for oral delivery: direct first, then nuance.",
            subtitle,
        ),
        Table(
            [[
                Paragraph("<b>Defense stance</b>", note_style),
                Paragraph(
                    "Do not overclaim. The method is credible where the SA closure and diffuse-to-sharp transfer are validated; it is explicitly cautionary where closure sensitivity dominates.",
                    note_style,
                ),
            ]],
            colWidths=[36 * mm, 137 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF2F8")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#7BA7C7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]),
        ),
        Spacer(1, 8),
    ]

    for section_index, (section, questions) in enumerate(QA_SECTIONS):
        if section_index in {2, 4, 6}:
            story.append(PageBreak())
        story.append(Paragraph(section, section_style))
        rows = []
        for number, (question, answer) in enumerate(questions, start=1):
            rows.append([
                Paragraph(f"{number}. {question}", q_style),
                Paragraph(answer, a_style),
            ])
        table = Table(
            rows,
            colWidths=[59 * mm, 114 * mm],
            repeatRows=0,
            hAlign="LEFT",
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2EC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        )
        story.append(table)
        story.append(Spacer(1, 6))

    doc.build(story, onFirstPage=pdf_footer, onLaterPages=pdf_footer)
    return output


def main() -> None:
    annotated = build_annotated_thesis_pdf()
    qa = build_qa_pdf()
    print(annotated)
    print(qa)


if __name__ == "__main__":
    main()

# Thesis Polish Review Notes

Review date: 2026-05-23.

These notes are based on the current LaTeX sources in `Master-Thesis/` and the implementation in `TurbulenceModels/` and `FluidTO/`. I did not edit the thesis text directly.

## Highest-priority changes

1. **Make the abstract result-driven.** The current abstract is consistent with the thesis structure, but it still reads mostly like a chapter preview. Add the actual thesis outcomes: SA is useful only within a bounded curvature range, the FEniCS-SA implementation is verified against Fluent once the pending metrics are filled, the pipe-bend frozen sensitivity passes coordinate and Taylor checks, and the pipe-bend optimized geometry survives SST re-analysis with only modest loss increase. Also add the diffuser, because Chapter 6 currently evaluates it but the abstract does not mention it.

2. **Fix the adjoint terminology.** The implementation differentiates the finite-element weak residual assembled in UFL for a final frozen-viscosity steady state. It is not a tape-based `dolfin-adjoint` derivative of the entire IPCS/Picard algorithm, and it is not a full turbulence adjoint. The cleanest wording is: **a frozen-turbulence weak-form/discrete residual adjoint of the final steady flow problem**. If you keep the phrase "continuous adjoint", qualify it as continuous-adjoint-inspired, because the code differentiates the discretized FEniCS form.

3. **Align Chapter 3 equations with the actual solver.** The steady standalone solver uses IPCS forms in `ChannelSimulation_SpalartAllmaras_Steady.py` and `UBendSimulation_SpalartAllmaras_Steady.py` with non-symmetric diffusion `inner((nu + nu_t) * grad(u), grad(v))`, pressure as kinematic pressure, and a final IPCS flow solve. Chapter 4/FluidTO uses the symmetric strain tensor and dynamic viscosity. Keep those distinct.

4. **Stop treating Dean-number thresholds as universal.** The current Chapter 2 language is good in spirit but too close to implying a hard breakdown threshold. Apalowo and Abuhatira are evidence, not proof. Frame them as bracketing evidence, then let your own Ansys VMFL030/VMFL048 checks define the case-specific operating range.

5. **Reduce appendix-table listing in Chapter 6.** The introduction currently enumerates too many appendix tables. Replace that block with one sentence saying detailed parameters and mesh studies are in Appendix B/C.

6. **Replace all placeholders before final submission.** Major placeholders remain in Chapters 5 and 6 and Appendices A/B. The final conclusions cannot be made strong until those tables and figures are filled.

## Abstract

Current status: consistent with the thesis topics, except it under-represents the diffuser benchmark and does not yet state enough results.

Recommended abstract structure:

1. One sentence on the problem: turbulent internal-flow TO as a required baseline for thermal-management manifolds.
2. One sentence on method: FEniCS, density-based Brinkman penalization, SA, reciprocal wall-distance, frozen-turbulence adjoint.
3. One sentence on verification: SA curvature scope from literature + Ansys, then FEniCS-SA versus Fluent channel/U-bend.
4. Two result sentences: pipe-bend sensitivity verification, pipe-bend SST re-analysis, laminar-vs-turbulent performance; later add U-bend/diffuser results when available.
5. One cautious conclusion: this is a stable baseline, not a validated full turbulent heat-transfer optimizer.

Draft result-style ending once all data are available:

> The pipe-bend case verifies the implemented frozen derivative: five coordinate finite-difference checks agree with the adjoint derivative to a maximum relative discrepancy of \(1.47\times10^{-4}\), and a directional Taylor test recovers second-order remainder decay. For the extracted pipe-bend geometry, Ansys SST \(k-\omega\) predicts only a 4.5% higher pressure drop and an 8.5% higher viscous dissipation than Ansys-SA, while the turbulent design reduces the SST pressure drop by 72% relative to the laminar-optimized geometry. These results show that the frozen-SA framework can produce robust low-loss turbulent designs, provided that final geometries remain subject to independent CFD validation.

Do not include this exact paragraph until the U-bend/diffuser status is also integrated.

## Abbreviations And Symbols

The current list is useful, but it misses several symbols used later and has one important ambiguity.

### Abbreviation additions

Add:

- AMG: Algebraic Multigrid
- BiCGSTAB: Bi-Conjugate Gradient Stabilized method
- DOF: Degree of Freedom
- GMRES: Generalized Minimal Residual method
- HDF5: Hierarchical Data Format version 5
- ILU: Incomplete LU factorization
- MUMPS: Multifrontal Massively Parallel Sparse direct Solver
- SNES: Scalable Nonlinear Equations Solver
- SUPG: Streamline-Upwind/Petrov-Galerkin
- XFEM: Extended Finite Element Method
- 2D and 3D are optional; include them only if your faculty expects every abbreviation.

### Symbol fixes and additions

Fix the current entry `$d, y$ & Distance to the nearest wall / Hydraulic diameter`. This conflates two different meanings. Use:

- \(y\) or \(d_w\): distance to nearest wall
- \(D\): pipe diameter
- \(D_h\): hydraulic diameter
- \(R_c\): bend radius of curvature

Add the following groups.

Fluid dynamics and turbulence:

- \(D, D_h, H, L\): pipe diameter, hydraulic diameter, channel height, length
- \(U_{\mathrm{ref}}, U_{\mathrm{bulk}}, U_{\max}\): reference, bulk, and maximum velocity
- \(u^+\): non-dimensional mean velocity
- \(\tau_w\): wall shear stress
- \(C_f\): skin-friction coefficient
- \(\delta_{ij}\): Kronecker delta
- \(\omega\): scalar vorticity in 2D, if used separately from \(\Omega_{ij}\)
- \(c_{b1}, c_{b2}, c_{v1}, c_{w1}, c_{w2}, c_{w3}\): SA constants
- \(f_{v1}, f_{v2}, f_w\): SA damping functions
- \(r_{\mathrm{SA}}, g_{\mathrm{SA}}\): SA wall-destruction auxiliary variables; avoid using plain \(r\), because \(r\) is already the filter radius

Numerical and verification:

- \(p_k\): kinematic pressure
- \(\Gamma_{\mathrm{in}}, \Gamma_{\mathrm{out}}, \Gamma_D\): inlet, outlet, and profile extraction boundaries
- \(s\): arclength coordinate along a profile line
- \(e_i\): pointwise velocity-profile error
- \(E_\infty, E_{L_2}\): maximum and relative \(L_2\) profile discrepancies
- \(\Delta h\): finite-difference perturbation
- \(S_i^{\mathrm{FD}}, S_i^{\mathrm{fr}}\): finite-difference and frozen-adjoint sensitivities
- \(\varepsilon_i^{\mathrm{fr}}\): relative frozen-adjoint sensitivity discrepancy

Topology optimization:

- \(\Pi_\beta\): Heaviside projection operator
- \(M\): active design mask
- \(\mathcal{H}\): Helmholtz filter operator; avoid \(H\) because channel height also uses \(H\)
- \(\Omega_D, \Omega_V\): objective and volume-integration regions
- \(V_{\mathrm{mask}}\): volume-mask measure
- \(v_f^{(k)}\): realized volume fraction at MMA iteration \(k\)
- \(r_V^{(k)}\): volume residual history
- \(\lambda\) or \(\mathbf{s}^*\): adjoint state; choose one notation and use it consistently
- \(\Phi_D\): sharp-geometry viscous dissipation metric

## Chapter 1: Introduction

The motivation is relevant and mostly convincing, but it should state the central gap earlier: **laminar fluid TO is mature, but industrial cooling manifolds often operate in turbulent regimes where RANS model choice and wall treatment affect the topology itself**. That is the strongest motivation for the thesis.

The density-based TO overview needs one correction. The figures in Figure 1.1 appear to show the **projected/physical density field** \(\bar{\gamma}\), not the raw optimizer variable \(\gamma\). Mention this lightly in Chapter 1 and leave the full density chain for Chapter 4:

> For the conceptual figures in this chapter, the displayed field represents the physical, projected density used by the flow equations. Chapter 4 distinguishes this field from the raw optimizer variable and the filtered density.

Also remove or soften the statement that a uniform initial \(\gamma=0.5\) is standard. Your benchmark initial densities match the volume fraction targets: 0.25, 0.27, and 0.30.

The current research questions are good, but they would be stronger as a short roadmap. Keep the questions in bold, then add 2-3 explanatory sentences under each:

- RQ1: answered by turbulence theory, Dean-number literature, and Ansys curvature benchmarks.
- RQ2: answered by Chapter 3 implementation plus Chapter 5 Fluent verification.
- RQ3: answered by sensitivity verification, FEniCS optimization, sharp-geometry Fluent SA/SST validation, and laminar-vs-turbulent comparisons.

Important wording issue: Chapter 1 currently mentions `dolfin-adjoint`. The code does not use `dolfin-adjoint` as the main derivative engine. Replace that sentence with wording about FEniCS/UFL enabling explicit weak-form residual differentiation.

## Chapter 2: Turbulence Modeling

Physics coverage is mostly sufficient for a master's thesis. Add only compact explanations, not another full turbulence textbook chapter:

- Add one small figure for Reynolds decomposition/RANS averaging.
- Add one figure for near-wall layers: viscous sublayer, buffer layer, log layer, \(y^+\approx1\) wall-resolved mesh, \(y^+>30\) wall-function mesh.
- Add 1-2 sentences on why eddy-viscosity models struggle with curvature: secondary motion changes Reynolds-stress anisotropy, while Boussinesq closures map all turbulence effects into one scalar \(\mu_t\).
- Consider citing Rumsey et al. for curvature-effect sensitivity; those entries are already in your `.bib`.

Dean-number discussion: keep it, but make it more cautious. The Dean number is academically relevant because it combines Reynolds number and curvature ratio, but the current conclusion should not read as "SA is valid below \(De=15000\) and invalid above \(De=18000\)." Instead say the cases **bracket a case-specific range** for your geometry and model setup.

Apalowo and Abuhatira: they are not proof of your findings. They are supporting evidence that SA performs acceptably for mild curvature and poorly for stronger curvature. Your own Fluent comparisons are the thesis-specific evidence. Phrase it that way.

SA equations: yes, Chapter 2 should show the transport equation and physical interpretation only. Put constants, damping functions, clipping, and weak forms in Chapter 3, because Chapter 2 treats SA as a model choice and Fluent black box.

Chapter 2 conclusion: make it explicitly answer RQ1:

- SA is suitable for wall-resolved internal turbulent TO because it is one-equation, robust, and avoids wall functions.
- Its limitation is not just "high Reynolds number"; it is curvature/separation-driven anisotropy under a scalar eddy-viscosity closure.
- Therefore, final topologies must be validated with an independent turbulence closure.

## Chapter 3: FEniCS SA Implementation

The FEniCS introduction is mostly good. Strengthen it by making the reason for FEniCS specific: FEniCS lets you write weak forms directly, use mixed finite-element spaces, and differentiate residual forms in UFL. Remove any implication that Fluent-like black-box CFD is being replicated feature-for-feature.

Figures would help. Add:

- A solver-coupling diagram: wall distance -> flow IPCS -> SA update -> Picard residual -> final flow.
- A benchmark mesh/wall-distance figure if available: contour of \(y\) or \(G\) for channel/U-bend.
- Optional: a small schematic of IPCS steps.

Code listings: keep them short and pseudo-code-like. They add value if they expose the exact algorithmic path. Do not paste long implementation. The current SA listing is useful. The Eikonal listing is less valuable because it repeats the equations; replace it with a smaller "weak residual + recovery" snippet or move it to an appendix.

Most important implementation mismatch:

- Standalone Chapter 3 solver uses the weak forms in `TurbulenceModels/ChannelSimulation_SpalartAllmaras_Steady.py` and `TurbulenceModels/UBendSimulation_SpalartAllmaras_Steady.py`: non-symmetric diffusion \( (\nu+\nu_t)\nabla u \), kinematic pressure, IPCS pseudo-time stepping.
- FluidTO uses a final monolithic frozen-viscosity residual with symmetric strain tensor and dynamic viscosity. That belongs in Chapter 4, not Chapter 3.

Current wall-distance equation matches `TurbulenceModels/Utilities.py`: the implemented residual is

\[
(1-\sigma_w)\int_\Omega |\nabla G|^2 z\,dx
-\sigma_w\int_\Omega G\nabla G\cdot\nabla z\,dx
-(1+2\sigma_w)\int_\Omega G^4 z\,dx=0,
\]

with \(G=G_0\) on physical walls and \(y=\max(1/(G+G_{\min})-1/G_0,0)\). Add the \(G_{\min}\) floor and positive-part recovery to the text if you claim exact implementation agreement.

Add a global benchmark solver listing. This would be more valuable than the current Eikonal listing:

1. initialize \(u,p,\tilde{\nu}\) and wall distance \(y\)
2. build IPCS forms with fixed \(\nu_t(\tilde{\nu})\)
3. solve flow to steady state with IPCS substeps
4. solve steady SA transport at fixed velocity
5. under-relax \(\tilde{\nu}\), apply floors/boundary values
6. repeat Picard loop until flow/turbulence residuals or diagnostics converge
7. run final flow solve and export \(u,p,\tilde{\nu}\)

## Chapter 4: Turbulent TO Methodology

Add a RAMP figure. The formula is clear mathematically, but a plot of \(\alpha(\bar{\gamma})\) for several \(q\) values will make the continuation strategy much clearer. Generate the plot yourself from your formula and cite Stolpe and Svanberg.

Non-design domains deserve a short subsection and figure. The current paragraph is dense. A figure showing active design, passive fluid inlet/outlet buffers, passive solid walls/separator, objective mask, and volume mask would immediately clarify the method.

Sensitivity analysis needs a little more math, but not a full Dilgen derivation. Add a compact Lagrangian block:

\[
R(\mathbf{s},\bar{\gamma};\nu_t^{fr},G^{fr})=0,\qquad
\mathcal{L}=J(\mathbf{s},\bar{\gamma})+\mathbf{s}^*R(\mathbf{s},\bar{\gamma};\nu_t^{fr},G^{fr}),
\]

\[
\left(\frac{\partial R}{\partial \mathbf{s}}\right)^T\mathbf{s}^*
=-\left(\frac{\partial J}{\partial \mathbf{s}}\right)^T,\qquad
\frac{dJ}{d\bar{\gamma}}
=
\frac{\partial J}{\partial\bar{\gamma}}
+\mathbf{s}^*\frac{\partial R}{\partial\bar{\gamma}}.
\]

Then explain that the gradient is chained through projection and the transpose filter before MMA. This is enough for rigor without making the thesis too mathematical.

Frozen turbulence needs the strongest explanation in the thesis. The current text is good but should be more explicit:

- A full adjoint would require adjoint equations for SA, \(\nu_t(\tilde{\nu})\), reciprocal wall distance \(G\), topology-dependent SA and wall-distance penalties, and any clipping/flooring used for robustness.
- That full derivative is more accurate in principle but much more fragile for evolving-wall topology optimization.
- Frozen turbulence keeps \(\nu_t\) and \(G\) updated in the forward loop, so the flow physics still see turbulence. The approximation is only in the sensitivity calculation.
- This is common in industrial adjoint workflows because it gives stable gradients at practical cost.
- Your finite-difference checks validate only this frozen derivative, not the full coupled derivative.

Use this distinction repeatedly: **forward physics are not frozen; only the derivative of the turbulence fields is frozen.**

Heaviside projection: add a figure of \(\Pi_\beta(\tilde{\gamma})\) for increasing \(\beta\) at \(\eta=0.5\), and explain that \(\eta\) is the density threshold around which the projection sharpens.

Section 4.8 is currently too text-heavy and partly duplicates boundary-condition material from Section 4.4. Split it:

- Algorithm overview
- Forward turbulent solve
- Frozen adjoint and sensitivity chain
- MMA and continuation

The pseudo-code is close and now includes MMA. Add the volume constraint gradient explicitly:

```
dV_dgamma_effective = assemble_volume_sensitivity()
dV_dgamma_raw = transpose_filter(dV_dgamma_effective)
gamma_raw = MMA.update(gamma_raw, J, dJ_dgamma_raw, g, dV_dgamma_raw)
```

MMA should not remain a black box. One paragraph is enough: MMA builds a local convex approximation using objective/constraint values and gradients, uses moving asymptotes and move limits to prevent unstable density jumps, and updates only active raw design DOFs while passive cells are restored.

## Chapter 5: Verification of the SA Solver

The chapter is structurally strong. It needs the placeholders filled and a slightly less appendix-heavy opening.

Main chapter should include:

- simple geometry drawing for channel
- simple geometry drawing for U-bend
- governing operating point: \(Re\), \(De\) if relevant, fluid properties, inlet/pressure forcing
- mesh requirement: wall-resolved \(y^+\approx1\), selected mesh only
- verification metrics and final quantitative errors

Appendix should keep:

- full solver tolerances
- all mesh hierarchy tables
- full grid-independence details
- exact Fluent setup tables

The channel and U-bend cases need to be clear before results. Generated schematics are acceptable and probably better than hunting for exact paper figures, because these are your exact shortened/periodic configurations.

Conclusion must answer RQ2 quantitatively after data are filled:

- channel: report \(U_{\mathrm{bulk}}\), \(C_f\), \(E_\infty\), \(E_{L_2}\)
- U-bend: report \(\Delta p\), \(E_\infty\), \(E_{L_2}\)
- state what the agreement is sufficient for: use as forward model in TO, not experimental validation of turbulence physics

## Chapter 6: Turbulent TO Results

The Chapter 6 intro currently lists too many appendix figures/tables. Replace with:

> Detailed physical parameters, solver settings, and Ansys grid-independence studies are collected in Appendices B and C; the main text reports only the setup choices needed to interpret the optimization and validation results.

Benchmark introductions are generally good. Keep the geometry/BC figures in the main text, because the reader needs them before seeing topologies.

Sensitivity verification for the pipe bend is correct **for the objective you optimized**, \(J_p\). You do not need to verify \(J_D\) unless you switch the pipe-bend objective to \(J_D\). The current method is defensible because it uses central finite differences, holds \(\tilde{\nu}\) and \(G\) fixed consistently with the frozen adjoint, re-solves the velocity-pressure state, and includes a directional Taylor check.

To make the sensitivity verification closer to literature standards, add or mention:

- at least one step-size sweep or explain why \(\Delta h=10^{-6}\) was chosen
- one check at a later/design-like state if feasible, not only the initial density
- solver tolerances tighter than the finite-difference objective changes
- a reminder that this verifies the frozen derivative only

For U-bend and diffuser convergence placeholders, use one double-axis placeholder/figure each, like the pipe bend, instead of two subfigures.

The Chapter 6 conclusion is currently too short and too generic. It should answer RQ3 directly:

- state whether SA optimized designs remain credible under Ansys-SA
- state whether SST \(k-\omega\) changes performance modestly or significantly
- state whether turbulent designs outperform laminar designs under turbulent re-analysis
- state which cases are fully demonstrated and which remain pending

For the completed pipe bend, you already have strong numbers: SST predicts 4.5% higher \(\Delta p\) and 8.5% higher \(\Phi_D\) than Ansys-SA, and the turbulent design gives 72.0% lower pressure drop than the laminar design under SST. Use these in the conclusion.

## Chapter 7: Conclusion And Future Work

Yes: repeat the three research questions in bold and answer each with a complete paragraph. This will make the conclusion much stronger and easier for examiners to map onto Chapter 1.

Suggested structure:

1. Short opening summary of the thesis.
2. **RQ1** exact text from Chapter 1 and answer with Chapter 2 findings.
3. **RQ2** exact text and answer with Chapter 5 quantitative verification.
4. **RQ3** exact text and answer with Chapter 6 quantitative validation.
5. Limitations.
6. Future work.

Limitations are good. Add one explicit limitation if it is not already clear enough: **the frozen adjoint is verified only against finite differences under the same frozen assumption, not against a full turbulence-adjoint derivative.**

Future-work references:

- 3D turbulent TO: cite Dilgen et al. because they demonstrate 2D and 3D turbulent examples.
- Large-scale 3D computational cost: optional citation to Aage et al. 2017 for giga-scale 3D TO, even though it is structural rather than turbulent flow.
- Adjoint hierarchy: cite Zymaris et al. 2009 and Margetis et al. 2024 for SA adjoint work, plus Wu et al. 2023 for CEV effects.
- Turbulent CHT: Maute 2023, Zhao 2021, Sun 2023 are already good.

## Figures To Add

Best generated figures:

- RANS/Reynolds decomposition schematic: own figure, cite Pope/Versteeg.
- Wall-layer figure with \(y^+\): own figure, cite Pope/White and optionally Spalding 1961.
- RAMP curve: own plot from Eq. 4.1, cite Stolpe and Svanberg.
- Heaviside projection curve: own plot from Eq. 4.??, cite Guest and Wang.
- Filter/domain figure: own schematic of active/passive masks, cite Lazarov and Bourdin.
- Algorithm workflow: own diagram; no external citation needed.
- Channel/U-bend benchmark geometry sketches: own schematics; cite your benchmark sources or Ansys manual as appropriate.

Avoid reprinting paper figures unless they add more than a clean self-generated schematic. Generated formula plots are easier to read and avoid permissions concerns.

## References Count

About 50 references is reasonable for a master's thesis in this area. Your current `references.bib` has roughly 60 entries, which is also fine. Quality and traceability matter more than the exact count: keep references that support methods, validation data, benchmark definitions, and future-work claims; remove unused or weak sources if the final bibliography feels padded.


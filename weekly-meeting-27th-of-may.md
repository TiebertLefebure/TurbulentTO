Weekly meeting - 27/05/2026

Slide 2:
Changing the thesis title is a possibility, not a necessity 
Heat transfer (energy equation) is not really included in this thesis TO framework
Title now: “Topology optimization of heat transfer applications operating in the turbulent regime”
New title example: “Topology optimization of flow configurations operating in the turbulent regime”

Slide 3:If research questions can be answered in the text, then these RQ’s are okay
RQ’s need an answer, delete them otherwise

Slide 4:Switch around chapter 4 (turbulent TO methodology) and chapter 5 (verification of SA solver) so the reader doesn’t first read about TO, then has an interruption with SA verification results, and then reads the TO results
Methodologies and results should follow up directly 
Slide 5:
Still talk about how you’re interested in energy applications in the introduction, the general field of study is about heat sinks/ heat exchanger/ …
But the thesis is focused on flow-only, TO framework is for flow-only applications
In the introduction, start broad and narrow it down to flow-only
That’s why the title will be changed to “flow configurations” instead of “heat transfer applications”
Explain the title in the introduction
Flow configurations are a stepping stone towards energy applications (with heat transfer)
Heat transfer devices require designs where pressure drop is minimised , even before considering heat transfer (energy equation) in the calculations
Manifolds are used in heat transfer applications, they need minimum pressure drop / dissipation
Manifolds don’t transfer a lot of heat: pressure drop is very important then (& dissipation) 
Flow junctions, flow collectors (they are flow-only) are also used in energy applications
Minmizing pressure drop is mandatory for these type of applications
Focus on flow-only inside general field of study of energy applications
Background and motivation should explain really well the scope of the thesis
Motivation can still talk about heat transfer applications, but make the transition towards flow-only (our thesis scope)
Flow-only devices and optimization is also already a challenge in itself, so we’re only tackling that
Throughout the motivation, narrow the scope to flow-only (narrow scope toward research questions)

Slide 6: 
E.g. a 60 page thesis should not spend 30 pages on turbulence modeling and physics
Length of chapter 2 currently is okay

Slide 7: 
Code snippets (in chapter 3 and 4) not necessary, code in appendix or uploaded separately 
Present PDE’s and their weak formulations (don’t write derivations if you didn’t make them yourself)
Focus on work you’ve done yourself; and work that answers the research questions
All aspects needed to answer RQ’s should be included in the thesis, be completeE.g. equation 3.14 gives the weak formulation of the relaxed Eikonal equation, that helps.
 
Slide 9-11: Pipe Bend benchmarkIs the develop inlet length long enough for the Re=5000 in the benchmark?
Alexandersen paper entrance lengths might be too short, they seem rather random
Turbulent fow must be able to develop through the passive fluid inlet extension
Uniform velocity BC at physical inlet
There exist analytical expressions to obtain minimal required inlet length for turbulent flows to be developed
Check if entrance lengths are long enough
Oscillations in convergence history plot for J/J0
Is design visibly changing and oscillating in ParaView?
Use smaller steps in MMA for pipe bend, too little damping in the optimization currently
J_p used (pipe bend & U-bend) vs. J_D used (diffuser): pressure drop is proportional to dissipation
Minimize pressure drop = minimize energy dissipation
Use J_D for all cases, will probably be more stable (less oscillations)
With J_p: minimising pressure in only 1 boundary (physical inlet)
Adjoints extremely sensitive, small gradients
MMA determines change in J_p by change in all cells of the domain
Some cells are really far away from that physical inlet
With J_D, you take integral over whole domain, you make sum of dissipation in each design cell
Aggregated energy dissipation is the better choice
Run pipe bend with J_D, J_D is the correct model
Using pressure drop J_p is a simplification, because they’re proportional
Pressure drop proportional to energy dissipation, it’s a shortcut to use J_pPlot J, not J/J0 for convergence historyJ/J0 is only useful when scales are far away from eachother (e.g. J0 in order of 1e+03, and final J is 1e-01)But here J0≈2 Pa and final J = 0.2 Pa
Be sure to optimise with J_p as objective function, not with J/J0 as objective function
Make sure to divide by the length of the inlet (gamma_in) when using the J_p objective (check formula 4.13 in text)
FEniCS-SA dissipation (result of optimization) compares well to the Ansys-validated SA value (validating turbulent design with Re=5000 flow) -> 0.0147 W/m final phi_D in FENiCS vs. 0.0153 W/m in AnsysBut the pressure drop value is less close to the Ansys-validated value  (validating turbulent design with Re=5000 flow) -> 0.2032 Pa in FENiCS) vs. 0.1457 Pa in AnsysThis is weird because dissipation is proportional to pressure drop, they should match evenly wellMaybe this is because of oscillations of both phi_D and delta p

Slide 12: U-Bend benchmark
U-Bend is the most interesting case!
Highest priority: try to run turbulent-TO U-bend at Re=5000
Run U-bend with J_D objective as well
If not possible to run at Re=5000, look for other asymmetrical topologies -> e.g. U-bend with Re=200
For Re=1, the laminar design is perfectly symmetric (creeping flow)
Ideally get a high-Re result
U-bend is the more challenging case
Asymmetrical topology starts to get visible even at low-Re case (e.g. Re=200)
You can really exploit turbulence in this benchmark case


Slide 13-15: Diffuser benchmark
Flow separation in diffuser is clearly visible (low velocity near outlet, where channel is diverging)
Blue in velocity plot = low velocity = separation zone
Zoom-in in Ansys to check if you can see separation zone (recirculation)Now there’s a Dirichlet velocity BC at outlet, you force outlet to be that wide (L/3), maybe that’s why channel diverges near the end when it really doesn’t want to (leads to dissipation)
With flow separation, SA model fails, Ansys validated SST k-omega values (dissipation, pressure drop) much lower than SA values when simulating turbulent flow through turbulent design
Where does this difference in values come from? 
For turbulent design (running turbulent flow at Re=3000 through this design in Ansys): SA gives 1.3E+05 W/m & 1.2E+05 Pa for dissipation & pressure drop, while SST k-omega gives 1.6E+04 W/m & 4.1E+04 Pa.For laminar design (running Re=3000 flow through this design in Ansys): SA gives 1.4E+05 W/m & 1.4E+05 Pa, while SST k-omega gives 1.3E+04 W/m & 3.6E+04 Pa.Figure 6.15 shows that both laminar & turbulent designs have flow separation.
You can prove your point from chapter 2 about validity of the SA model
SA fails when too much flow separation occurs
Try to really explain the results and be critical
You have to be able to defend yourself on the presentation
Don’t just say “we see this”, why do we see it?
FEniCS uses a SA-solver optimiser, can fail when flow separation starts playing a role
<-> pipe bend has a nice flow in the bend (Figure 6.5 shows the flow in the turbulent design, nicely flowing everywhere in the bend), you can see this in the Ansys validation velocity plots, there’s no flow separation there
<-> for pipe bend: turbulent design shows that SST k-omega values (dissipation, pressure drop) in Ansys are close to SA values (SA gives 0.0153 W/m & 0.1457 Pa for turbulent design in Ansys, while SST gives 0.0166 W/m & 0.1522 Pa) 
<-> for pipe bend: turbulent design performs better than laminar design when simulating turbulent flow, both with SA and k-omega in Ansys (lower dissipation and pressure drop for turbulent design)

** Prof. Blommaert **
You can even check wether SA fails for flow separation with e.g. a backwards facing step simulation (can be FEniCS and/or Ansys)
Which metrics do you use then?
Suggested answer:
Use Ansys as the main evidence if time is limited, because the same backwards-facing-step case can be compared directly with SA and SST k-omega. Use FEniCS-SA only as a supporting check of the in-house solver. The main metric should be the reattachment length x_r/h, obtained from the lower-wall shear stress or skin-friction coefficient changing sign after the step. Add recirculation-zone size or reverse-flow length, pressure recovery/drop, and local eddy-viscosity or dissipation-density plots if the goal is to connect the result to the diffuser discrepancy.
Placement: Chapter 2 should contain the short physical argument that SA is known to be weaker in separated adverse-pressure-gradient flow. A computed backwards-facing-step result belongs with solver/model verification, so Chapter 5 or Appendix A, not Chapter 4. Chapter 6 can then refer back to it when explaining the diffuser.
Steady run: yes. A steady RANS backwards-facing-step calculation is defensible because it represents the mean flow. The current FEniCS file is `TurbulenceModels/BackStepSimulation_SpalartAllmaras_Steady.py`, with config `TurbulenceModels/Configs/ConfigBackStep_SpalartAllmaras_Steady.py`. It uses the shared steady SA IPCS/Picard driver and logs reattachment length, reverse-flow size, pressure drop, eddy-viscosity levels, and dissipation-proxy metrics.

For diffuser: turbulent design performs better than laminar when simulating turbulent flow (Re=3000) in Ansys with SA, but turbulent design performs worse when simulating turbulent flow with SST k-omega
Try pressure BC in outlet (diffuser) instead of velocity outlet
Are separation zones (diverging channel) still there after optimization with this new BC?Run for +/- 30 MMA iterations
Check evolution of topology

Possibilities for diffuser
1)) Trim diverging channel part near outlet in Ansys DesignModeler and run again with turbulent flow conditions (Re=3000) with both models
Make the diffuser into just a straight channel near the outlet (horizontal)Is the main contributors of problem (problem = way higher dissipation for SA) located in the diverging channel part?
If dissipation is much lower for SA when you cut off diverging part, then you know the problem is located there, and you can write conclusions about flow separation
Run again with SA and SST k-omega turbulent flow -> are the differences in delta p and phi_D still that large between SA & SST k-omega? Can we “blame” the differences on the diverging part of the diffuser?
2)) plot objective function space (J_D as objective function) in ParaView
Not just the integral (sum) over all cells, but the J_D value per cell
Make .pvd file to plot in ParaView
Do the corners (diverging channel part) indeed have high dissipation? 
Try to prove that you’re over-predicting energy dissipation in the corners with SA model
Ansys: energy dissipation metric (custom field function with strain rate & effective viscosity), plot that metric, check corners
For both turbulence models SA & SST k-omega, compare how they predict turbulence

Under- or overpredicting turbulence/dissipation in what regions?
Plot turbulence related variables, not only projected density / velocity / pressure
Plot turbulence variables, because you’re trying to predict those values with RANS models
Right or wrong RANS models (SA vs. SST k-omega): plot values to verify
Only difference between RANS models is inside the turbulence related variables (turbulent intensity, turbulent viscosity)
Navier-Stokes equations are the same for all RANS models, so velocity and pressure are modelled the same
Laminar & turbulent part of dissipation (effective viscosity = molecular viscosity + turbulent viscosity)
Dissipation density = effective viscosity * strain rate * strain rate
Segregate the laminar & turbulent part of dissipation
Laminar part is modelled the same way for both RANS models (just a material property)
Turbulent part is predicted differently by SA & SST k-omega models, so plot for both models and show the difference
Main goal: explain results, why does SA fail? where does failure come from (which location in the diffuser)? how is turbulence predicted there by both RANS models? who is overpredicting the dissipation ?

LIST OF THINGS WITH HIGHEST PRIORITY:

1)) U-bend computation, ideally with Re=5000 
U-bend is most interesting case, focus computational time on this

2)) Fix oscillations in objective for the pipe bend
If you can’t fix oscillations: run without continuation stages for longer inner iterations (continuation is not changing the value of J_p, it’s just oscillating around 1 average value, so just run at the q/beta/move limit where the objective is low already and started oscillating), but run for many MMA iterations (e.g. 1000)
If you can’t fix oscillations and run with 1 continuation stage: mention more strict continuation for future cases (“Future work”), more robust analysis of material model (if oscillations don’t go away)
Run with J_D, full energy dissipation objective
Compare Alexandersen paper pipe bend topology vs. FEniCS-SA pipe bend, do they match?
Lay topologies on top of each other, they should match
If they don’t overlap -> mention future robust analysis
Always validate with Ansys, strongest comparison all-round (FEniCS-SA values from optimization vs. Ansys SA & SST k-omega values from CFD validation)
3)) Figures in thesis text
No scientific notation (xxE+yy) when not needed, keep it clean
Larger font size for pictures, it has to be readable when printed out
Excel figures, ParaView, Ansys: large enough colour bar, numbers have to be readable when printed out, so font 11-12-13If you can’t properly show your results, it was all for nothing -> BE CAREFULSo make all axes, color bars, numbers clearly readableTry to maintain the same size of Excel plots across the entire thesis
Same for ParaView & Ansys plots: same size
Make Ansys plots white in the background! how do you change this in Ansys Fluent?

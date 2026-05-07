# Dilgen Pipe Bend SA Precursor Profiles

Dilgen et al. (2018), Sec. 5.1, prescribe fully developed turbulent inlet
profiles for the 90 degree channel bend. The paper gives the operating point
but does not tabulate the profiles:

- half inlet height `H = 0.1 m`
- bulk velocity `Ub = 5 m/s`
- kinematic viscosity `nu = 5e-5 m^2/s`
- `Re_H = Ub H / nu = 10000`

Generate the repository-local Spalart-Allmaras inlet profiles with:

```bash
python3 PrecursorProfiles/DilgenPipeBendSA/generate_dilgen_sa_channel_profiles.py
```

The script writes:

- `dilgen_sa_channel_profile.csv`: local wall-to-wall coordinate `y`,
  streamwise velocity `u_x`, scalar `nu_tilde`, and diagnostics.
- `dilgen_sa_channel_profile.json`: operating point and convergence metadata.

The Dilgen config reads this CSV and maps local `y = 0..0.2 m` onto the inlet
segment `x = 0, y = 0.7..0.9`.

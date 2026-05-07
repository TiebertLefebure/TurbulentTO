# Dilgen U-bend SA Precursor Profile

Dilgen et al. (2018), Sec. 6.1, prescribe fully developed turbulent channel
inlet profiles for the 2D U-bend case. The generated CSV in this directory uses
the repository's 1D Spalart-Allmaras channel precursor with:

- bulk velocity `Ub = 2 m/s`
- half channel height `H = 0.1 m`
- kinematic viscosity `nu = 4e-5 m^2/s`
- `Re_H = Ub H / nu = 5000`

Regenerate with:

```bash
python3 PrecursorProfiles/DilgenPipeBendSA/generate_dilgen_sa_channel_profiles.py \
  --bulk-velocity 2.0 \
  --half-height 0.1 \
  --kinematic-viscosity 4.0e-5 \
  --output PrecursorProfiles/DilgenUBendSA/dilgen_sa_channel_profile.csv
```

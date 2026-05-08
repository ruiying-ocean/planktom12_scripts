# Namelist Notes

Operational notes on NEMO/PlankTOM namelist blocks — focused on non-obvious
behaviour, decoupling choices, and known gotchas. Grow this file as more
namelist questions come up.

---

## `&namtra_qsr` — penetrative solar radiation

NEMO requires **exactly one** light-penetration scheme active when
`ln_traqsr=.true.` The consistency check at `traqsr.F90:477` calls `ctl_stop`
if zero or more than one scheme flags are set.

### The four schemes

| switch        | scheme                                 | uses inline biology? |
|---------------|----------------------------------------|----------------------|
| `ln_qsr_rgb`  | 3-band RGB (Morel), Chl-dependent      | no — Chl from file or constant |
| `ln_qsr_2bd`  | 2-band Paulson–Simpson                 | no |
| `ln_qsr_bio`  | bio-model `etot3` (PISCES/TOP)         | **yes** |
| `ln_qsr_sms`  | PlankTOM `etot3`                       | **yes** |

`ln_qsr_bio` and `ln_qsr_sms` route the live biology's optical attenuation
back into the temperature tendency — that's the path that couples physics
to inline biology. Keep both `.false.` to break the coupling.

### Recommended: physics decoupled from inline biology

Use **RGB + SeaWiFS Chl climatology**. The file
`chlorophyll.nc` is auto-staged by `setUpRun.sh` from
`/gpfs/data/greenocean/software/resources/ModelResources/Chlorophyll/chlorophyll.nc`
(see `configs/setUpData_*.dat`).

```fortran
&namtra_qsr
   sn_chl      ='chlorophyll', -1, 'CHLA', .true., .true., 'yearly', '', '', ''
   cn_dir      = './'
   ln_traqsr   = .true.
   ln_qsr_rgb  = .true.
   ln_qsr_2bd  = .false.
   ln_qsr_bio  = .false.
   ln_qsr_sms  = .false.
   nn_chldta   = 1          ! 1 = read file, 0 = constant 0.05 mg/m³
   rn_abs      = 0.58
   rn_si0      = 0.35
   rn_si1      = 24.4
   ln_qsr_ice  = .true.
/
```

### `chlorophyll.nc` spec

- Shape `(x=182, y=149, time=12)` on the ORCA2 grid — 2D surface only
- Variable `CHLA`, units mg Chl m⁻³
- Provenance: SeaWiFS climatology 1999–2005 (`chla.seawifs.clim9905.third`)
- NEMO reads the surface slab only at `traqsr.F90:247` and clamps to
  `[0.03, 10]` for the Morel lookup; deeper levels would be ignored.
- Land `_FillValue=-1e+34` is benign — masked out downstream by `tmask`.

### Alternative schemes

- **`ln_qsr_2bd=.true.`** — 2-band Paulson–Simpson, no Chl, no file. Same
  decoupling outcome, slightly less realistic optics. Use this if
  `chlorophyll.nc` is unavailable.
- **`ln_qsr_rgb=.true.` with `nn_chldta=0`** — RGB but with constant Chl=0.05
  mg/m³ baked in (no spatial structure).

### Verifying the active scheme at runtime

After model start, `ocean.output` echoes the chosen scheme. Expected output
when configured as above:

```
R-G-B   light penetration - Chl data
trc_oce_rgb : Initialisation of the optical look-up table
```

Quick checks:

```tcsh
# Show the relevant switches in a namelist
grep -E "ln_traqsr|ln_qsr_(rgb|2bd|bio|sms)|nn_chldta" namelist_ref_*

# Confirm runtime resolution
grep -A 2 "light penetration" ocean.output
```

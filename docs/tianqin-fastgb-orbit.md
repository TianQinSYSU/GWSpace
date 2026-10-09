# TianQin native FastGB orbit consistency

Upstream `82fac16d31ac6d42eb692739f19a68ad763898b3` still uses
`Omega_tq = 1.9923849908611068e-5 rad/s` in `include/spacecrafts.h`.
This is a 3.65-day orbit, despite the accompanying Kepler-law comment.
Python `TianQinOrbit.f_0` instead uses
`sqrt(G_SI * EarthMass / R**3) / (2*pi)`.

The native call chain is `GCBWaveform.get_fastgb_fd_single` →
`libFastGB.ComputeXYZ_FD` → `Fast_GB` in `src/GB.c` →
`spacecraft_TianQin` in `src/spacecrafts.c`. The same orbit is exposed by
`libFastGB.Orbits`/`get_pos`. The TD chain is `get_AET_td` → `get_XYZ_td` →
`get_y_slr_td` → `detectors['TQ']`/`TianQinOrbit` → `alpha_detector`/`f_0`.

The fix evaluates the native angular frequency with Kepler's law. Both
implementations use the physical inputs from `include/constants.h`:
`G_SI = 6.67430e-11`, `EarthMass = 5.9722e24 kg`, and the new shared
`TianQinOrbitRadius_SI = 1e8 m`. The existing build translates that header
into `gwspace/constants.py`. Python derives its arm length from the shared
radius, retaining the existing `f_0` formula and subclass behavior. The C
radius aliases the shared radius; no independently rounded angular-frequency
constant remains. Other response, FFT, normalization and buffer-size
conventions are unchanged.

| Quantity | Before (native C) | After (C and Python) |
| --- | ---: | ---: |
| Orbital frequency (Hz) | 3.1709791983764586e-6 | 3.17753369855143e-6 |
| Angular frequency (rad/s) | 1.9923849908611068e-5 | 1.9965033047806353e-5 |
| Period (86400-second days) | 3.65 | 3.6424709136367137 |

Previously the second-harmonic offset accumulated 1.0335135876 Fourier bins
in 78,840,000 seconds (2.5 × 365 days).

## Regression evidence

`tests/test_tianqin_fastgb_orbit.py` checks quarter-period rotation and
full-period closure through the compiled orbit, plus all three spacecraft
at 1001 times across 2.5 years. After the fix, the maximum geocentric
position discrepancy is 1.71e-13 light seconds. SSB positions differ by at
most 4.84e-9 light seconds because of the existing independent rounding of
Earth's orbital frequency; that separate convention is unchanged.

The response test uses a single zero-planet binary (6.22 mHz, supplied
`fdot=7.484049960353154e-16 Hz/s`, `fddot=0`) over 2.5 years. It compares native
FastGB at oversample=16 against actual first-generation real TD TDI → FFT
at 60-second cadence, on the same integer bins within f0 ±10 microHz.
It converts both to physical Fourier units using the existing `dt` factors.
There is no fitted phase, frequency shift, interpolation or calibration.

| Channel / metric | Before | After |
| --- | ---: | ---: |
| A raw unweighted overlap | 0.03183865152 | 0.9999994400 |
| E raw unweighted overlap | 0.03184179687 | 0.9999993694 |
| A complex relative L2 | 1.3929411 | 0.0023199523 |
| E complex relative L2 | 1.3929408 | 0.0023507552 |

The original upstream angle was also rebuilt in this checkout: all three
new regressions fail before the fix. After restoring the fix, all eight
repository tests pass in the GW Python 3.9 / NumPy 1.26.4 environment.
An independently built and installed wheel also passes all eight tests.
Calculations use one thread on a 20-core host, leaving at least two cores
free. No simulation HDF5 files are regenerated or modified.

The residual ~0.23% is allowed: native frequency-domain TDI acts on the
modulated links, while TD currently freezes geometry at output times.
Finite-record slow-envelope sampling/leakage and chirp amplitude/transfer
approximations also remain. The acceptance gates are relative L2 <1% and
raw overlap >0.9999 for A/E, not exact response equality. T is not an
acceptance target in this regression.

Reproduce after building/installing the wheel:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

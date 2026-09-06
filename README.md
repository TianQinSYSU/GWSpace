# GWSpace: A multi-mission Science Data Simulator for Space-based Gravitational Wave Detection

GWSpace is a multi-mission science data simulator for space-based gravitational wave detection.
It is a Python package that can compute correlated gravitational wave signals that could be detected by TianQin, LISA and Taiji simultaneously in a possible joint detection scenario,
either in time domain (GCB, EMRI and Burst) or in frequency domain (BHB and SGWB).
For more details, see the [GWSpace documentation](https://github.com/TianQinSYSU/GWSpace/blob/main/docs/GWSpace.pdf), [arXiv:2309.15020](https://arxiv.org/abs/2309.15020), or the published version in [Classical and Quantum Gravity, 42, 165005 (2025)](https://iopscience.iop.org/article/10.1088/1361-6382/adf409) ([DOI:10.1088/1361-6382/adf409](https://doi.org/10.1088/1361-6382/adf409)).

![gwspace-structure](https://raw.githubusercontent.com/TianQinSYSU/GWSpace/main/docs/gwspace-structure.png "gwspace-structure")

## Installation

GWSpace requires a C compiler and [GSL](https://www.gnu.org/software/gsl/) to build its extension modules.

### Conda (recommended)

The environment file installs GSL with Conda and installs GWSpace with all documented optional features except the separately maintained EMRI and ringdown backends:

```shell
git clone https://github.com/TianQinSYSU/GWSpace
cd GWSpace
conda env create --file environment.yml
conda activate gwspace
```

### pip

Install GSL first, for example with `brew install gsl` on macOS or `sudo apt-get install libgsl-dev` on Ubuntu. Then run:

```shell
git clone https://github.com/TianQinSYSU/GWSpace
cd GWSpace
python -m pip install .
```

The build locates GSL through `GSL_PREFIX`, `gsl-config`, the active Conda environment, or `/usr`, in that order. To use a non-standard installation:

```shell
GSL_PREFIX=/path/to/gsl python -m pip install .
```

The same environment variable is used while building the eccentric-waveform extra. Set it explicitly when Python and GSL come from different environment managers.

Optional Python dependencies are grouped by feature:

```shell
python -m pip install ".[eccentric]"    # pyEccentricFD v0.2.0
python -m pip install ".[sgwb]"         # stochastic-background support
python -m pip install ".[emri]"         # EMRI support; requires Python 3.12+
python -m pip install ".[ringdown]"     # ringdown waveform support
# Several groups can be installed together, for example:
python -m pip install ".[eccentric,emri,ringdown]"
```

## GW waveforms

As seen in the figure above, different gravitational-wave sources require different waveforms.
The following waveforms are included in GWSpace unless otherwise noted:

- EMRI: `FastEMRIWaveforms` (`few`)

  - GWSpace supports FastEMRIWaveforms 2.x. Install it with `python -m pip install ".[emri]"` using Python 3.12 or newer.

- Galactic compact binary (GCB): `FastGB` and `GCBWaveform`

  - `FastGB`:
    - A modified version of GCB waveform generation code `Galaxy` in the Mock LISA Data Challenge (MLDC).
    - It uses a fast/slow decomposition of the waveform to reduce the computational cost, see [arXiv:0704.1808](https://arxiv.org/abs/0704.1808) for more details.
  - GCB time-domain waveform generation using Python: See class `GCBWaveform`.

- Binary black hole (BBH): `IMRPhenomD`

  - `pyIMRPhenomD`: `IMRPhenomD` waveform in C code developed by Michael Puerrer.
  - [`PyIMRPhenomD`](https://github.com/XGI-MSU/PyIMRPhenomD): `IMRPhenomD` waveform but in a pure python code, compiled with the numba just in time compiler.
    - If you prefer this one, **you need to install it manually.**

- Stellar-mass BBH (with eccentricity): `EccentricFD`

  - This is a modified version of `EccentricFD` waveform, which is specially for space-detector responses.
    - Original codes see files in [LALSuite](https://github.com/lscsoft/lalsuite/tree/master/lalsimulation/lib)
  - GWSpace uses the separately maintained [pyEccentricFD](https://github.com/HumphreyWang/pyEccentricFD) package.

- Stochastic gravitational wave background (SGWB):

  - With the help of `healpy` to generate a SGWB signal of power law type.

## Author lists

- [En-Kun Li](https://github.com/ekli-sysu)
- [Han Wang](https://github.com/HumphreyWang)
- [Ya-Nan Li](https://github.com/liyn55)
- [Yi-Ming Hu](https://github.com/yiminghu-SYSU)
- ...

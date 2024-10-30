# GWSpace: A multi-mission Science Data Simulator for Space-based Gravitational Wave Detection

GWSpace is a multi-mission science data simulator for space-based gravitational wave detection.
It is a Python package that can compute correlated gravitational wave signals that could be detected by TianQin, LISA and Taiji simultaneously in a possible joint detection scenario,
either in time domain (GCB, EMRI and Burst) or in frequency domain (BHB and SGWB).
For more details, see [doc file](./docs/GWSpace.pdf) or [arXiv:2309.15020](https://arxiv.org/abs/2309.15020).

![gwspace-structure](./docs/gwspace-structure.png?raw=true "gwspace-structure")


## Quick install

```shell
git clone --recurse-submodules https://github.com/TianQinSYSU/GWSpace
cd /GWSpace
pip install -r requirements.txt . --global-option="--with-gsl=/your/gsl/path"
```
- Remove `-r requirements.txt` if you want to install GWSpace only.
- If `--with-gsl` is not given, default `gsl` path is `/usr`.

## GW waveforms

As seen in the figure above, different gravitational wave sources requires different waveforms.
The following waveform will be automatically complied during the installation unless otherwise noted:

- EMRI: `FastEMRIWaveforms` (`few`)

  - We use [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) for EMRI. If you want to do data analysis of EMRI, **you need to install it manually.**
  - It requires `gsl` and `lapack`.

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
  - If you want to check the original codes, see files in [LALSuite](https://github.com/lscsoft/lalsuite/tree/master/lalsimulation/lib).
  - This has been linked as a **submodule** of GWSpace, click [here](https://github.com/HumphreyWang/pyEccentricFD) to check the submodule itself.
    - This submodule is included in `requirements.txt`, it will be installed when installing this list.

- Stochastic gravitational wave background (SGWB):

  - With the help of `healpy` to generate a SGWB signal of power law type.


## Library dependence

### C language: gsl:

All waveforms will be compiled with `gsl`.

- To find the lib path of `gsl`:
```shell
echo $LD_LIBIARY_PATH | grep gsl
```

- To find the software installed manually:
```shell
ldconfig -p | grep gsl
```

- Install `gsl` **only if** no such library in your system. In Ubuntu, you can install it by
```shell
sudo apt-get install libgsl-dev
```

### Python package:

Please check `requirements.txt` for details, or you can directly install them by
```shell
pip install -r requirements.txt
```

### For EMRI: install `lapack` & `lapacke`

- After [downloading](https://www.netlib.org/lapack/) `lapack`, compile both `lapack` and `lapacke`, for example
```shell
cd lapack-3.11.0
cp make.inc.example make.inc
make lapacklib
make lapackelib
```

- Copy the `liblapacke.a` and `liblapack.a` to your lib path (e.g. `/usr/local/lib`)
- Copy files in the `./LAPACKE/include` to your include path (e.g. `/usr/local/include`)

- *If you cannot do copy above*, you need to add lapack path to your environment variables, e.g. add PATH to `~/.bashrc`
```shell
export LIBRARY_PATH=$LIBRARY_PATH:/xxx/lapack-3.11.0
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/xxx/lapack-3.11.0/LAPACKE/include
```

## Author lists

- [En-Kun Li](https://github.com/ekli-sysu)
- [Han Wang](https://github.com/HumphreyWang)
- [Ya-Nan Li](https://github.com/liyn55)
- ...

# GWSpace: A multi-mission Science Data Simulator for Space-based Gravitational Wave Detection

GWSpace is a multi-mission science data simulator for space-based gravitational wave detection. It is a Python package that can compute correlated gravitational wave signals that could be detected by TianQin, LISA and Taiji simultaneously in a possible joint detection scenario, either in time domain (GCB, EMRI and Burst) or in frequency domain (BHB and SGWB). For more details, see [doc file](./docs/GWSpace.pdf) or [arXiv:2309:15020](https://arxiv.org/abs/2309.15020).

![gwspace-structure](./docs/gwspace-structure.png?raw=true "gwspace-structure")


## Quick install

```shell
python setup.py install --with-gsl=/your/gsl/path
```


## GW waveforms

As seen in the figure above, we use different waveforms for different gravitational wave sources:

- EMRI: `FastEMRIWaveforms` or `few`

  - We use [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) for EMRI. If you want to do data analysis of EMRI, you have to install it manually.

  - Need to install `gsl` and `lapack`

The following waveform will be automatically complied during the installation:

- Galactic binary: `FastGB`

  - This is a modified version of 

- Binary black hole (BBH): [`pyIMRPhenomD`](https://github.com/XGI-MSU/PyIMRPhenomD)

  - This module implements the IMRPhenomD in a pure python code, compiled with the numba just in time compiler.

- Stellar-mass BBH (with eccentricity): `EccentricFD`

  - This is a modified version of `EccentricFD` waveform, which is specially for space-detector responses.

  - If you want to check the original codes, see files in [LALSuite](https://github.com/lscsoft/lalsuite/tree/master/lalsimulation/lib)

  - If you want a pure modified version of this waveform, check [this link](https://github.com/HumphreyWang/pyEccentricFD)


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

### Python package:

check `requirements.txt` for details, or you can directly install them by

```shell
pip install -r requirements.txt
```

### For EMRI: install `lapack` & `lapacke`

- As `lapack-3.11.0` already includes `lapacke`, you can compile the `lapack`, and then enter the `LAPACKE` dir, and compile `lapacke`. 
- The generated lib `liblapacke.a` will be generated at the top dir.
- Copy the `liblapacke.a` and `liblapack.a` to your include path (e.g. `/usr/local/include`)
- Copy files in the `LAPACKE/include` to your include path (e.g. `/usr/local/lib`)

## Author lists

- [En-Kun Li](https://github.com/ekli-sysu)
- [Han Wang](https://github.com/HumphreyWang)
- Ya-Nan Li
- ...

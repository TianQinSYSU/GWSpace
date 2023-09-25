# GWSpace: A multi-mission Science Data Simulator for Space-based Gravitational Wave Detection

This is a simple Code for the Science Data Simulator of Space-based gravitational wave detectors.

![gwspace-structure](./docs/gwspace-structure.png?raw=true "gwspace-structure")

To compile the code, using `python setup.py install --with-gsl=/your/gsl/path`

## Lib dependence

1. `gsl`

The waveform `fastgb`, `eccentric`, `pyIMRPhenomD` should be compiled with `gsl`.

2. `FastEMRIWaveforms` or `few`

The waveform of EMRI is dependent on the `FastEMRIWaveforms` code. When you using it, please be sure that you have installed it. The code can be found in [https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms).

Be careful: TO install the `FastEMRIWaveforms`, you need install the `gsl` and `lapack`.

3. Other python lib

In order to use the `PyIMRPhenomD` code, you should install `numpy`, `scipy`, `numba` and `interpolation` in your python environment. 

- `numpy`
- `scipy`
- `numba`
- interpolation : `PyIMRPhenomD` needs this repo

4. tips 

- To find the lib path of `gsl`, you can use
```
$ echo $LD_LIBIARY_PATH | grep gsl
```

- To find the software installed Manually, or use
```
$ ldconfig -p | grep gsl
```


### install `lapack` & `lapacke`

Be careful, `lapack-3.11.0` already include `lapacke`, one should compile the `lapack`, and then enter the `LAPACKE` dir, and compile `lapacke`. 
The generated lib `liblapacke.a` will be generated at the top dir.
Then, copy the `liblapacke.a` and `liblapack.a` and etc in the `lib` to some path of your `lapack` package.
Copy files in the `LAPACKE/include` and head files in the include to your `lapack` include dir.

### install `numba` etc.

Install in your own path
```
pip install numba
```

## Author lists

- [En-Kun Li]()
- [Han Wang]()
- [Ya-Nan Li]()
- ...

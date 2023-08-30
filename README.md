# Code for Space-borne Gravitational-Wave Detectors (CSGWD)

To compile the waveforms, you can use `install_local.sh`

## Lib dependence

In order to use the `PyIMRPhenomD` code, you should install `numpy`, `scipy`, `numba` and `interpolation` in your python environment. 
TO use the `FastEMRIWaveforms`, you need install the `gsl` and `lapack`.

- numpy
- scipy
- numba
- interpolation : `PyIMRPhenomD` needs this repo
- gsl
- lapack, lapacke
- 

### install lapack & lapacke

Be careful, `lapack-3.11.0` already include `lapacke`, one should compile the `lapack`, and then enter the `LAPACKE` dir, and compile `lapacke`. 
The generated lib `liblapacke.a` will be generated at the top dir.
Then, copy the `liblapacke.a` and `liblapack.a` and etc in the `lib` to some path of your `lapack` package.
Copy files in the `LAPACKE/include` and head files in the include to your `lapack` include dir.

### install numba 

Install in your own path
```
pip install numba
```

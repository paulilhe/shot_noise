## Shot Noise processes
This package is related to the simulation and the estimation of the statistical elements
of shot-noise processes based on the two estimation techniques introduced in the paper
Nonparametric estimation of shot-noise processes.

## Download python packages
Get the parallel Python package by running
```
pip install pp
```
Install the patsy library to compute B-splines
```
pip install patsy
```

## Get f2py
f2py allows to wrap Fortran functions to Python.
Follow the installation procedure given by https://sysbio.ioc.ee/projects/f2py2e/#download
and ensure it is well installed by typing
```
f2py
```
To list the available Fortran compilers are installed on your computer, write
```
f2py -c --help-fcompiler
```

## Wrap the Fortran scripts
Build the extension modules needed to perform the estimation procedure
```
f2py -c -m design_operation matrix_operations.f90
```
# 2D-INSE-GFS

This repository contains a solver for the two-dimensional incompressible Navier-Stokes Equation using the Galerkin Fourier Spectral Method. It is provided without a license and is intended strictly for educational purposes. Redistribution, modification, or reuse of this code without our permission is not allowed.

## Image
Vorticity Plot generated by KolmogorovForcing program in this repository.
<p align="center">
   <img src=https://github.com/gkanba/2D-INSE-GFS/blob/master/example/example_omg.svg>
</p>

## Features

This repository includes two primary programs:
1. **Navier-Stokes Equation Solver with Kolmogorov-Type Forcing**  
   A solver implementing Kolmogorov-type forcing for various flow dynamics studies and proof for validity of this computation.

2. **Navier-Stokes Equation Solver with Forcing based on Annulus Fourier Coefficient**  
   A solver using annulus Fourier coefficient forcing for specialized simulations.

## Dependencies

The following libraries are required to build and run this project:
- **[Eigen 3.4](https://eigen.tuxfamily.org/dox/)**: A C++ template library for linear algebra.
- **[FFTW](http://www.fftw.org/)**: A C library for computing the discrete Fourier transform.
- **[nlohmann/json](https://github.com/nlohmann/json)**: A JSON library for C++.

Ensure these libraries are properly installed and linked before building the project.

## Usage
> Detailed instructions for setting up and running the programs will be added here.

## References

We refered some useful and valuable educational contributions below:
- XX1
- XX2
---

All copyrights related to this contribution belongs to Gakuto Kambayashi.

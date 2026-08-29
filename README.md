# Fokker–Planck Simulations with No-Flux Boundary Conditions

Numerical simulation of stationary Fokker–Planck equations on bounded domains using particle-based stochastic simulations.

The corresponding stochastic differential equation is

$$
dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t,
$$

with Fokker–Planck equation

$$
\frac{\partial p}{\partial t}
=-\nabla\cdot(bp)+ \frac{1}{2}\nabla\cdot\nabla\cdot(\Sigma p),
\qquad \Sigma=\sigma\sigma^\mathsf{T}.
$$

Particle motion is approximated using Euler–Maruyama with

$$
X_{n+1}
=X_n + \varepsilon^2 b(X_n)
+\varepsilon \sigma(X_n)\xi_n,
\qquad \varepsilon=\sqrt{\Delta t}.
$$

The main boundary treatments implemented are **reflection** and **rejection**, used to approximate no-flux behaviour at the boundary.

The code is intended for approximating **stationary probability densities**. It does not currently support studying the time evolution of the Fokker–Planck solution. Although can be extended with some modifiction.

## Repository structure

```text
.
├── noFlux1D_notebooks/
│   ├── data_generation_1D.ipynb
│   ├── noFlux1D_main.ipynb
│   └── parameter-specific simulation and analysis notebooks
│
├── noFlux2D_notebooks/
│   ├── data_generation_2D.ipynb
│   ├── noFlux2D_main.ipynb
│   └── parameter-specific simulation and analysis notebooks
│
├── sources/
│   ├── simulation.py
│   ├── models.py
│   ├── analysis.py
│   ├── plotting.py
│   └── data_storage.py
│
└── requirements.txt
```

### 1D simulations

The 1D code considers an interval domain. The notebooks contain different combinations of drift, diffusion coefficient, boundary rule and time-step size.

File and folder names generally encode the parameters used. For example, names such as

```text
b-0_s-1
b-1_s-xp1
```

represent different choices of drift `b` and diffusion coefficient `sigma`.

### 2D simulations

The 2D code mainly considers disk domains, with additional support for rectangular domains.

As in the 1D code, the notebook and data naming conventions describe the drift, diffusion matrix and other simulation settings. For example, `s-I` denotes an identity diffusion matrix, while `s-M` denotes a non-identity matrix.

## Generating data

Simulation data are **not included in this repository**.

The data-generation notebooks are intended to be modified for the required problem. Simulation parameters are specified in the parameter cells, after which the following cells run the simulations for the selected settings.

The main parameters normally varied are:

* drift \(b\);
* diffusion coefficient or matrix \(\sigma\);
* boundary treatment;
* time-step size \(\Delta t=\varepsilon^2\).

The simulations used during development generally used approximately **10 million particles** and final simulation time \(T=2\). These values are not requirements and can be changed as needed.

Users may also wish to define their own data-storage and naming conventions for new experiments.

## Reusable functions

Most reusable functionality is contained in `sources/`.

* `simulation.py` — particle simulation and boundary-handling routines.
* `models.py` — drift and diffusion models. Can be expanded for time-dependent.
* `analysis.py` — analysis and error-calculation utilities.
* `plotting.py` — plotting and visualisation functions.
* `data_storage.py` — utilities for saving and loading generated simulation data.

These functions can be reused with different drift and diffusion settings without relying on the parameter-specific notebooks.

## Requirements

The project was developed using Python 3.12.

Install the required packages with

```bash
pip install -r requirements.txt
```

## Code note

Some of the plotting formatting and adjustment code, as well as parts of the organisational structure of the Python source files, were produced or reorganised with substantial assistance from generative AI.

## Acknowledgement

Developed as part of dissertation work for the **MSc Computational Applied Mathematics** programme at the **University of Edinburgh, 2026**.

## License

This project is released under the **MIT License**. The code may be used, modified and redistributed with appropriate attribution.

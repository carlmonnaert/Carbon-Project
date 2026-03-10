# Carbon Project

Numerical simulation of a simplified carbon cycle using an explicit Euler integrator in Python.

![Stable simulation preview](./data/stable_to_2100_dt_0_dot_1_year.png)

## Overview

This project models exchanges of carbon between 8 reservoirs:

1. Atmosphere
2. Carbonate rock
3. Deep ocean
4. Fossil fuel carbon
5. Plants
6. Soils
7. Surface ocean
8. Vegetated land area (%)

The main script (`carbone.py`) computes time evolution from 1850 to 2100 and generates normalized plots for selected variables.

The current implementation runs a longer scenario by default (1850 to 2600) and exports a traceable PDF filename containing initial conditions, simulated duration, and time step.

## Repository Tree

```text
Carbon Project/
├── .git/
├── .gitignore
├── README.md
├── carbone.py
├── consigne.pdf
├── explanation.pdf
└── data/
└── reports/
```

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

## Run

From the project root:

```bash
python3 carbone.py
```

## Current Simulation Settings

In `main()`:

- Start year: `1850`
- End year: `1850 + 750` (i.e. `2600`)
- Time step: `0.1` year
- Deforestation parameter: `2`

## Output

Running the script displays the figure and also saves it in `./data/` with an encoded name:

- `./data/plot_<run_tag>_years<...>_dt<...>.pdf`

Example:

- `./data/plot_atm7p5e2_rock1e8_deep3p8e4_fossil7p5e3_plant5p6e2_soil1p5e3_surf8p9e2_veg1e2_years7p5e2_dt1e-1.pdf`

Notes on encoding:

- values are represented in scientific notation,
- `p` replaces the decimal point for filename safety (e.g. `7p5e2` means $7.5\times10^2$).

## Plotting

The generated figure uses a clean 2x2 layout with 3 active panels:

- top-left: reservoir dynamics,
- top-right: climate and biosphere effects,
- bottom-left: ocean-geology-land coupling,
- bottom-right: intentionally blank.

Normalization is applied per curve using min-max scaling:

$$
y_{norm}=\frac{y-y_{min}}{y_{max}-y_{min}}
$$

with constant series mapped to `0.5`.

Each legend entry includes the original value range `[min, max]`, and legend text color matches the corresponding curve color.

## Technical Report

A detailed implementation report is provided in:

- `report_carbon_model.tex`

It includes the model equations, numerical method, and a linear/nonlinear decomposition in matrix-vector form.

## Notes

- `@njit` (Numba) decorators are currently commented out.
- Fossil-fuel combustion is computed from piecewise linear interpolation of `FossFuelData`.
- The numerical scheme is forward Euler (`step` function).

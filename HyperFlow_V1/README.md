# HyperFlow V1

HyperFlow is an unsteady one-dimensional compressible reacting-flow solver for variable-area ducts. V1 uses NND/HLLC fluxes, explicit Heun time integration and Strang splitting with Cantera chemistry.

## Sod shock-tube example

A 1 m constant-area tube, initially at rest, with a diaphragm at 0.5 m. Left state: 100 kPa and 300 K; right state: 10 kPa and 240 K. Air, chemistry off, 400 cells.

![Sod shock-tube comparison](examples/sod_shock/sod_compare_params.png)

*Density, pressure, velocity and temperature at 0.4025 ms, compared with the exact ideal-gas Riemann solution using gamma = 1.4.*

## Core capabilities

- Unsteady transport of mass, momentum, total energy and species.
- Prescribed duct area and initial flow field.
- Finite-rate chemistry through Cantera constant-volume reactors.

[Compact formulation](docs/TechnicalNotes.md) · [V2 additions](../HyperFlow_V2/README.md#changes-from-v1)

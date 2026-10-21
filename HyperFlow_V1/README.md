# HyperFlow

One-dimensional variable-area compressible duct flow with optional finite-rate chemistry (Cantera).

## Repository layout

```text
HyperFlow/
├── README.md
├── Makefile
├── src/                 # C++ solver sources
├── docs/                # Theory & verification notes
├── examples/
│   ├── sod_shock/       # chemistry off
│   └── reacting_duct/   # chemistry on
└── tools/               # Initial-field & post-compare scripts
```

## Equations

$$
\partial_t \mathbf{U} + \partial_x \mathbf{F} = \mathbf{S}_A + \mathbf{S}_{\mathrm{chem}}
$$

- $\mathbf{U}=[\rho A,\rho u A,\rho A e_t,\rho A Y_k]^\top$
- $\mathbf{S}_A$: momentum geometric source $pA\,\mathrm{d}\ln A/\mathrm{d}x$
- $\mathbf{S}_{\mathrm{chem}}$: operator-split constant-volume chemistry (`chemistry: true`)

Space: NND + HLLC; time: Heun; Strang splitting when chemistry is on.

## Dependencies

- C++17, Eigen3, nlohmann_json, HDF5, Cantera, OpenMP

## Build

```bash
make
```

## Case layout

```text
case/
  config.json
  initial.h5      # datasets x[nface], A[nface], U[nVar,nface]
  out/            # snapshots after a run
```

Run examples:

```bash
./HyperFlow examples/sod_shock        # chemistry off (Sod)
./HyperFlow examples/reacting_duct   # chemistry on (hot premixed H2/air)

python tools/make_sod_initial.py
python tools/make_reacting_initial.py
```

## `config.json` fields

| Field | Meaning |
|------|------|
| `mechanism` | Cantera mechanism |
| `chemistry` | `true` / `false` |
| `dt`, `t_final`, `n_snapshots` | Time marching |
| `boundary.inlet` | `p,T,u,X` (Riemann inlet reservoir) |
| `boundary.outlet` | `p,T,X` (subsonic back-pressure ambient) |

No injection, mixing, or wall friction; area and initial field come from `initial.h5`.

Theory and Sod verification: see [`docs/TechnicalNotes.md`](docs/TechnicalNotes.md).

## Sod comparison plot

```bash
./HyperFlow examples/sod_shock
python tools/compare_sod.py   # → examples/sod_shock/sod_compare_params.png
```

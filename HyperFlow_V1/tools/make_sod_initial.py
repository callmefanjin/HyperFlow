#!/usr/bin/env python3
"""Create examples/sod_shock/initial.h5 for HyperFlow (dimensional Sod tube)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cantera as ct
import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "sod_shock" / "initial.h5",
    )
    ap.add_argument("--ncell", type=int, default=400)
    ap.add_argument("--mechanism", default="h2o2.yaml")
    ap.add_argument("--X", default="O2:1.0, N2:3.76")
    args = ap.parse_args()

    nface = args.ncell + 1
    x = np.linspace(0.0, 1.0, nface)
    A = np.ones(nface)
    x_membrane = 0.5

    gas = ct.Solution(args.mechanism)
    nsp = gas.n_species
    nVar = 3 + nsp
    U = np.zeros((nVar, nface))

    def fill_state(T: float, p: float, u: float, i: int) -> None:
        gas.TPX = T, p, args.X
        rho = gas.density
        e = gas.int_energy_mass
        et = e + 0.5 * u * u
        Y = gas.Y
        rhoA = rho * A[i]
        U[0, i] = rhoA
        U[1, i] = rhoA * u
        U[2, i] = rhoA * et
        U[3:, i] = rhoA * Y

    for i, xi in enumerate(x):
        if xi < x_membrane:
            fill_state(300.0, 1.0e5, 0.0, i)
        else:
            fill_state(240.0, 1.0e4, 0.0, i)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.attrs["format_version"] = 3
        f.attrs["kind"] = "initial"
        f.attrs["ncell"] = args.ncell
        f.attrs["nface"] = nface
        f.attrs["nVar"] = nVar
        f.attrs["nSpecies"] = nsp
        f.attrs["mechanism"] = args.mechanism
        f.create_dataset("x", data=x)
        f.create_dataset("A", data=A)
        f.create_dataset("U", data=U)

    print(f"Wrote {args.out}  nface={nface} nVar={nVar} rhoL={U[0,0]:.6f} rhoR={U[0,-1]:.6f}")


if __name__ == "__main__":
    main()

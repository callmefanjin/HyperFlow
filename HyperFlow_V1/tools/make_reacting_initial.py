#!/usr/bin/env python3
"""Create examples/reacting_duct/initial.h5 — hot premixed H2/air, constant area."""

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
        default=Path(__file__).resolve().parents[1]
        / "examples"
        / "reacting_duct"
        / "initial.h5",
    )
    ap.add_argument("--ncell", type=int, default=80)
    ap.add_argument("--mechanism", default="h2o2.yaml")
    ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=1200.0)
    ap.add_argument("--p", type=float, default=1.0e5)
    ap.add_argument("--u", type=float, default=50.0)
    ap.add_argument("--L", type=float, default=0.2)
    args = ap.parse_args()

    nface = args.ncell + 1
    x = np.linspace(0.0, args.L, nface)
    A = np.ones(nface)

    gas = ct.Solution(args.mechanism)
    gas.set_equivalence_ratio(args.phi, "H2:1", "O2:1, N2:3.76")
    gas.TP = args.T, args.p
    rho = gas.density
    e = gas.int_energy_mass
    et = e + 0.5 * args.u * args.u
    Y = gas.Y.copy()
    nsp = gas.n_species
    nVar = 3 + nsp

    U = np.zeros((nVar, nface))
    for i in range(nface):
        rhoA = rho * A[i]
        U[0, i] = rhoA
        U[1, i] = rhoA * args.u
        U[2, i] = rhoA * et
        U[3:, i] = rhoA * Y

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

    print(
        f"Wrote {args.out}\n"
        f"  nface={nface} L={args.L}m  T={args.T}K p={args.p:.0f}Pa u={args.u}m/s phi={args.phi}\n"
        f"  rho={rho:.4f}  YH2={float(gas['H2'].Y):.4f}  YO2={float(gas['O2'].Y):.4f}"
    )


if __name__ == "__main__":
    main()

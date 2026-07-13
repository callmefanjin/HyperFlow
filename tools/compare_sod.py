#!/usr/bin/env python3
"""Compare HyperFlow Sod final snapshot vs gamma=1.4 exact Riemann solution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from riemann_exact import riemann_wave_params, sample_riemann_array

GAMMA = 1.4
X_MEMBRANE = 0.5
FS_LABEL = 14
FS_TICK = 13
FS_TITLE = 15
FS_SUPTITLE = 16
FS_LEGEND = 13

WAVE_LINE_KW = {
    "expansion": dict(color="gray", ls=":", lw=1.0, alpha=0.85, label="Expansion wave"),
    "contact": dict(color="green", ls="-.", lw=1.0, alpha=0.85, label="Contact discontinuity"),
    "shock": dict(color="red", ls="--", lw=1.0, alpha=0.85, label="Shock"),
}


def load_final_h5(path: Path):
    with h5py.File(path, "r") as f:
        x = np.asarray(f["x"][:], dtype=float)
        rho = np.asarray(f["rho"][:], dtype=float).ravel()
        p = np.asarray(f["p"][:], dtype=float).ravel()
        u = np.asarray(f["u"][:], dtype=float).ravel()
        T = np.asarray(f["T"][:], dtype=float).ravel()
        t = float(f.attrs.get("time", 0.0))
        A = np.asarray(f["A"][:], dtype=float).ravel() if "A" in f else np.ones_like(x)
    return x, A, rho, p, u, T, t


def add_exact_wave_lines(ax, x_ex):
    def kw(name):
        return {k: v for k, v in WAVE_LINE_KW[name].items() if k != "label"}

    ax.axvline(x_ex["rarefaction_head"], **kw("expansion"))
    ax.axvline(x_ex["rarefaction_tail"], **kw("expansion"))
    ax.axvline(x_ex["contact"], **kw("contact"))
    ax.axvline(x_ex["shock"], **kw("shock"))


def make_unified_legend(fig, t_phys):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0",
               markersize=6, alpha=0.65, label="HyperFlow"),
        Line2D([0], [0], color="C1", ls="--", lw=2.0,
               label=f"Exact ($\\gamma$={GAMMA}, t={t_phys}s)"),
        Line2D([0], [0], color=WAVE_LINE_KW["expansion"]["color"],
               ls=WAVE_LINE_KW["expansion"]["ls"], lw=1.2,
               label=WAVE_LINE_KW["expansion"]["label"]),
        Line2D([0], [0], color=WAVE_LINE_KW["contact"]["color"],
               ls=WAVE_LINE_KW["contact"]["ls"], lw=1.2,
               label=WAVE_LINE_KW["contact"]["label"]),
        Line2D([0], [0], color=WAVE_LINE_KW["shock"]["color"],
               ls=WAVE_LINE_KW["shock"]["ls"], lw=1.2,
               label=WAVE_LINE_KW["shock"]["label"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=FS_LEGEND)


def plot_four_panels(fig, axs, x_hf, series_hf, series_ex, x_ex, t_phys):
    hf_kw = dict(color="C0", marker="o", markersize=3, linestyle="none", alpha=0.65)
    ex_kw = dict(color="C1", linewidth=2.0, linestyle="--")
    panels = [
        (axs[0, 0], series_hf[0], series_ex[0], r"$\rho$ (kg/m$^3$)", "Density"),
        (axs[0, 1], series_hf[1], series_ex[1], r"$p$ (Pa)", "Pressure"),
        (axs[1, 0], series_hf[2], series_ex[2], r"$u$ (m/s)", "Velocity"),
        (axs[1, 1], series_hf[3], series_ex[3], r"$T$ (K)", "Temperature"),
    ]
    for ax, y_hf, y_ex, ylabel, title in panels:
        ax.plot(x_hf, y_hf, **hf_kw)
        ax.plot(x_hf, y_ex, **ex_kw)
        add_exact_wave_lines(ax, x_ex)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
    for ax in axs[1, :]:
        ax.set_xlabel(r"$x$ (m)", fontsize=FS_LABEL)
    make_unified_legend(fig, t_phys)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path,
                    default=root / "examples" / "sod_shock")
    ap.add_argument("--h5", type=Path, default=None,
                    help="Final state HDF5 (default: case/out/state_t1of1.h5)")
    args = ap.parse_args()
    case = args.case
    h5_path = args.h5 or (case / "out" / "state_t1of1.h5")
    cfg = json.loads((case / "config.json").read_text())

    x, A, rho, p, u, T, t_h5 = load_final_h5(h5_path)
    t_phys = float(cfg["t_final"])
    if abs(t_h5 - t_phys) > 1e-12:
        print(f"WARNING: HDF5 time={t_h5} vs config t_final={t_phys}")

    inlet = cfg["boundary"]["inlet"]
    outlet = cfg["boundary"]["outlet"]
    p_L, T_L = float(inlet["p"]), float(inlet["T"])
    p_R, T_R = float(outlet["p"]), float(outlet.get("T", T_L))
    # Use numerical left density + ideal-gas R consistent with T_L,p_L
    rho_L = float(rho[0])
    R_gas = p_L / (rho_L * T_L)
    rho_R = p_R / (R_gas * T_R)

    ic = dict(u_L=0.0, rho_L=rho_L, p_L=p_L, u_R=0.0, rho_R=rho_R, p_R=p_R, R_gas=R_gas)
    xi = x - X_MEMBRANE
    u_ex, p_ex, rho_ex = sample_riemann_array(
        xi, t_phys, ic["u_L"], ic["rho_L"], ic["p_L"],
        ic["u_R"], ic["rho_R"], ic["p_R"], GAMMA,
    )
    T_ex = p_ex / (rho_ex * R_gas)

    w = riemann_wave_params(
        ic["u_L"], ic["rho_L"], ic["p_L"],
        ic["u_R"], ic["rho_R"], ic["p_R"], GAMMA,
    )
    x_ex = {
        "rarefaction_head": X_MEMBRANE + w["Z_1_head"] * t_phys,
        "rarefaction_tail": X_MEMBRANE + w["Z_1_tail"] * t_phys,
        "contact": X_MEMBRANE + w["u_star"] * t_phys,
        "shock": X_MEMBRANE + w["Z_2"] * t_phys,
    }

    print("=== IC / exact star ===")
    print(f"  left:  rho={rho_L:.5f}, p={p_L:.0f}, T={T_L}")
    print(f"  right: rho={rho_R:.5f}, p={p_R:.0f}, T={T_R}")
    print(f"  t={t_phys}s  p*={w['p_star']:.1f}  u*={w['u_star']:.2f}")
    print(f"  waves x: head={x_ex['rarefaction_head']:.4f} tail={x_ex['rarefaction_tail']:.4f} "
          f"contact={x_ex['contact']:.4f} shock={x_ex['shock']:.4f}")

    # CSV (HyperFlow final field)
    csv_path = case / "sod_final.csv"
    with csv_path.open("w") as f:
        f.write("x,A,rho,p,u,T,rho_exact,p_exact,u_exact,T_exact\n")
        for i in range(len(x)):
            f.write(
                f"{x[i]:.10e},{A[i]:.10e},{rho[i]:.10e},{p[i]:.10e},{u[i]:.10e},{T[i]:.10e},"
                f"{rho_ex[i]:.10e},{p_ex[i]:.10e},{u_ex[i]:.10e},{T_ex[i]:.10e}\n"
            )
    print(f"Wrote {csv_path}")

    series_hf = (rho, p, u, T)
    series_ex = (rho_ex, p_ex, u_ex, T_ex)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    plot_four_panels(fig, axs, x, series_hf, series_ex, x_ex, t_phys)
    fig.suptitle(f"Parameter comparison (dimensional)  t={t_phys}s",
                 fontsize=FS_SUPTITLE, y=1.07)
    plt.tight_layout()
    png_path = case / "sod_compare_params.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {png_path}")

    # L2 in star region
    mask = (x > 0.55) & (x < 0.58)
    print("star 0.55-0.58 L2: "
          f"rho={np.sqrt(np.mean((rho[mask]-rho_ex[mask])**2)):.4e} "
          f"p={np.sqrt(np.mean((p[mask]-p_ex[mask])**2)):.2f} "
          f"u={np.sqrt(np.mean((u[mask]-u_ex[mask])**2)):.3f}")


if __name__ == "__main__":
    main()

"""Re-plot the multi-sensor front/back + per-sensor maps with a GREY silhouette
underlay, so the three states are visually distinct:
    grey   = present but on the OTHER side (shown in the paired panel)
    dark   = this side, poorly covered
    bright = this side, well covered
Reads experiments/multi_sensor_trade/unions.npz. No GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from results import output_root   # noqa: E402

OUT = output_root() / "experiments" / "multi_sensor_trade"
SEES_FRONT = {"R-back": True, "L-back": True, "R-fwd": False, "L-fwd": False}


def recov(u, present, S_full):
    r = np.full(len(u), -30.0)
    r[present] = np.clip(20 * np.log10(u[present] / S_full[present] + 1e-30), -30, 0)
    return r


def panel(ax, xsign, side, rec, present, q0, title):
    ax.scatter(xsign * q0[present, 0], q0[present, 2], s=2, c="0.82", linewidths=0)   # silhouette
    m = present & side
    sc = ax.scatter(xsign * q0[m, 0], q0[m, 2], s=2, c=rec[m], cmap="inferno",
                    vmin=-30, vmax=0, linewidths=0)
    cov = 100.0 * (rec[m] >= -10).sum() / max(m.sum(), 1)
    ax.set_aspect("equal"); ax.set_title(f"{title}\ncov@-10dB = {cov:.0f}%", fontsize=9)
    ax.set_xlabel("width x (m)")
    return sc


def main():
    d = np.load(OUT / "unions.npz")
    q0, present, S_full, front = d["q0"], d["present"], d["S_full"], d["front"]
    angles = [int(a) for a in d["angles"]]
    print(f"re-plotting {len(angles)} angle(s): {angles}")

    for a in angles:
        # front/back union
        rec = recov(d[f"u{a}"], present, S_full)
        fig, axs = plt.subplots(1, 2, figsize=(8, 7), dpi=120)
        panel(axs[0], +1, front, rec, present, q0, "FRONT")
        sc = panel(axs[1], -1, ~front, rec, present, q0, "BACK (mirrored L-R)")
        axs[0].set_ylabel("height z (m)")
        fig.colorbar(sc, ax=axs, label="recovered vs potential (dB)")
        fig.suptitle(f"4-sensor union, cant {a}°   (grey = present / other side)")
        fig.savefig(OUT / f"blindmap_{a}.png"); plt.close(fig)

        # per-sensor
        fig, axs = plt.subplots(2, 2, figsize=(9, 11), dpi=110)
        for ax, name in zip(axs.flat, ["R-back", "L-back", "R-fwd", "L-fwd"]):
            sf = SEES_FRONT[name]
            rec_s = recov(d[f"{name}__{a}"], present, S_full)
            sc = panel(ax, 1 if sf else -1, front if sf else ~front, rec_s, present, q0,
                       f"{name}  (sees {'FRONT' if sf else 'BACK'})")
        fig.colorbar(sc, ax=axs, label="recovered vs potential (dB)")
        fig.suptitle(f"What each sensor sees — cant {a}°   (grey = present / other side)")
        fig.savefig(OUT / f"persensor_{a}.png"); plt.close(fig)
        print(f"  cant {a}: blindmap + persensor re-plotted")

    print(f"Done. -> {OUT}")


if __name__ == "__main__":
    main()

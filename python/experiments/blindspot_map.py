"""Blind-spot map: body regions no aspect ever lights, from the cached potential.

Loads the full-aspect potential built by coverage_vs_range.py (needs a matching
--pose/--decimate/--ref-* so the cache exists) and marks patches whose potential
never rises within PRESENT_DB of the peak = never returns at any aspect (deep
self-occlusion: armpits, inner thighs, behind arms, ...). No GPU needed.

Output -> RA_OUTPUT_ROOT/experiments/blindspot/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from results import output_root         # noqa: E402
from bodies import load_real_scene      # noqa: E402
from experiments.coverage_vs_range import PRESENT_DB  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("--ref-range", type=float, default=90.0)
    p.add_argument("--ref-step", type=float, default=0.5)
    args = p.parse_args()

    out = output_root() / "experiments" / "blindspot"
    out.mkdir(parents=True, exist_ok=True)

    q_full, _ = load_real_scene(args.pose, assets=[], zbuffer=False, density_weight=True)
    q0 = q_full[::args.decimate]

    cache = (output_root() / "experiments" / "potential_cache" /
             f"{Path(args.pose).name}_dec{args.decimate}_r{args.ref_range:.0f}_s{args.ref_step:.2f}.npy")
    if not (cache.exists() and len(np.load(cache, mmap_mode="r")) == len(q0)):
        sys.exit(f"potential cache missing: {cache}")
    S = np.load(cache)
    present = S >= S.max() * 10 ** (PRESENT_DB / 20.0)
    blind = ~present
    print(f"blind (never within {PRESENT_DB:.0f} dB of peak at any aspect): "
          f"{blind.sum():,}/{len(q0):,} ({100*blind.mean():.1f}%)")

    fig, ax = plt.subplots(figsize=(4.5, 7.5), dpi=130)
    ax.scatter(q0[present, 0], q0[present, 2], s=2, c="0.75", linewidths=0, label="covered")
    ax.scatter(q0[blind, 0], q0[blind, 2], s=4, c="crimson", linewidths=0, label="BLIND")
    ax.set_aspect("equal"); ax.set_xlabel("width x (m)"); ax.set_ylabel("height z (m)")
    ax.set_title(f"Blind spots (full-aspect ±{args.ref_range:.0f}° potential)\n"
                 f"{100*blind.mean():.1f}% of body never lit  (decimate {args.decimate})")
    ax.legend(loc="upper right", markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "blindspot_map.png"); plt.close(fig)
    (out / "blindspot.txt").write_text(
        f"blind {blind.sum()}/{len(q0)} ({100*blind.mean():.2f}%); "
        f"ref +-{args.ref_range:.0f} deg / {args.ref_step:.2f}; decimate {args.decimate}\n",
        encoding="utf-8")
    print(f"Done. -> {out}")


if __name__ == "__main__":
    main()

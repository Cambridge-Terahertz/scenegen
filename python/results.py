"""Layered result storage + previews for browsing on a mounted NAS.

Per run we write a small, self-contained folder::

    <run_dir>/
        meta.json                       # config / pose / timing  (tiny)
        results.h5                      # heavy arrays, chunked + gzipped
            /recon/img       (nx,ny,nz) complex64   reconstructed volume
            /recon/grid/{xs,ys,zs}                   voxel axes
            /fm/{sig,f,tx,rx}                        forward-model data (optional)
            /model/targ      (ntarg,3)  float32      scatterer cloud (optional)
        preview/
            max_projections.png         # XY / XZ / YZ, dB  (view instantly)

The heavy arrays live in HDF5 so a viewer can pull a single slice over the
network without loading the whole volume. The preview PNG is what you actually
glance at from the Mac mount; ``contact_sheet`` tiles them into a campaign index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

import h5py
import matplotlib
matplotlib.use("Agg")  # headless: no display needed on the server
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

__all__ = ["save_run", "load_run", "contact_sheet"]

_H5_OPTS = dict(chunks=True, compression="gzip", compression_opts=4)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_run(run_dir, *, img: np.ndarray, grid, meta: dict,
             sig: Optional[np.ndarray] = None, f: Optional[np.ndarray] = None,
             tx: Optional[np.ndarray] = None, rx: Optional[np.ndarray] = None,
             targ: Optional[np.ndarray] = None, db_floor: float = -40.0) -> Path:
    """Write the layered layout for one run. Returns the run directory.

    img  : (nx, ny, nz) complex reconstructed volume
    grid : object with .xs/.ys/.zs voxel axes (recon.ReconGrid)
    meta : JSON-serialisable run metadata
    """
    run_dir = Path(run_dir)
    (run_dir / "preview").mkdir(parents=True, exist_ok=True)

    with h5py.File(run_dir / "results.h5", "w") as h:
        rec = h.create_group("recon")
        rec.create_dataset("img", data=np.asarray(img, np.complex64), **_H5_OPTS)
        g = rec.create_group("grid")
        for name, ax in (("xs", grid.xs), ("ys", grid.ys), ("zs", grid.zs)):
            g.create_dataset(name, data=np.asarray(ax, np.float64))
        if sig is not None:
            fm = h.create_group("fm")
            fm.create_dataset("sig", data=np.asarray(sig, np.complex64), **_H5_OPTS)
            for name, arr in (("f", f), ("tx", tx), ("rx", rx)):
                if arr is not None:
                    fm.create_dataset(name, data=np.asarray(arr, np.float64))
        if targ is not None:
            h.create_dataset("model/targ", data=np.asarray(targ, np.float32), **_H5_OPTS)

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _save_max_projections(run_dir / "preview" / "max_projections.png",
                          img, grid, db_floor, title=str(meta.get("run_id", run_dir.name)))
    return run_dir


def _save_max_projections(path, img, grid, db_floor, title=""):
    mag = np.abs(np.asarray(img))
    peak = mag.max() if mag.size else 1.0
    db = 20.0 * np.log10(np.maximum(mag, peak * 1e-9) / max(peak, 1e-30))

    # img axes are (x, y=depth, z=height)
    projections = [
        ("XY  (width-depth)", db.max(axis=2).T, grid.xs, grid.ys, "x [m]", "depth y [m]"),
        ("XZ  (width-height)", db.max(axis=1).T, grid.xs, grid.zs, "x [m]", "height z [m]"),
        ("YZ  (depth-height)", db.max(axis=0).T, grid.ys, grid.zs, "depth y [m]", "height z [m]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, P, h, v, xl, yl) in zip(axes, projections):
        im = ax.imshow(P, origin="lower", aspect="auto", cmap="jet",
                       vmin=db_floor, vmax=0,
                       extent=[h[0], h[-1], v[0], v[-1]])
        ax.set_title(name); ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, label="dB")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load (lazy by default)
# ---------------------------------------------------------------------------
def load_run(run_dir, *, load_img: bool = True, load_fm: bool = False) -> dict:
    """Read back a run. With load_img=False you get the h5 handle for slicing
    (e.g. a single depth plane) without pulling the whole volume."""
    run_dir = Path(run_dir)
    out = {"meta": json.loads((run_dir / "meta.json").read_text())}
    h = h5py.File(run_dir / "results.h5", "r")
    out["grid"] = {k: h[f"recon/grid/{k}"][:] for k in ("xs", "ys", "zs")}
    if load_img:
        out["img"] = h["recon/img"][:]
    else:
        out["img"] = h["recon/img"]  # lazy h5 dataset; caller closes via out["_h5"]
    if load_fm and "fm" in h:
        out["fm"] = {k: h[f"fm/{k}"][:] for k in h["fm"]}
    out["_h5"] = h
    return out


# ---------------------------------------------------------------------------
# Campaign index: tile run previews into one browseable image
# ---------------------------------------------------------------------------
def contact_sheet(results_root, out_png, *, preview="preview/max_projections.png",
                  cols: int = 3, thumb_w: int = 700) -> Path:
    """Tile every run's preview PNG into a labelled grid for quick browsing."""
    results_root = Path(results_root)
    runs = sorted(d for d in results_root.iterdir()
                  if d.is_dir() and (d / preview).exists())
    if not runs:
        raise FileNotFoundError(f"no runs with {preview} under {results_root}")

    thumbs = []
    for d in runs:
        im = Image.open(d / preview).convert("RGB")
        h = int(im.height * thumb_w / im.width)
        thumbs.append((d.name, im.resize((thumb_w, h), Image.LANCZOS)))

    pad, label_h = 8, 26
    cell_w = thumb_w + 2 * pad
    cell_h = max(t.height for _, t in thumbs) + label_h + 2 * pad
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for i, (name, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = c * cell_w + pad, r * cell_h + pad
        draw.text((x, y), name, fill="black", font=font)
        sheet.paste(t, (x, y + label_h))

    out_png = Path(out_png)
    sheet.save(out_png)
    return out_png


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # tiny end-to-end smoke test on CPU: forward+recon two scenes -> save -> sheet
    import tempfile
    import raforward as rf
    from recon import ReconGrid, backproject, C0

    root = Path(tempfile.mkdtemp(prefix="ra_results_"))
    grid = ReconGrid(xs=np.linspace(-0.2, 0.2, 41),
                     ys=np.linspace(1.85, 2.15, 31),
                     zs=np.linspace(-0.3, 0.3, 41))
    for tag, target in [("pose_a", [0.05, 2.0, -0.1]), ("pose_b", [-0.1, 2.05, 0.08])]:
        demo = rf.Scene.demo(ntarg=1, nrx=400, nf=128)
        scene = rf.Scene(q=np.array([target], float), s=np.array([1.0]),
                         tx=demo.tx, rx=demo.rx, k=demo.k)
        f = scene.k * C0 / (2 * np.pi)
        sig = rf.forward(scene, backend="cpu")
        img = backproject(sig, tx=scene.tx, rx=scene.rx, f=f, grid=grid,
                          npad=2048, backend="cpu")
        save_run(root / tag, img=img, grid=grid,
                 meta={"run_id": tag, "target": target, "nf": scene.nf, "nrx": scene.nrx},
                 sig=sig, f=f, tx=scene.tx, rx=scene.rx)
        print(f"saved {tag}: {(root / tag / 'results.h5').stat().st_size/1e3:.0f} KB h5")

    # lazy slice read (no full-volume load)
    r = load_run(root / "pose_a", load_img=False)
    mid = r["img"][:, r["img"].shape[1] // 2, :]   # one depth plane only
    print(f"lazy slice shape {mid.shape} (pulled one plane, not the volume)")
    r["_h5"].close()

    sheet = contact_sheet(root, root / "index.png")
    print(f"contact sheet -> {sheet}")
    print(f"\nbrowse: open {root}/index.png  (and per-run preview/max_projections.png)")

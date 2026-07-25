import numpy as np, glob, os, csv, time
D = "/home/jonah/dataset4_2026-07-21"
fs = sorted(glob.glob(D + "/*.npz"))
COLS = ["file","body","gender","height","build","stance","obj_name","is_threat","site",
        "stature_cm","mass_kg","bmi","waist_cm","n_seated","n_visible","visibility",
        "obliquity_deg","han_peak","han_obj_db","han_clutter","han_contrast_db"]
rows = []
t0 = time.time()
for i, f in enumerate(fs):
    d = np.load(f, allow_pickle=True)
    r = {"file": os.path.basename(f)}
    for k in COLS[1:]:
        try:
            v = d[k]
            r[k] = v.item() if v.shape == () else float("nan")
        except Exception:
            r[k] = float("nan")
    # n_obj voxels (object extent proxy)
    try:
        r["n_obj_vox"] = int(d["obj_vox"].shape[0])
    except Exception:
        r["n_obj_vox"] = 0
    rows.append(r)
    if (i+1) % 500 == 0:
        print("%d/%d  %.0fs" % (i+1, len(fs), time.time()-t0), flush=True)
out = "/home/jonah/probe/ds4_table.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS + ["n_obj_vox"])
    w.writeheader()
    w.writerows(rows)
print("WROTE", out, len(rows), "rows in %.0fs" % (time.time()-t0), flush=True)

"""Build pilot_fine_driver.py from pilot_aperture_driver.py: image grid step DS7_GSTEP (m) instead of a
hard-coded 1 cm, so each aperture is imaged at <= PSF/2. The contrast/clutter metric keeps its own
1 cm local cube (cost, and comparability with the 2026-09-30 pilot). Asserts each edit hits once."""
src = open("pilot_aperture_driver.py").read()
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a[:80], src.count(a)); src = src.replace(a, b)
rep('DS7_NAME = "pilot_aperture_2026-09-30"', 'DS7_NAME = "pilot_fine_2026-10-01"')
rep('XS = np.arange(-0.90, 0.9001, 0.01)      # x, cross-range, +ve to the array\'s LEFT\n'
    'YS = np.arange(-1.05, 1.0501, 0.01)      # y, vertical, +ve up',
    'GSTEP = float(_env("GSTEP", "0.01"))       # image grid step (m); set <= PSF/2 = lam*R/(2D)\n'
    'XS = np.arange(-0.90, 0.90 + GSTEP/2, GSTEP)   # x, cross-range, +ve to the array\'s LEFT\n'
    'YS = np.arange(-1.05, 1.05 + GSTEP/2, GSTEP)   # y, vertical, +ve up\n'
    'XS1 = np.arange(-0.90, 0.9001, 0.01); YS1 = np.arange(-1.05, 1.0501, 0.01)   # 1 cm, clutter metric only')
rep("    ix = np.clip(np.round((pts[:, 0]-grid.xs[0])/0.01).astype(np.int64), 0, nx-1)\n"
    "    iy = np.clip(np.round((pts[:, 1]-grid.ys[0])/0.01).astype(np.int64), 0, ny-1)",
    "    ix = np.clip(np.round((pts[:, 0]-grid.xs[0])/GSTEP).astype(np.int64), 0, nx-1)\n"
    "    iy = np.clip(np.round((pts[:, 1]-grid.ys[0])/GSTEP).astype(np.int64), 0, ny-1)")
rep("        lxs = XS[(XS >= gc[0]-0.27) & (XS <= gc[0]+0.27)]\n        lys = YS[(YS >= gc[1]-0.27) & (YS <= gc[1]+0.27)]",
    "        lxs = XS1[(XS1 >= gc[0]-0.27) & (XS1 <= gc[0]+0.27)]\n        lys = YS1[(YS1 >= gc[1]-0.27) & (YS1 <= gc[1]+0.27)]")
rep("    ix = np.clip(np.round((q_vis[:, 0] - XS[0]) / 0.01).astype(np.int32), 0, len(XS) - 1)\n"
    "    iy = np.clip(np.round((q_vis[:, 1] - YS[0]) / 0.01).astype(np.int32), 0, len(YS) - 1)",
    "    ix = np.clip(np.round((q_vis[:, 0] - XS[0]) / GSTEP).astype(np.int32), 0, len(XS) - 1)\n"
    "    iy = np.clip(np.round((q_vis[:, 1] - YS[0]) / GSTEP).astype(np.int32), 0, len(YS) - 1)")
rep("                gx = np.clip(np.round((gk[:,0]-XS[0])/0.01).astype(np.int32), 0, len(XS)-1)\n"
    "                gy = np.clip(np.round((gk[:,1]-YS[0])/0.01).astype(np.int32), 0, len(YS)-1)",
    "                gx = np.clip(np.round((gk[:,0]-XS[0])/GSTEP).astype(np.int32), 0, len(XS)-1)\n"
    "                gy = np.clip(np.round((gk[:,1]-YS[0])/GSTEP).astype(np.int32), 0, len(YS)-1)")
rep("rec.update(aper_D_m=AP_D, pitch_m=PITCH,", "rec.update(grid_step_m=GSTEP, aper_D_m=AP_D, pitch_m=PITCH,")
assert "/0.01)" not in src.replace("(gk[:, 0]-xs0)/0.01", "").replace("(gk[:, 1]-ys0)/0.01", ""), "stray 1 cm"
open("pilot_fine_driver.py", "w").write(src); print("written pilot_fine_driver.py", len(src))

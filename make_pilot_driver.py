"""Build pilot_aperture_driver.py from dataset7_driver.py (ds7 untouched). Adds:
  DS7_D (aperture, m), DS7_PITCH (element pitch, m), DS7_NFREQ, DS7_ROI (half-width, m; 0=off),
  DS7_SITES (comma list), DS7_NOBASE=1 (skip body-only baseline), per-site body crop with a
  3 cm raised-cosine edge taper, provenance fields. Asserts each replacement hits exactly once."""
src = open("dataset7_driver.py").read()
def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a[:70], src.count(a)); src = src.replace(a, b)
rep('DS7_NAME = "dataset7_2026-09-01"', 'DS7_NAME = "pilot_aperture_2026-09-30"')
rep('NFREQ = 256', 'NFREQ = int(_env("NFREQ", "256"))')
rep('g = np.linspace(-0.067 / 2, 0.067 / 2, 130)',
    'AP_D = float(_env("D", "0.067"))\nPITCH = float(_env("PITCH", str(0.067 / 129)))\n'
    'NEL = int(round(AP_D / PITCH)) + 1\ng = np.linspace(-AP_D / 2, AP_D / 2, NEL)\n'
    'ROI_W = float(_env("ROI", "0"))        # half-width of the per-site body crop (m); 0 = full body\n'
    'ROI_T = 0.03                          # raised-cosine edge taper width (m)')
rep('RX = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(130 * 130)])',
    'RX = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(NEL * NEL)])')
rep('APER = _taylor(130, nbar=5, sll=_sll, norm=False)', 'APER = _taylor(NEL, nbar=5, sll=_sll, norm=False)')
rep('APER = np.hanning(130)', 'APER = np.hanning(NEL)')
rep('APER = np.ones(130)', 'APER = np.ones(NEL)')
rep('PLAUSIBLE = {(n, s) for n, sites in pc.PLACEMENT.items() for s in sites}',
    'PLAUSIBLE = {(n, s) for n, sites in pc.PLACEMENT.items() for s in sites}\n'
    '_SITES = _env("SITES", "")\n'
    'if _SITES:\n'
    '    _keep = set(_SITES.split(","))\n'
    '    SITES_OF_VIEW = {v: [s for s in ss if s in _keep] for v, ss in SITES_OF_VIEW.items()}\n'
    'NOBASE = _env("NOBASE", "0") == "1"')
rep('    out = [("none", "nothreat")]', '    out = [] if NOBASE else [("none", "nothreat")]')
# per-site context builder, defined before the main loop
rep('nscan = 0\nt_start = time.time()',
'''def _taper(d):
    t = np.clip(d / ROI_T, 0.0, 1.0); return 0.5 - 0.5 * np.cos(np.pi * t)

def site_ctx(site, model, Vp, F, stance, yc, bq, sb, keep_b0):
    """Body crop around the site's centroid: points within +/-ROI_W in x and y (world), edge-tapered."""
    pl0 = anatomy.sample_placement(model, site, np.random.default_rng(0), jitter=False)
    tri = Vp[F[pl0["face"]]]; c = (np.asarray(pl0["bary"])[:, None] * tri).sum(0)
    cw = to_world(c[None].astype(np.float64), stance, yc)[0]
    dx = ROI_W - np.abs(bq[:, 0] - cw[0]); dy = ROI_W - np.abs(bq[:, 1] - cw[1])
    roi = keep_b0 & (dx > 0) & (dy > 0)
    sbw = sb * (_taper(dx) * _taper(dy))
    sig_b = fwd(bq[roi], sbw[roi])
    zlo, zhi = bq[roi][:, 2].min(), bq[roi][:, 2].max()
    grid_s = ReconGrid(xs=XS, ys=YS, zs=np.arange(zlo - 0.05, zhi + 0.05, DR))
    return dict(roi=roi, sbw=sbw, sig_body=sig_b, grid=grid_s, dmap=depth_map_of(bq[roi]),
                env=build_envelope(grid_s, bq[roi]) if ENVELOPE else None, centre=cw)

nscan = 0
t_start = time.time()
print(f"  PILOT: D={AP_D*100:.1f} cm  pitch={PITCH*1000:.2f} mm  NEL={NEL}x{NEL}  NFREQ={NFREQ}  "
      f"ROI={'off' if ROI_W <= 0 else f'+/-{ROI_W*100:.0f} cm'}", flush=True)''')
# skip the full-body forward/envelope when cropping
rep('        sig_body = fwd(bq[keep_b0], sb[keep_b0])',
    '        sig_body = fwd(bq[keep_b0], sb[keep_b0]) if ROI_W <= 0 else None\n        _ctx = {}')
rep('        env = build_envelope(grid, bq[keep_b0]) if ENVELOPE else None   # conformal body shell (cached)',
    '        env = build_envelope(grid, bq[keep_b0]) if (ENVELOPE and ROI_W <= 0) else None   # conformal body shell (cached)\n'
    '        keep_full, sb_full = keep_b0, sb')
# per placement: bind the site context
rep('''            if os.path.exists(f_out):
                nscan += 1; continue
''','''            if os.path.exists(f_out):
                nscan += 1; continue
            if ROI_W > 0:
                if site not in _ctx:
                    _ctx[site] = site_ctx(site, model, Vp, F, stance, yc, bq, sb_full, keep_full)
                cx = _ctx[site]
                keep_b0, sb, sig_body, grid, dmap, env = cx["roi"], cx["sbw"], cx["sig_body"], cx["grid"], cx["dmap"], cx["env"]
                field_base = None
''')
# provenance
rep('                       zero_vis=False,\n                       **common)',
    '                       zero_vis=False,\n                       **common)\n'
    '            rec.update(aper_D_m=AP_D, pitch_m=PITCH, n_el=NEL, nfreq=NFREQ, roi_w_m=ROI_W,\n'
    '                       zs=grid.zs.astype(np.float32), depth_map=dmap,\n'
    '                       roi_centre=(_ctx[site]["centre"] if ROI_W > 0 else np.full(3, np.nan)))')
rep('f_out = f"{OUT}/ds7_{bid}', 'f_out = f"{OUT}/pilot_D{int(round(AP_D*1000))}_{bid}')
rep('    if all(os.path.exists(f"{OUT}/ds7_{bid}', '    if all(os.path.exists(f"{OUT}/pilot_D{int(round(AP_D*1000))}_{bid}')
open("pilot_aperture_driver.py", "w").write(src); print("written", len(src))

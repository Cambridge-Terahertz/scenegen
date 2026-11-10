"""Dataset 7 (ds7): dataset6 plus object-fidelity, stance and multipath fixes.

WHAT CHANGED vs dataset6 -- all measured, see the commit log:
  * GUN MATERIALS. Guns 2-5 were solid steel AND carried interior surface no sensor can see.
    Stripping removes 57-65% of them; the polymer/steel split then costs a consistent
    -8.5 to -9.1 dB. gun2 gets BRIGHTER when stripped -- interior was partly cancelling.
  * OBJECT-SCALE Z-BUFFER. The joint body+object pass uses a 2 cm window tuned for a 30 cm
    body, so a 3 cm object kept its own interior. Ray-cast truth: the old chain was +8.9 dB
    wrong face-on and +2.3 dB at grazing even after stripping.
  * L3 STANCE. Arms up, forearms vertical, palms to the array (screening_pose(arms="up")).

NO MULTIPATH, deliberately. The threat-skin dihedral corner -- the double bounce this
single-bounce model structurally cannot produce -- was built and validated separately in
~/stop_pose/multipath.py (stationary_pairs + return_visible + fast_double_bounce_from_pairs).
Its 17-site occlusion survey settled the question: corner-vs-scene runs -21 dB at the very
strongest (wb_front_right, front hip) down to -72 dB, and the BOOST TO SCENE PEAK IS +0.0 dB AT
EVERY SITE. A gun lying flat on skin has a median contact wedge of 159 deg, deep in the dead zone
(a corner needs 68-114 deg), so only ~12% of contact points form one at all. The corner is a real,
aspect-robust SECOND signature at hip/appendix carries -- worth confidence and threat/non-threat
discrimination there -- but it never exceeds the direct return, so it would add nothing to a
detector trained on these fields.

It is also computationally out of reach here: measured on realistic clouds, stationary_pairs is
68 s and return_visible 336 s PER PLACEMENT, ~405 s against the ~9 s a whole scan takes now. ds7
runs ~320 placements per body, so it is ~6 h/body for the pairing alone. If the hip discrimination
question is ever revisited, run it as a separate subset study, not in this loop.

Do NOT reintroduce the flat-mirror image-source shortcut that was briefly tried here: mirroring
Tx/Rx across the seat plane gives every object point a specular partner on an infinite plane, so
the whole gun face participates. That is the same divergence the band-pairing engine showed (a
shallow contact has a wide in-phase band and sums as the full pair count) and it produced a bounce
BRIGHTER than the direct return. stationary_pairs -- one specular partner per point -- is the fix.
  * MULTIPATH. Object<->body double bounce, saved separately so training can use either.

Inherited from dataset 6 (ds6_m4v): matched-placement, jittered, 4-view regeneration of dataset5.

Merges the two half-finished dataset5 successors and adds the two fixes neither had:
  * dataset5_4view_driver.py (2026-08-18) -> 4 views + conformal-envelope recon
  * dataset5_driver.py       (2026-08-24) -> the lighter aperture taper

WHAT CHANGED vs dataset5 -- four fixes, see ds6_scope.html:

  1. MATCHED PLACEMENT   [fixes the LOCATION SHORTCUT]
     dataset5 put threats at all 36 sites but benign only at the 14 reviewer-approved ones, so
     22 sites were threat-only and a position-only classifier (bbox meta, NO pixels) scored
     0.989 AUC. Here EVERY object -- threat and benign alike -- is eligible at every site the
     view can see, so P(site | threat) == P(site | benign) exactly and site carries zero label
     information by construction. The reviewer plausibility map is kept per-scan as `plausible`
     so analysis can still report the realistic subset.
       DS7_PLACEMENT=cross   (default) full object x visible-site cross -- exactly matched
       DS7_PLACEMENT=sample  draw DS7_NPLACE per (body,view): site ~ U(visible), obj ~ U(17)

  2. PLACEMENT JITTER    [stops within-site memorisation]
     dataset5 froze all four seating DOF: sample_placement(rng=default_rng(0), jitter=False)
     (the site's area-centroid face, identical for every body and object), a fixed per-object
     roll, standoff pinned at 12 mm, flip always False. Here all four are drawn from a rng
     seeded per (seed0, body, view, idx, object, site) -- reproducible, resume-safe:
       barycentric  area-weighted face + uniform bary inside it
       roll         ROLL_OF[obj] + N(0, DS6_ROLL_JITTER_DEG)     default sigma = 20 deg
       standoff     U(DS6_GAP_LO, DS6_GAP_HI)                    default 6-20 mm (clothing)
       flip         Bernoulli(0.5)
     The actual drawn values are recorded (roll_deg, gap_m, flip) -- dataset5 could not.
     NOTE jitter alone does NOT fix the shortcut; fix 1 is the primary one.

  3. APERTURE TAPER      [matches the real system's resolution]
     dataset5 used taylor(sll=40) across the 130x130 aperture -> 5.3 cm cross-range FWHM @ 2 m
     against the real reflectarray's ~3.0 cm. Measured: rect 3.7 | t25 4.4 | t30 4.8 | t40 5.3.
     ds6 defaults to rect (closest to real). CRITICALLY, the aperture (cross-range) and
     frequency (range) tapers are now INDEPENDENT knobs -- dataset5's driver coupled them, so
     asking for a rect aperture silently also dropped the range taper. The frequency taper
     stays taylor sll=40 (6.25 mm range res, already Pareto-optimal vs hann).
       DS7_APER=rect|taylor|t25|hann   DS7_TAYLOR_SLL=40   DS7_FREQ_WINDOW=taylor|hann|none

  4. FOUR VIEWS          [front / back / left / right]
     Body rotated about the vertical axis vs the fixed aperture. Each site is imaged only from
     the view(s) that actually see it (see placements()), so there are no unlearnable
     fully-occluded far-side scans. DS7_ORIENTS=front,back,left,right

UNCHANGED from dataset5 (must stay comparable): 2.0 m standoff, +/-5 cm complex range gate,
1 cm cross-range grid, 5 mm range step, body-blocked delta forward, han_db definition, the 3D
save (field_gated shell + gate_cells lookup), the 17-object objects_v5 catalogue and its
materials, and the 200-body shapes_10x10 set.

STILL NOT INCLUDED: roughness + noise floor (plan sec.7 step 2). SPECULAR dataset.

All DS6_* knobs fall back to their DS5_* spelling, so existing launch scripts keep working.

  DS7_OUT=/path DS7_NBODIES=8 DS7_ORIENTS=front,back,left,right DS7_APER=rect \
    ~/miniconda3/envs/ra/bin/python dataset6_driver.py
"""
import datetime, hashlib, json, os, sys, time
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir("/Users/jonah"):
    os.environ.setdefault("SMPLX_MODELS", os.path.expanduser("~/smplx_models"))
    os.environ.setdefault("AMASS_ROOT", os.path.expanduser("~/amass"))
os.environ.setdefault("RA_PHASE_RECURSION", "1")
for p in (os.path.expanduser("~/walkthrough"),
          os.path.expanduser("~/Documents/code/model_gen/python"),
          "/Users/jonah/Documents/scratch_results/threat_dataset"):
    if os.path.isdir(p):
        sys.path.insert(0, p)
sys.path.insert(0, HERE)

from mesh_sample import sample_mesh                      # noqa: E402
from bodies_smplx import subdivide_front                 # noqa: E402
import anatomy, attach, threats, body as bodylib         # noqa: E402
import placement_config as pc                            # noqa: E402  (registers the 45 sites)
import raforward as rf                                   # noqa: E402
from raforward import Scene, C0                          # noqa: E402
from recon import ReconGrid, backproject, backproject_points  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame  # noqa: E402   the ONE statement of the coordinate convention
from bodies import zbuffer_cull                          # noqa: E402
from stance import screening_pose                        # noqa: E402

if not rf._HAS_CUPY:
    sys.exit("FATAL: no CuPy -- use ~/miniconda3/envs/ra/bin/python on a GPU host (bose/tesla).")

def _env(name, default=None):
    """DS7_<name>, falling back to DS6_ then DS5_ so old launch scripts keep working."""
    for pref in ("DS7_", "DS6_", "DS5_"):
        v = os.environ.get(f"{pref}{name}")
        if v is not None:
            return v
    return default

DS7_NAME = "pilot_fine_2026-10-01"
OUT = _env("OUT", os.path.expanduser(f"~/{DS7_NAME}"))
os.makedirs(OUT, exist_ok=True)
TD = "/Users/jonah/Documents/scratch_results/threat_dataset" if os.path.isdir(
    "/Users/jonah/Documents/scratch_results/threat_dataset") else HERE
# Require the MESHES, not just the directory: the repo tracks objects_v5/manifest.json but
# gitignores the .obj files, so a checkout leaves an objects_v5 dir that exists and is empty of
# meshes. Selecting it by isdir alone picks that one and dies later on trimesh.load.
V5 = next(p for p in (f"{HERE}/objects_v5", f"{TD}/objects_v5",
                      os.path.expanduser("~/probe/objects_v5"),
                      os.path.expanduser("~/walkthrough/objects_v5"),  # container layout
                      ) if os.path.isdir(p) and any(f.endswith(".obj") for f in os.listdir(p)))
GLOCK_STL = next(p for p in (f"{HERE}/glock17_ext.stl", f"{TD}/glock17_ext.stl",
                             os.path.expanduser("~/walkthrough/glock17_ext.stl"),
                             os.path.expanduser("~/threat_dataset/glock17_ext.stl"))
                 if os.path.exists(p))
PHONE = next(p for p in (os.path.join(os.path.dirname(GLOCK_STL), "phone_hull.stl"),
                         os.path.expanduser("~/walkthrough/phone_hull.stl"),
                         os.path.expanduser("~/threat_dataset/phone_hull.stl"))
             if os.path.exists(p))

LAM = C0 / 290e9
NFREQ = int(_env("NFREQ", "256"))
STANDOFF = float(_env("STANDOFF", "2.0"))
GATE_T = float(_env("GATE", "0.05"))
SIG_EVERY = int(_env("SIG_EVERY", "128"))
GAP = 0.012
# WORLD FRAME (see frame.py): +z into the scene with z=0 at the array, +y up, +x LEFT.
# The image plane is therefore (x, y) and RANGE is z -- the legacy driver had (x, z) and range y.
GSTEP = float(_env("GSTEP", "0.01"))       # image grid step (m); set <= PSF/2 = lam*R/(2D)
XS = np.arange(-0.90, 0.90 + GSTEP/2, GSTEP)   # x, cross-range, +ve to the array's LEFT
YS = np.arange(-1.05, 1.05 + GSTEP/2, GSTEP)   # y, vertical, +ve up
XS1 = np.arange(-0.90, 0.9001, 0.01); YS1 = np.arange(-1.05, 1.0501, 0.01)   # 1 cm, clutter metric only
DR = 0.005                                # range (z) bin
KGATE = int(round(2 * GATE_T / DR)) + 1

AP_D = float(_env("D", "0.067"))
PITCH = float(_env("PITCH", str(0.067 / 129)))
NEL = int(round(AP_D / PITCH)) + 1
g = np.linspace(-AP_D / 2, AP_D / 2, NEL)
ROI_W = float(_env("ROI", "0"))        # half-width of the per-site body crop (m); 0 = full body
ROI_T = 0.03                          # raised-cosine edge taper width (m)
xx, yy = np.meshgrid(g, g, indexing="ij")
RX = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(NEL * NEL)])   # array in the x-y plane, z=0
TX = np.array([[-0.15, 0.0, 0.0]])       # same physical spot; x flips sign in the world frame
FREQ = np.linspace(275e9, 305e9, NFREQ)
KV = 2 * np.pi * FREQ / C0
# ---- FIX 3: aperture (cross-range) and frequency (range) tapers, now INDEPENDENT ----------
# Two windows on two different axes. dataset5's driver conflated them behind one DS5_WINDOW, so
# selecting a rect aperture also silently set freq_window="none" and degraded RANGE sidelobes
# that were never the problem. They are separate knobs here.
#
#   APERTURE (DS6_APER) -> CROSS-RANGE. Measured FWHM @ 2 m, 6.7 cm sim aperture:
#       rect 3.7 cm  |  t25 4.4  |  t30 4.8  |  taylor sll40 5.3 (dataset5)  |  hann 6.2
#     The real ~7 cm reflectarray resolves ~3.0 cm, so sll40 was over-windowed by 1.8x.
#     ds6 DEFAULTS TO RECT -- the closest match to real. Cost is rect's -13 dB sidelobes
#     (the grid/cross artefacts), which the real untapered aperture also has.
#
#   FREQUENCY (DS7_FREQ_WINDOW) -> RANGE. Unchanged at taylor sll=40 (6.25 mm range res);
#     recon._windows shows it is Pareto-optimal vs hann (narrower AND 8.6 dB quieter), so
#     there is no reason to touch it. Must be a name backproject knows: taylor|hann|none.
APER_NAME = _env("APER", _env("WINDOW", "rect"))
FREQ_WINDOW = _env("FREQ_WINDOW", "taylor")
_SLL = float(_env("TAYLOR_SLL", "40"))
if APER_NAME in ("taylor", "t25", "t30"):
    from scipy.signal.windows import taylor as _taylor
    _sll = {"t25": 25.0, "t30": 30.0}.get(APER_NAME, _SLL)
    APER = _taylor(NEL, nbar=5, sll=_sll, norm=False)
elif APER_NAME == "hann":
    APER = np.hanning(NEL)
elif APER_NAME in ("rect", "none"):
    APER = np.ones(NEL)
else:
    sys.exit(f"DS6_APER must be one of rect|taylor|t25|t30|hann, got {APER_NAME!r}")
if FREQ_WINDOW not in ("taylor", "hann", "none"):
    sys.exit(f"DS7_FREQ_WINDOW must be taylor|hann|none, got {FREQ_WINDOW!r}")
# ARMS=up is the L3/AIT portal stance (upper arms level, forearms vertical, palms to the array)
# and is what the full run uses. "down" reproduces datasets 3-6 exactly, for comparison runs.
ARMS = _env("ARMS", "up")
POSE = screening_pose(ARMS)
S_SKIN = threats.MATERIALS["skin"][0]
SHAPES = json.load(open(_env("SHAPES", f"{HERE}/shapes_10x10.json")))
# DS7_BODY0 / DS6_NBODIES select a contiguous slab of bodies. One body per AWS Batch array task
# is the shard unit: the driver reuses the posed mesh across all four views, so sharding any
# finer would re-pay the ~96 s body prep per view.
_B0N = int(_env("BODY0", "0"))
_NB = int(_env("NBODIES", "0"))
SHAPES = SHAPES[_B0N:_B0N + _NB] if _NB else SHAPES[_B0N:]
if not SHAPES:
    sys.exit(f"no bodies selected: BODY0={_B0N} NBODIES={_NB}")

# ---------------------------------------------------------------- object catalogue ----------
# gun1 = the dataset4 two-material Glock (build_library does the steel/polymer split). Everything
# else = an objects_v5 mesh (already canonical + real-scale). An object that ships `_steel.obj` /
# `_polymer.obj` parts is built as a multi-material Threat; anything else stays single-material.
#
# Guns 2-5 gained parts because they are polymer-framed pistols (Glock 19, FMK 9C1, Grand Power
# K100) that were modelled as solid steel -- too bright, and missing the bright-slide-over-dim-frame
# structure a detector should key on. The Tanfoglio GT27 ships steel-only: it is cast zinc alloy
# throughout. Those part meshes are also INTERIOR-STRIPPED -- 57-65% of each raw mesh was bore,
# magwell and frame internals that no sensor can see, worth several dB of pure fiction.
OBJECTS = {"gun1_glock17": (threats.build_library(glock_stl=GLOCK_STL, phone_stl=PHONE)["pistol"], True)}
for name, o in pc.OBJECTS.items():
    if name == "gun1_glock17":
        continue
    _steel = f"{V5}/{name}_steel.obj"
    if os.path.exists(_steel):
        parts = [(trimesh.load(_steel, process=False), "steel")]
        _poly = f"{V5}/{name}_polymer.obj"
        if os.path.exists(_poly):
            parts.append((trimesh.load(_poly, process=False), "polymer"))
        thr = threats.Threat(name, o["kind"], o["material"], parts=parts, conformal=False)
    else:
        mesh = trimesh.load(f"{V5}/{name}.obj", process=False)
        thr = threats.Threat(name, o["kind"], o["material"], mesh=mesh, conformal=False)
    OBJECTS[name] = (thr, o["kind"] == "threat")
ROLL_OF = {name: float(o["roll_deg"]) for name, o in pc.OBJECTS.items()}
ROLL_OF["gun1_glock17"] = 0.0

# ---- guns-only / object allowlist (backward-compatible; no-op unless DS7_OBJECTS set) ----
_allow = _env("OBJECTS", "")
if _allow:
    keep = set(x for x in _allow.split(",") if x)
    missing = keep - set(OBJECTS)
    if missing:
        sys.exit(f"DS7_OBJECTS names not in catalogue: {sorted(missing)}")
    OBJECTS = {n: v for n, v in OBJECTS.items() if n in keep}
    if not OBJECTS:
        sys.exit("DS7_OBJECTS filtered everything")
    print(f"[DS7_OBJECTS] restricted to {sorted(OBJECTS)}", flush=True)

BACK_SITES = {"small_of_back", "upper_back", "wb_back_mid", "wb_back_l", "wb_back_r",
              "thigh_back_l", "thigh_back_r", "shin_back_l", "shin_back_r"}
def stance_of(site):
    return "back" if site in BACK_SITES else "front"

# View orientations = body rotated about the vertical axis vs the FIXED aperture (0=front,
# 180=back are the original dataset5; 90/-90 = left/right systems of the 4-system rig).
VIEW_DEG = {"front": 0.0, "back": 180.0, "left": 90.0, "right": -90.0}
ORIENTS = tuple(_env("ORIENTS", "front,back,left,right").split(","))
MAXPLACE = int(_env("MAXPLACE", "0"))   # >0 = cap placements/orientation (smoke test)
for _o in ORIENTS:
    if _o not in VIEW_DEG:
        sys.exit(f"DS6_ORIENTS entry {_o!r} not in {list(VIEW_DEG)}")

def _side_of(site):
    """left/right/mid classification from the site's _l / _r suffix (mid = central sites:
    belly, chest, small_of_back, upper_back, wb_front_mid, wb_back_mid)."""
    if site.endswith("_l"):
        return "left"
    if site.endswith("_r"):
        return "right"
    return "mid"

def _sees(orient, s):
    """Does a view actually SEE this site? Each site is imaged only from the view(s) that do,
    so there are NO fully-occluded far-side scans (a threat labelled 'threat' but invisible is
    unlearnable noise for a per-view detector):
      front / back : sites whose stance_of() matches (unchanged from the original dataset5)
      left         : left-side (_l) + central (mid) sites -- a left-mounted system's field of view
      right        : right-side (_r) + central (mid) sites
    Central sites (belly, chest, ...) are grazing-visible from either side, so they appear in
    both lateral views, as the real 4-system rig would capture them; _l and _r sites appear in
    exactly one. Total left+right ~= front+back, not double."""
    if orient in ("front", "back"):
        return stance_of(s) == orient
    if orient == "left":
        return _side_of(s) in ("left", "mid")
    if orient == "right":
        return _side_of(s) in ("right", "mid")
    return True

# ---- FIX 1: matched placement -------------------------------------------------------------
# dataset5 drew sites from pc.PLACEMENT, which gave threats all 36 sites but benign only the 14
# the confuser review approved. 22 sites were therefore threat-only and site location alone
# scored 0.989 AUC with no pixels at all. Here the site pool is the SAME for every object, so
# P(site | threat) == P(site | benign) exactly -- the shortcut is closed by construction, not by
# hoping a model ignores it.
#
# The reviewer map survives as PLAUSIBLE: every scan records whether its (object, site) pair was
# one a human called realistic, so analysis can report the full matched set AND the plausible
# subset. Yes, this seats a necklace at an ankle sometimes. That is the right trade --
# unrealistic-but-matched beats realistic-but-confounded, and the flag keeps both options open.
OBJ_NAMES = [n for n in OBJECTS]                       # gun1 first, then the objects_v5 order
SITES_OF_VIEW = {v: [s for s in pc.THREAT_SITES if _sees(v, s)] for v in VIEW_DEG}
PLAUSIBLE = {(n, s) for n, sites in pc.PLACEMENT.items() for s in sites}
_SITES = _env("SITES", "")
if _SITES:
    _keep = set(_SITES.split(","))
    SITES_OF_VIEW = {v: [s for s in ss if s in _keep] for v, ss in SITES_OF_VIEW.items()}
NOBASE = _env("NOBASE", "0") == "1"

PLACEMENT_MODE = _env("PLACEMENT", "cross")            # cross | sample
NPLACE = int(_env("NPLACE", "0"))                      # sample mode: placements per (body, view)
SEED0 = int(_env("SEED", "20260825"))
if PLACEMENT_MODE not in ("cross", "sample"):
    sys.exit(f"DS6_PLACEMENT must be cross|sample, got {PLACEMENT_MODE!r}")
if PLACEMENT_MODE == "sample" and NPLACE <= 0:
    sys.exit("DS7_PLACEMENT=sample needs DS7_NPLACE > 0")

# ---- FIX 2: placement jitter --------------------------------------------------------------
# dataset5 pinned all four seating DOF, so a given site produced a byte-identical seat frame for
# every body and every object. Each is now drawn from a per-placement rng. Secondary to fix 1:
# jitter stops exact memorisation of discrete positions, but it cannot on its own fix a site
# distribution that differs between the classes.
JITTER = _env("JITTER", "1") == "1"
VIEW_SEED = _env("VIEW_SEED", "0") == "1"   # FIX 6: 1 = legacy per-view placement draw
ROLL_JIT = float(_env("ROLL_JITTER_DEG", "20"))        # sigma, degrees, about the surface normal
GAP_LO = float(_env("GAP_LO", "0.006"))                # clothing standoff, metres
GAP_HI = float(_env("GAP_HI", "0.020"))

def _seed_of(*parts):
    """Stable 64-bit seed from a tuple of strings -- reproducible across hosts and resumes
    (Python's hash() is salted per process, so it cannot be used here)."""
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")

def placements(orient, bid):
    """[(obj_name, site)] for one (body, view) -- baseline first so it lands even if a run is
    cut short. Deterministic given (bid, orient, SEED0), so resume-skip stays exact.

    cross  : every object x every visible site. Exactly matched marginals, no sampling noise.
    sample : NPLACE draws with site ~ U(visible) and object ~ U(17), independently -- matched in
             expectation, and it decouples dataset size from the 17 x n_sites factorial."""
    sites = SITES_OF_VIEW[orient]
    out = [] if NOBASE else [("none", "nothreat")]
    if PLACEMENT_MODE == "cross":
        out += [(n, s) for n in OBJ_NAMES for s in sites]
    else:
        rng = np.random.default_rng(_seed_of(SEED0, bid, orient))
        out += [(OBJ_NAMES[i], sites[j])
                for i, j in zip(rng.integers(0, len(OBJ_NAMES), NPLACE),
                                rng.integers(0, len(sites), NPLACE))]
    return out


def bid_of(shp):
    return f"{shp['gender'][0]}_{shp['height']}_{shp['build']}"

def to_world(P, orient, y_center, is_dir=False):
    """SMPL-X -> world frame. The transform lives in frame.py so it is stated exactly once."""
    return frame.smplx_to_world(P, VIEW_DEG[orient], y_center, STANDOFF, is_dir=is_dir)

def fwd(q, s):
    sig = np.zeros((NFREQ, len(RX)), np.complex64)
    for c0 in range(0, len(q), 2_000_000):
        sig += rf.forward(Scene(q=q[c0:c0 + 2_000_000], s=s[c0:c0 + 2_000_000],
                                tx=TX, rx=RX, k=KV), precision="fp32")
    return sig

def mask_of(idx, n):
    m = np.zeros(n, bool); m[np.asarray(idx).ravel()] = True; return m

# ENVELOPE recon (DS7_ENVELOPE=1, default): reconstruct ONLY the +/-gate surface-band voxels
# via backproject_points (a verified-lossless subset of the dense cube) and scatter them into a
# zero cube of grid.shape. Downstream code (han_db max-projection, gate(), object/clutter metrics)
# is byte-for-byte unchanged -- the only difference from full-cube recon is that off-surface
# voxels are exactly 0 instead of the ~noise-floor they carry in the dense recon. Cuts the recon
# from ~10.8M voxels to ~1e5 (the band), the per-scan bottleneck. Set DS7_ENVELOPE=0 for the
# original dense recon (used to validate parity).
ENVELOPE = _env("ENVELOPE", "1") == "1"
# Object-scale self-occlusion pass. See the call site for why the body-scale one is not enough.
OBJ_ZBUF = _env("OBJ_ZBUF", "1") == "1"
OBJ_ZPIX = float(_env("OBJ_ZPIX", "0.001"))      # 1 mm: the object is ~10 cm, not ~1 m
OBJ_ZDEPTH = float(_env("OBJ_ZDEPTH", "0.004"))  # keep 4 mm behind the front surface.
# 4 mm, not 2 mm: a z-buffer at (keep_depth, pixel) implicitly rejects surface sloped beyond
# atan(keep_depth/pixel) -- 63 deg at 2 mm/1 mm. At grazing (the left/right views) ~77% of genuinely
# visible surface sits ABOVE 75 deg obliquity, so 2 mm removed 52-74% of real signal on badge, keys,
# knives and gun2, with spurious truncation glints. 4 mm raises the cutoff to 76 deg while still
# culling knife interiors, which sit >=4 mm back. Worst-case peak error vs ray-cast ground truth:
# old chain +8.9 dB, 2 mm -0.8, 4 mm +0.5, 8 mm +5.4.
_APER_W = np.outer(APER, APER).ravel().astype(np.float32)

def _to_np(a):
    try:
        return rf.cp.asnumpy(a)
    except Exception:
        return a

def recon_complex(sig, grid):
    return _to_np(backproject(sig * _APER_W[None, :], tx=TX, rx=RX, f=FREQ, grid=grid,
                              npad=4096, freq_window=FREQ_WINDOW, precision="fp32"))

# CONFORMAL 3D envelope: a thin shell that hugs the ACTUAL kept surface scatterers (body keep_b0
# and, per scan, the visible object gq[kt]) -- NOT the per-pixel nearest depth map. Root cause of the
# earlier han_db clipping: the band was hung off dmap = the *nearest* depth per cross-range cell, so
# at grazing (most of a side view) a 1 cm cell holding near-arm AND torso-behind (both kept by the
# front-2cm cull) only banded the nearest -> the deeper real return, and any protruding object, were
# clipped. Banding around EVERY kept scatterer fixes both at the source, needs no cross-range
# dilation guess, and stays object-safe for any protrusion. The band is in RANGE (y / line of sight,
# +/-ENV_KY bins), matching the range-gated reconstruction -- NOT along the surface normal, which at
# grazing points across the line of sight and would miss the range-smeared return -- plus a small
# +/-ENV_KXZ cross-range pad for the PSF. Thin (~1e5 voxels, conformal) vs full columns (~1e6) vs the
# dense cube (1.08e7). field_gated (the dmap +/-gate band) is a subset of this shell, so lossless.
ENV_KY = int(_env("ENV_KY", str(KGATE // 2)))    # +/- range bins (default +/-5 cm)
ENV_KXZ = int(_env("ENV_KXZ", "2"))             # +/- cross-range px (PSF pad, ~2 cm)

def _shell_flat(pts, grid, ky=None, kxz=None):
    """Flat grid-voxel indices within +/-ky range-bins (z) and +/-kxz cross-range px of pts."""
    ky = ENV_KY if ky is None else ky
    kxz = ENV_KXZ if kxz is None else kxz
    nx, ny, nz = grid.shape
    ix = np.clip(np.round((pts[:, 0]-grid.xs[0])/GSTEP).astype(np.int64), 0, nx-1)
    iy = np.clip(np.round((pts[:, 1]-grid.ys[0])/GSTEP).astype(np.int64), 0, ny-1)
    iz = np.clip(np.round((pts[:, 2]-grid.zs[0])/DR).astype(np.int64), 0, nz-1)
    cell = np.unique(ix*(ny*nz) + iy*nz + iz)                 # unique surface cells at image res
    cix, r = np.divmod(cell, ny*nz); ciy, ciz = np.divmod(r, nz)
    drange = np.arange(-ky, ky+1); dxy = np.arange(-kxz, kxz+1)   # +/-ky is RANGE (z) now
    DX, DYc, DZr = np.meshgrid(dxy, dxy, drange, indexing="ij")
    ox, oy, oz = DX.ravel(), DYc.ravel(), DZr.ravel()
    ex = np.clip(cix[:, None]+ox[None, :], 0, nx-1)
    ey = np.clip(ciy[:, None]+oy[None, :], 0, ny-1)
    ez = np.clip(ciz[:, None]+oz[None, :], 0, nz-1)
    return np.unique((ex*(ny*nz) + ey*nz + ez).ravel())

def env_from_flat(flat, grid):
    ny, nz = grid.shape[1], grid.shape[2]
    ix, r = np.divmod(flat, ny*nz); iy, iz = np.divmod(r, nz)
    ix, iy, iz = ix.astype(np.int32), iy.astype(np.int32), iz.astype(np.int32)
    pts = np.column_stack([grid.xs[ix], grid.ys[iy], grid.zs[iz]]).astype(float)
    return dict(ix=ix, iy=iy, iz=iz, points=np.ascontiguousarray(pts),
                shape=grid.shape, flat=flat)

def build_envelope(grid, body_pts):
    """Conformal shell for the body's kept scatterers (cached per orientation)."""
    return env_from_flat(_shell_flat(body_pts, grid), grid)

def env_with_object(env_body, grid, obj_pts):
    """Body shell UNION the object's own conformal shell -- per scan, so a protruding object is
    reconstructed at its true footprint (no dilation guess, no clipping)."""
    flat = np.union1d(env_body["flat"], _shell_flat(obj_pts, grid))
    return env_from_flat(flat, grid)

def recon_envelope(sig, env):
    vals = _to_np(backproject_points(sig * _APER_W[None, :], tx=TX, rx=RX, f=FREQ,
                                     points=env["points"], npad=4096,
                                     freq_window=FREQ_WINDOW, precision="fp32"))
    field = np.zeros(env["shape"], np.complex64)
    field[env["ix"], env["iy"], env["iz"]] = vals
    return field

def obj_clutter_lin(sig, grid, mag_global, gc, gk):
    """(obj_lin, clut): 90th-pct |field| at the object voxels, and over the r in (0.08, 0.25) m
    annulus around the object centre. The annulus samples OFF-surface voxels the envelope leaves
    at zero, so in ENVELOPE mode recompute both from a small DENSE local cube spanning the annulus
    -- built from subsets of the same XS/YS/zs coords, so its voxel values are byte-identical to
    the full dense recon there (per-voxel backprojection is independent of grid extent). This also
    replaces the original's full 10.8M-voxel meshgrid, done once per scan."""
    if ENVELOPE:
        lxs = XS1[(XS1 >= gc[0]-0.27) & (XS1 <= gc[0]+0.27)]
        lys = YS1[(YS1 >= gc[1]-0.27) & (YS1 <= gc[1]+0.27)]
        lzs = grid.zs[(grid.zs >= gc[2]-0.27) & (grid.zs <= gc[2]+0.27)]
        lmag = np.abs(recon_complex(sig, ReconGrid(xs=lxs, ys=lys, zs=lzs)))
        gxs, gys, gzs, xs0, ys0, zs0 = lxs, lys, lzs, lxs[0], lys[0], lzs[0]
    else:
        lmag = mag_global
        gxs, gys, gzs, xs0, ys0, zs0 = grid.xs, grid.ys, grid.zs, XS[0], YS[0], grid.zs[0]
    ox = np.clip(np.round((gk[:, 0]-xs0)/0.01).astype(np.int32), 0, len(gxs)-1)
    oy = np.clip(np.round((gk[:, 1]-ys0)/0.01).astype(np.int32), 0, len(gys)-1)
    oz = np.clip(np.round((gk[:, 2]-zs0)/DR).astype(np.int32), 0, len(gzs)-1)
    obj_lin = float(np.percentile(lmag[ox, oy, oz], 90))
    X, Y, Z = np.meshgrid(gxs, gys, gzs, indexing="ij")
    r = np.sqrt((X-gc[0])**2 + (Y-gc[1])**2 + (Z-gc[2])**2)
    ann = (r > 0.08) & (r < 0.25)
    clut = float(np.percentile(lmag[ann], 90)) if ann.any() else np.nan
    return obj_lin, clut

def depth_map_of(q_vis):
    ix = np.clip(np.round((q_vis[:, 0] - XS[0]) / GSTEP).astype(np.int32), 0, len(XS) - 1)
    iy = np.clip(np.round((q_vis[:, 1] - YS[0]) / GSTEP).astype(np.int32), 0, len(YS) - 1)
    dm = np.full((len(XS), len(YS)), np.inf)
    np.minimum.at(dm, (ix, iy), q_vis[:, 2])          # depth is z; min = nearest the array
    dm[np.isinf(dm)] = np.nan
    return dm.astype(np.float32)

def gate(field, grid, dmap):
    """Range-gate the cube about the per-cell surface depth -> (n_cells, KGATE) complex64 +
    gate_cells lookup (flat (ix,iy) index; iz implicit from dmap + zs[0] + DR). Lossless subset."""
    cells = np.flatnonzero(np.isfinite(dmap.ravel()))
    ix, iy = np.unravel_index(cells, dmap.shape)
    iz0 = np.round((dmap.ravel()[cells] - grid.zs[0]) / DR).astype(np.int32)
    off = np.arange(KGATE, dtype=np.int32) - KGATE // 2
    iz = np.clip(iz0[:, None] + off[None, :], 0, len(grid.zs) - 1)
    return field[ix[:, None], iy[:, None], iz].astype(np.complex64), cells.astype(np.int32)


_B0 = bid_of(SHAPES[0])                       # representative: counts are body-independent
_PLAN = {st: placements(st, _B0) for st in ORIENTS}
if MAXPLACE:
    _PLAN = {st: p[:MAXPLACE] for st, p in _PLAN.items()}
N_PER_BODY = sum(len(p) for p in _PLAN.values())
N_THREAT = sum(1 for p in _PLAN.values() for o, s in p if o != "none" and OBJECTS[o][1])
print(f"[{datetime.datetime.now():%H:%M}] DATASET7 ({DS7_NAME}): {len(SHAPES)} bodies x "
      f"{N_PER_BODY} placements = {len(SHAPES)*N_PER_BODY} scans "
      f"({len(SHAPES)*N_THREAT} threat / {len(SHAPES)*(N_PER_BODY-N_THREAT)} no-threat)", flush=True)
print(f"  objects: " + ", ".join(f"{n}{'*' if t else ''}" for n, (_, t) in OBJECTS.items())
      + "   (* = threat)", flush=True)
print(f"  views: " + "  ".join(f"{st}={len(p)}({len(SITES_OF_VIEW[st])} sites)"
                               for st, p in _PLAN.items()), flush=True)
print(f"  placement={PLACEMENT_MODE}"
      + (f" nplace={NPLACE}" if PLACEMENT_MODE == "sample" else "")
      + f"  MATCHED (every object at every visible site)  seed={SEED0}", flush=True)
print(f"  jitter={'ON' if JITTER else 'OFF'}"
      + (f" (bary + roll sigma {ROLL_JIT:g} deg + gap U({GAP_LO*1000:g},{GAP_HI*1000:g}) mm"
         f" + flip 50/50)" if JITTER else " -- frozen centroid seating, as dataset5"), flush=True)
print(f"  standoff={STANDOFF} m  aper={APER_NAME} freq_window={FREQ_WINDOW}  "
      f"gate=+/-{GATE_T*100:.0f} cm (K={KGATE})  envelope={ENVELOPE}  "
      f"sig every {SIG_EVERY}  GPU={rf._HAS_CUPY} -> {OUT}", flush=True)

def _taper(d):
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
      f"ROI={'off' if ROI_W <= 0 else f'+/-{ROI_W*100:.0f} cm'}", flush=True)
def _plan_of(bid):
    """{view: [(idx, obj, site)]} for one body. The index is part of the filename because
    sample mode can legitimately draw the same (object, site) twice for a body/view -- without
    it those two scans would collide and the second would silently overwrite the first."""
    out = {}
    for st in ORIENTS:
        p = placements(st, bid)
        if MAXPLACE:
            p = p[:MAXPLACE]
        out[st] = [(i, o, s) for i, (o, s) in enumerate(p)]
    return out

for bi, shp in enumerate(SHAPES):
    bid = bid_of(shp)
    plan = _plan_of(bid)
    if all(os.path.exists(f"{OUT}/pilot_D{int(round(AP_D*1000))}_{bid}_{st}_{i:04d}_{o}_{s}.npz")
           for st in ORIENTS for i, o, s in plan[st]):
        print(f"  skip {bid} (done)", flush=True); nscan += N_PER_BODY; continue

    t0 = time.time()
    betas = np.array(shp["betas"])
    model = bodylib.get_model(shp["gender"])
    VC = bodylib.canonical_verts(model)
    Vp, F = bodylib.posed_verts(model, betas=betas, body_pose=POSE)
    m = trimesh.Trimesh(vertices=Vp, faces=F, process=False); m.fix_normals()
    yc = 0.5 * (Vp[:, 1].min() + Vp[:, 1].max())   # SMPL-X y is up; vertical centre
    Vs, Fs, _ = subdivide_front(m, 0.5 * LAM)
    bq_raw, _, _ = sample_mesh(Vs, Fs, 0.125 * LAM, front_cull=False)
    el = (time.time() - t_start) / 60
    print(f"[{datetime.datetime.now():%H:%M}] BODY {bi+1}/{len(SHAPES)} {bid}: "
          f"{len(bq_raw):,} pts ({time.time()-t0:.0f}s)  [{el:.0f} min elapsed]", flush=True)

    for stance in ORIENTS:
        t0 = time.time()
        bq = to_world(bq_raw.astype(np.float64), stance, yc)
        sb = np.full(len(bq), S_SKIN)
        keep_b0 = mask_of(zbuffer_cull(bq, pixel=0.003, depth_axis=2, keep_depth=0.02), len(bq))
        sig_body = fwd(bq[keep_b0], sb[keep_b0]) if ROI_W <= 0 else None
        _ctx = {}

        zlo, zhi = bq[keep_b0][:, 2].min(), bq[keep_b0][:, 2].max()   # range extent
        grid = ReconGrid(xs=XS, ys=YS, zs=np.arange(zlo - 0.05, zhi + 0.05, DR))
        dmap = depth_map_of(bq[keep_b0])
        env = build_envelope(grid, bq[keep_b0]) if (ENVELOPE and ROI_W <= 0) else None   # conformal body shell (cached)
        keep_full, sb_full = keep_b0, sb
        field_base = None   # body-only recon for this orientation, reused for fully-occluded objects
        common = dict(body=bid, stance=stance, view=stance, view_deg=VIEW_DEG[stance],
                      standoff_m=STANDOFF, aper=APER_NAME, freq_window=FREQ_WINDOW,
                      window=APER_NAME,          # legacy alias: dataset5 readers expect `window`
                      jittered=JITTER, placement_mode=PLACEMENT_MODE, seed=SEED0,
                      arms=ARMS,               # "up" = L3 portal stance; "down" = datasets 3-6
                      obj_zbuf=OBJ_ZBUF,       # object-scale self-occlusion cull
                      obj_zdepth=OBJ_ZDEPTH,
                      view_seeded=VIEW_SEED,   # FIX 6: False = the 4 views share one placement
                      gate_t=GATE_T, kgate=KGATE, **shp,
                      xs=XS.astype(np.float32), ys=YS.astype(np.float32),
                      zs=grid.zs.astype(np.float32), depth_map=dmap, gap_nominal_m=GAP,
                      frame="world_z_range_y_up_x_left",
                      ext=np.array([XS[0]*100, XS[-1]*100, YS[0]*100, YS[-1]*100]))
        print(f"  {bid}/{stance}: {int(keep_b0.sum()):,} visible, grid {grid.shape}, "
              f"{time.time()-t0:.0f}s", flush=True)

        for pidx, obj_name, site in plan[stance]:
            t1 = time.time()
            f_out = f"{OUT}/pilot_D{int(round(AP_D*1000))}_{bid}_{stance}_{pidx:04d}_{obj_name}_{site}.npz"
            if os.path.exists(f_out):
                nscan += 1; continue
            if ROI_W > 0:
                if site not in _ctx:
                    _ctx[site] = site_ctx(site, model, Vp, F, stance, yc, bq, sb_full, keep_full)
                cx = _ctx[site]
                keep_b0, sb, sig_body, grid, dmap, env = cx["roi"], cx["sbw"], cx["sig_body"], cx["grid"], cx["dmap"], cx["env"]
                field_base = None

            if obj_name == "none":
                sig, ng, n_seated, pl, is_threat = sig_body, 0, 0, None, False
                gq = kt = gn = None
                roll = gap = np.nan
            else:
                obj, is_threat = OBJECTS[obj_name]
                # FIX 2 -- one rng per placement, seeded from the identity of the placement, so
                # the draw is identical on any host and survives a resume mid-body.
                # FIX 6: the placement RNG must NOT depend on the view. With `stance` in the seed
                # each view re-drew bary/roll/gap/flip, so the four views were four DIFFERENT SCENES
                # rather than four aspects of one -- measured on a single (body, obj, site): roll
                # -4.33 / -14.82 / +53.63 deg and gap 6.5 / 11.8 / 19.1 mm across three views. That
                # blocks multi-view fusion and aspect-dependence studies, which is much of why four
                # views exist. Dropping `stance` gives ONE physical placement seen four ways.
                # DS6_VIEW_SEED=1 restores the per-view draw, to reproduce the 2026-08-25 pilot.
                _key = ((SEED0, bid, stance, pidx, obj_name, site) if VIEW_SEED
                        else (SEED0, bid, pidx, obj_name, site))
                prng = np.random.default_rng(_seed_of(*_key))
                pl = anatomy.sample_placement(model, site, prng, jitter=JITTER)
                # Draw roll here rather than via attach()'s roll_jitter so the ACTUAL value is
                # known and recorded -- attach() would apply it internally and not report it.
                roll = ROLL_OF[obj_name] + (prng.normal(0, ROLL_JIT) if JITTER else 0.0)
                gap = float(prng.uniform(GAP_LO, GAP_HI)) if JITTER else GAP
                pl.update(roll=roll, standoff=gap,
                          flip=bool(prng.random() < 0.5) if JITTER else False)
                gq, gn, gs = attach.attach(obj, Vp, VC, F, pl, target_nn=0.125 * LAM,
                                           vert_normals=m.vertex_normals)
                gq = to_world(gq, stance, yc)
                gn = to_world(gn, stance, yc, is_dir=True)
                n_seated = len(gq)
                gq, gn, gs = attach.front_cull(gq, gn, gs, view_dir=(0.0, 0.0, -1.0))
                q_all = np.vstack([bq, gq])
                keep = mask_of(zbuffer_cull(q_all, pixel=0.003, depth_axis=2, keep_depth=0.02), len(q_all))
                kb, kt = keep[:len(bq)], keep[len(bq):]
                # The joint pass above uses keep_depth=2 cm, a depth tuned for a ~30 cm-deep BODY.
                # A pistol is 3 cm thick, so that window spans the whole object and keeps its
                # interior -- bore, magwell, frame walls -- which then radiate. Measured on the
                # knives, whose meshes are 27-34% interior, that is ~11 dB of pure fiction. So run a
                # second pass at OBJECT scale over the object's own points, dropping surface hidden
                # behind the object itself. Body handling is untouched.
                if OBJ_ZBUF and kt.any():
                    _i = np.flatnonzero(kt)
                    _k2 = mask_of(zbuffer_cull(np.ascontiguousarray(gq[_i]), pixel=OBJ_ZPIX,
                                               depth_axis=2, keep_depth=OBJ_ZDEPTH), len(_i))
                    kt = np.zeros_like(kt); kt[_i[_k2]] = True
                ng = int(kt.sum())
                shadow = keep_b0 & ~kb
                sig = sig_body.copy()
                if shadow.any():
                    sig -= fwd(bq[shadow], sb[shadow])
                if ng:
                    sig += fwd(gq[kt], gs[kt])

            # Envelope recon (+ reuse the body-only field for fully-occluded objects, whose
            # signal is exactly sig_body). Falls back to the dense cube if DS7_ENVELOPE=0.
            if ENVELOPE:
                if obj_name != "none" and ng == 0 and not shadow.any():
                    if field_base is None:
                        field_base = recon_envelope(sig_body, env)
                    field = field_base
                elif obj_name == "none":
                    field = recon_envelope(sig, env)
                    field_base = field
                else:                                    # visible object: add its conformal shell
                    field = recon_envelope(sig, env_with_object(env, grid, gq[kt]))
            else:
                field = recon_complex(sig, grid)
            mag = np.abs(field)
            pk = float(mag.max())
            db = (20*np.log10(np.maximum(mag.max(2), pk*1e-4)/pk)).astype(np.float32)
            fg, cells = gate(field, grid, dmap)

            rec = dict(obj_name=obj_name, is_threat=bool(is_threat), site=site,
                       han_db=db, han_peak=pk, field_gated=fg, gate_cells=cells,
                       n_seated=n_seated, n_visible=ng,
                       # FIX 2 provenance: the drawn seating DOF, which dataset5 could not record
                       # because they were constants. roll_deg/gap_m are NaN on the baseline scan.
                       placement_idx=pidx, roll_deg=roll, gap_m=gap,
                       # FIX 1 provenance: was this (object, site) one the confuser review called
                       # realistic? Lets analysis split the matched set from the plausible subset.
                       plausible=bool((obj_name, site) in PLAUSIBLE),
                       # FIX 5: set True below when the object contributes no visible points.
                       zero_vis=False,
                       **common)
            rec.update(grid_step_m=GSTEP, aper_D_m=AP_D, pitch_m=PITCH, n_el=NEL, nfreq=NFREQ, roi_w_m=ROI_W,
                       zs=grid.zs.astype(np.float32), depth_map=dmap,
                       roi_centre=(_ctx[site]["centre"] if ROI_W > 0 else np.full(3, np.nan)))
            if ng:
                gk = gq[kt]
                gx = np.clip(np.round((gk[:,0]-XS[0])/GSTEP).astype(np.int32), 0, len(XS)-1)
                gy = np.clip(np.round((gk[:,1]-YS[0])/GSTEP).astype(np.int32), 0, len(YS)-1)
                gz = np.clip(np.round((gk[:,2]-grid.zs[0])/DR).astype(np.int32), 0, len(grid.zs)-1)
                gc = gk.mean(0)
                obj_lin, clut = obj_clutter_lin(sig, grid, mag, gc, gk)
                obj_vox = np.unique(np.column_stack([gx, gy, gz]), axis=0).astype(np.int16)
                obj_aabb = np.vstack([gk.min(0), gk.max(0)])
                rec.update(face=pl["face"], bary=np.array(pl["bary"]), flip=bool(pl["flip"]),
                           seat_xyz=gc, obj_aabb=obj_aabb, obj_vox=obj_vox,
                           visibility=ng/n_seated,
                           obliquity_deg=float(np.degrees(np.arccos(
                               np.clip(np.abs(-gn[kt][:,2]).mean(), 0, 1)))),
                           han_obj_db=20*np.log10(max(obj_lin, pk*1e-5)/pk),
                           han_clutter=clut, han_contrast_db=20*np.log10(obj_lin/(clut+1e-30)))
                rec.update(gun_vox=obj_vox if is_threat else np.zeros((0, 3), np.int16),
                           gun_aabb=obj_aabb if is_threat else np.full((2, 3), np.nan),
                           han_gun_db=rec["han_obj_db"] if is_threat else np.nan)
            else:
                # FIX 5: an object contributing NO visible points leaves nothing in the image --
                # no voxels, no contrast, nothing to learn. Calling that a threat is label noise.
                # The scan is a truthful image of a body with nothing showing, so no-threat is the
                # correct label and it becomes a hard negative. Keep the original label and the
                # reason so the choice stays auditable and reversible. The baseline (obj_name
                # "none") also lands here, but it is already no-threat and carries no object.
                if obj_name != "none":
                    rec.update(zero_vis=True, is_threat_orig=bool(is_threat),
                               relabel_reason="zero_visibility", is_threat=False)
                rec.update(seat_xyz=np.full(3, np.nan), obj_vox=np.zeros((0,3), np.int16),
                           obj_aabb=np.full((2,3), np.nan), visibility=np.nan,
                           obliquity_deg=np.nan, han_obj_db=np.nan, han_clutter=np.nan,
                           han_contrast_db=np.nan, gun_vox=np.zeros((0,3), np.int16),
                           gun_aabb=np.full((2,3), np.nan), han_gun_db=np.nan)
            if nscan % SIG_EVERY == 0:
                rec["sig"] = sig.astype(np.complex64)
            np.savez_compressed(f_out, **rec)
            nscan += 1
            print(f"    {bid}/{stance}/{obj_name:<16}/{site:<16} "
                  f"{('THREAT' if is_threat else 'benign') if ng else 'novis ':<6} vis "
                  f"{rec['visibility'] if ng else 0:.3f} contrast "
                  f"{rec['han_contrast_db']:+6.1f} dB  ({time.time()-t1:.0f}s)"
                  + ("  +sig" if "sig" in rec else ""), flush=True)
        del bq, sb, sig_body
    del bq_raw, Vp, F, m, Vs, Fs

print(f"[{datetime.datetime.now():%H:%M}] DATASET7 done: {nscan} scans in "
      f"{(time.time()-t_start)/60:.0f} min -> {OUT}", flush=True)

"""Threat + benign-clutter object library.

Every object is returned in a CANONICAL OBJECT FRAME:
    +x = long axis      (barrel / blade / pipe length)
    +y = mid axis       (width)
    +z = thin axis      (pressed against the body; +z points AWAY from the body)
centered on its centroid. attach.py maps that frame onto the body surface, so
every object seats the same way and `roll=0` always means "long axis along the body".

Objects carry a material: `s` (isotropic scalar reflectivity, the forward model's
only material knob) and `rough` (surface roughness sigma in wavelengths, used to
displace points along their normals -- see materials.py for why this matters).

Threat geometry here is PARAMETRIC (trimesh primitives), not scanned CAD, on
purpose: it lets us sweep the confusion axis (a phone and a sheet-explosive differ
only in thickness and reflectivity) and control size/aspect directly. Real CAD from
the Target_Library drops in through the same `sample_mesh` path when available.
"""
from __future__ import annotations
import numpy as np
import trimesh

C0 = 299_792_458.0
LAM = C0 / 290e9                       # 1.034 mm at 290 GHz

# Material presets: (s, roughness_sigma_in_lambda).
# s is the scalar reflectivity used by the forward model; metals ~1, dielectrics
# lower. `rough` is the surface-roughness sigma -- a smooth mirror is ~0.
MATERIALS = {
    "steel":     (1.00, 0.02),   # machined metal: smooth, bright
    "alloy":     (0.95, 0.02),
    # Glock frame (Nylon 6). EFFECTIVE reflectivity, not the bare Fresnel value -- see below.
    "polymer":   (0.35, 0.05),
    "ceramic":   (0.35, 0.05),   # ceramic blade: dielectric, LOW return -- the hard threat
    "plastic":   (0.30, 0.05),
    "explosive": (0.45, 0.08),   # PE4/C4-like dielectric slab
    "skin":      (0.60, 0.30),   # body roughness is still the big unknown -- see DESIGN.md
    "fabric":    (0.25, 0.50),
}

# WHY polymer s = 0.35 and not the bare Fresnel 0.273
# ---------------------------------------------------------------------------------
# THz PENETRATES dielectrics, so a Glock's polymer frame is semi-transparent and the
# STEEL MAGAZINE inside the grip contributes. But that internal metal cannot be
# separated from the surface, so it does not need a two-layer model -- it folds into
# an EFFECTIVE surface reflectivity:
#
#   * polyamide at 300 GHz: n = 1.75, extinction k = 0.047 (constant 0.1-1.0 THz)
#     => alpha = 4*pi*k/lambda0 ~= 5.9 cm^-1 (intensity), e-folding depth ~1.7 mm.
#     NB k constant => alpha ~ f LINEARLY, not f^2.
#   * grip wall t ~ 2.5 mm. Round-trip AMPLITUDE factor is exp(-alpha*t) -- the round
#     trip (x2) and the power->amplitude (/2) cancel; easy to get wrong.
#       steel behind the wall : exp(-5.9*0.25) * 0.926 (Fresnel, 2 interfaces) = 0.212
#       polymer surface       : (n-1)/(n+1)                                    = 0.273
#       incoherent sum        : sqrt(0.273^2 + 0.212^2)                        = 0.35
#   * and it is UNRESOLVABLE: 2.5 mm at n=1.75 is 4.4 mm of optical depth, versus a
#     range resolution of c/2B = 5 mm. Surface and buried metal land in the SAME range
#     cell, so there is no layered signature to capture -- only a brighter grip.
#
# Limits of validity: this loses the Fabry-Perot ripple (a 2.5 mm gap puts the free
# spectral range INSIDE our 30 GHz band, so it would not average out), and it BREAKS
# for thick dielectrics -- metal behind ~1-2 cm of plastic (a gun in a hard case) WOULD
# resolve in depth and would need a real two-layer model.
# Perspective: this is a ~2 dB correction. The body-roughness gap (our body behaves
# like a bare mannequin, not a clothed person -- see DESIGN.md) is worth far more.

# Glock 17: which connected components are steel vs polymer. The mesh carries NO
# material names (ZBrush auto-groups, no usemtl), but the parts ARE separate solids,
# so we classify by height: the slide+sights ride on top of the frame. `y` is the
# canonical mid axis (see _canon), and the split sits at the slide's underside.
GLOCK_STEEL_ABOVE = 0.015        # metres, centroid height above which a part is steel


def _canon(mesh_or_pts):
    """Center, and orient principal axes -> (long=x, mid=y, thin=z) with a
    deterministic sign convention (no SVD sign flipping between runs)."""
    pts = np.asarray(mesh_or_pts, np.float64)
    pts = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    R = Vt                                     # rows = principal axes, descending extent
    # deterministic signs: make the largest-|component| entry of each axis positive
    for i in range(3):
        if R[i, np.argmax(np.abs(R[i]))] < 0:
            R[i] *= -1
    if np.linalg.det(R) < 0:                   # keep it a rotation
        R[2] *= -1
    return pts @ R.T                           # long->x, mid->y, thin->z


def _glock_parts(glock_stl):
    """Split the Glock into steel vs polymer parts, AND re-canonicalize on the BARREL.

    Materials: the mesh has no material names (ZBrush polygroups, no usemtl), but the
    gun IS built from separate solids -- 967 connected components. The steel slide and
    sights ride ON TOP of the polymer frame, so a height split on the component
    centroid separates them. Grip stippling arrives as ~950 tiny components on the
    frame and correctly falls out as polymer.

    Frame: do NOT use PCA of the whole gun to define the long axis. A pistol's grip
    hangs down and back, which drags the principal axis ~35 deg off the barrel -- so
    `roll=0` would mean "along an arbitrary statistical axis", the gun would sit canted
    at every site, and a roll sweep would sample nothing interpretable. Instead take the
    long axis of the SLIDE, which IS the barrel. Then:
        +x = barrel (muzzle direction)
        +z = slide normal (pressed against the body)
    and roll=0 means barrel-up, roll=180 barrel-down -- the realistic waistband carry.
    """
    m = trimesh.load(glock_stl, process=True)
    comps = m.split(only_watertight=False)
    steel, poly = [], []
    for c in comps:
        (steel if c.vertices.mean(0)[1] > GLOCK_STEEL_ABOVE else poly).append(c)
    slide = trimesh.util.concatenate(steel)
    frame = trimesh.util.concatenate(poly)

    # barrel axis = the slide's own principal axis; thin axis = the gun's flat axis
    S = np.asarray(slide.vertices, np.float64)
    _, _, Vs = np.linalg.svd(S - S.mean(0), full_matrices=False)
    barrel = Vs[0]
    A = np.vstack([np.asarray(slide.vertices), np.asarray(frame.vertices)])
    _, _, Va = np.linalg.svd(A - A.mean(0), full_matrices=False)
    thin = Va[2]                                   # flat axis of the whole pistol
    thin -= (thin @ barrel) * barrel
    thin /= np.linalg.norm(thin)
    mid = np.cross(thin, barrel)
    R = np.vstack([barrel, mid, thin])             # rows = new axes
    if np.linalg.det(R) < 0:
        R[1] *= -1
    ctr = A.mean(0)

    out = []
    for mesh, mat in [(slide, "steel"), (frame, "polymer")]:
        mm = mesh.copy()
        mm.vertices = (np.asarray(mm.vertices, np.float64) - ctr) @ R.T
        out.append((mm, mat))
    return out


class Threat:
    """An object to conceal.

    Either a single mesh with one material, or `parts` = [(mesh, material), ...] for
    a heterogeneous object. The Glock is the motivating case: a steel slide on a
    POLYMER frame is ~11.7 dB of internal contrast, and that "bright bar over a dim
    body" structure is part of what makes a pistol look like a pistol rather than a
    generic metal lump -- exactly what we want a detector to key on.

    Sampling every part at the same target_nn keeps one uniform point density across
    the whole object, so `s` stays pure material (see DESIGN.md).
    """

    def __init__(self, name, kind, material, *, mesh=None, parts=None,
                 conformal=False, notes=""):
        self.name, self.kind, self.material = name, kind, material
        self.mesh = mesh
        self.parts = parts                     # [(trimesh, material_name), ...]
        self.conformal = conformal             # True -> bends onto the body surface
        self.notes = notes
        self.s, self.rough = MATERIALS[material]

    def sample(self, target_nn):
        """-> (pts (M,3), normals (M,3), s (M,)) in the canonical object frame."""
        from mesh_sample import sample_mesh
        chunks = self.parts if self.parts else [(self.mesh, self.material)]
        Q, N, S = [], [], []
        for mesh, mat in chunks:
            V = np.asarray(mesh.vertices, np.float64)
            F = np.asarray(mesh.faces, np.int64)
            q, n, _ = sample_mesh(V, F, target_nn, front_cull=False)
            Q.append(q.astype(np.float64))
            N.append(n.astype(np.float64))
            S.append(np.full(len(q), MATERIALS[mat][0]))
        return np.vstack(Q), np.vstack(N), np.concatenate(S)

    def _verts(self):
        if self.parts:
            return np.vstack([np.asarray(m.vertices) for m, _ in self.parts])
        return np.asarray(self.mesh.vertices)

    def extent(self):
        return np.ptp(self._verts(), axis=0)

    def half_thickness(self):
        return 0.5 * self.extent()[2]

    def materials(self):
        if self.parts:
            return sorted({m for _, m in self.parts})
        return [self.material]

    def __repr__(self):
        e = self.extent() * 100
        return (f"<Threat {self.name} [{self.kind}] {self.material} s={self.s} "
                f"{e[0]:.0f}x{e[1]:.0f}x{e[2]:.1f}cm>")


# ---------------------------------------------------------------- constructors

def _slab(L, W, T, bevel=0.0):
    """Rounded rectangular slab, long axis +x, thin axis +z."""
    b = trimesh.creation.box(extents=[L, W, T])
    if bevel:
        b = b.subdivide()
    return b


def _tapered_blade(L, W, T):
    """Knife: tapered flat blade + a thinner handle. Long axis +x."""
    # blade: wedge tapering to a point in +x, thinning in z toward the edge
    x = np.array([-L / 2, -L / 6, L / 2])
    w = np.array([W * 0.55, W, 0.02 * W])          # handle, ricasso, tip
    t = np.array([T, T * 0.9, T * 0.15])
    V, F = [], []
    for i, (xi, wi, ti) in enumerate(zip(x, w, t)):
        V += [[xi, -wi / 2, -ti / 2], [xi, wi / 2, -ti / 2],
              [xi, wi / 2, ti / 2], [xi, -wi / 2, ti / 2]]
    V = np.array(V, np.float64)
    for k in range(2):                              # loft the 2 sections
        a = 4 * k
        b = 4 * (k + 1)
        for j in range(4):
            j2 = (j + 1) % 4
            F += [[a + j, a + j2, b + j2], [a + j, b + j2, b + j]]
    F += [[0, 1, 2], [0, 2, 3]]                     # cap the butt
    m = trimesh.Trimesh(vertices=V, faces=np.array(F, np.int64), process=True)
    m.fix_normals()
    return m


def _pipe(L, D, wall=0.004):
    """Capped steel pipe, long axis +x."""
    c = trimesh.creation.cylinder(radius=D / 2, height=L, sections=48)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return c


def _cyl(L, D, axis="x", sections=24):
    c = trimesh.creation.cylinder(radius=D / 2, height=L, sections=sections)
    if axis == "x":
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return c


def build_library(glock_stl="glock17_ext.stl", phone_stl="phone_hull.stl"):
    """The object catalogue. THREAT + BENIGN (benign = the false-positive drivers).

    EVERY object is a MESH, sampled through the same `mesh_sample.sample_mesh` as the
    body. That is load-bearing: with isotropic scatterers the recon brightness of a
    resolution cell goes as the number of points in it, so ONE uniform density
    everywhere is what makes `s` mean "material" and nothing else.

    RETIRED ASSETS -- do not use, they silently corrupt brightness:
      * gun.xyz   -- 7.6% EXACT duplicate points. The forward model sums coherently,
                     so N coincident points give N^2 power: a doubled point is +6 dB,
                     not +3. Also p90/median NN = 2.99 (clumpy), 5.4% of points with
                     gaps > lambda/2, and outlier-corrupted (39% long on its principal
                     axis). Visually it does not even read as a pistol.
      * phone.xyz -- a REGULAR 1.000 mm lattice = 1.0 lambda, 97.5% above lambda/2:
                     exactly the configuration our own diagnostic showed grows a
                     +7 dB grating lobe.
      * the 1911 shell inside man_gun.stl -- only 11% of its normals face outward
                     (a closed surface must be ~50%), i.e. broken winding on an open
                     non-manifold shell; impossible 193 cm^2 area. Not trustworthy.
    """
    lib = {}

    # ---- THREATS -----------------------------------------------------------
    # Glock 17 Gen 5, CGTrader high-poly (ZBrush sculpt, royalty-free incl. AI use).
    # 2.11M faces @ 0.135 lambda median edge -- 3.7x finer than the lambda/2 gate, so
    # NO subdivision is needed (which sidesteps Loop-vs-midpoint edge rounding on a
    # machined object). Ships at true real-world scale; normals 48% outward = closed.
    lib["pistol"] = Threat(
        "pistol", "threat", "steel", parts=_glock_parts(glock_stl),
        notes="REAL Glock 17 Gen5, EXTERIOR SHELL ONLY (see exterior_shell.py -- the "
              "raw solid carries the bore/magwell/frame internals, ~half its 1038 cm2, "
              "which THz can never see; stripping removes 46% of the area). TWO "
              "materials: steel slide+sights over a POLYMER frame/grip (~11.7 dB "
              "internal contrast). The reference threat.")

    lib["knife_steel"] = Threat(
        "knife_steel", "threat", "steel", mesh=_tapered_blade(0.22, 0.035, 0.004),
        notes="22cm steel blade")
    lib["knife_ceramic"] = Threat(
        "knife_ceramic", "threat", "ceramic", mesh=_tapered_blade(0.22, 0.035, 0.004),
        notes="SAME geometry as knife_steel, dielectric material -- isolates the "
              "material axis from the shape axis")

    lib["pipe_bomb"] = Threat(
        "pipe_bomb", "threat", "steel", mesh=_pipe(0.15, 0.042),
        notes="capped pipe; a smooth convex cylinder -> ONE tracking glint")

    lib["sheet_explosive"] = Threat(
        "sheet_explosive", "threat", "explosive", mesh=_slab(0.20, 0.12, 0.020),
        conformal=True,
        notes="conformal dielectric slab taped to the torso -- the hardest threat: "
              "thin, low-s, and it BENDS to the body so it has no rigid edge")

    lib["det_wire"] = Threat(
        "det_wire", "threat", "steel", mesh=_cyl(0.30, 0.002, sections=12),
        conformal=True,
        notes="30cm det wire, 2mm dia -- sub-resolution in cross-section but a "
              "strong specular line when broadside")

    # ---- BENIGN CLUTTER (the false-positive drivers) -----------------------
    # Real scanned phone, rebuilt as a convex hull of the (lattice-corrupt) cloud. A
    # phone is a rounded box, so the hull is faithful: median cloud->hull deviation
    # 0.00 mm, p95 0.82 mm. Watertight, and resamples clean at lambda/8.
    lib["phone"] = Threat(
        "phone", "benign", "alloy", mesh=trimesh.load(phone_stl, process=False),
        notes="REAL scanned phone (s=0.9 in the legacy assets.json), hull-rebuilt. "
              "The PRIMARY confuser for sheet_explosive -- a bright slab at pocket "
              "height. If a detector can't separate these two, the dataset taught it "
              "nothing useful.")
    lib["belt_buckle"] = Threat(
        "belt_buckle", "benign", "steel", mesh=_slab(0.075, 0.050, 0.005),
        notes="bright metal at exactly the appendix-carry height")
    lib["keys"] = Threat(
        "keys", "benign", "steel", mesh=_slab(0.060, 0.030, 0.006),
        notes="small bright metal clutter, pocket height")
    lib["wallet"] = Threat(
        "wallet", "benign", "plastic", mesh=_slab(0.095, 0.075, 0.015),
        conformal=True, notes="low-s conformal slab -- confuser for sheet_explosive")

    return lib


if __name__ == "__main__":
    lib = build_library()
    print(f"{'name':<17}{'kind':<8}{'mat':<11}{'s':>5} {'rough':>6}  "
          f"{'LxWxT (cm)':>16} conformal")
    for t in lib.values():
        e = t.extent() * 100
        print(f"{t.name:<17}{t.kind:<8}{t.material:<11}{t.s:>5.2f} "
              f"{t.rough:>6.2f}  {e[0]:>5.1f}x{e[1]:>4.1f}x{e[2]:>4.1f}   "
              f"{'yes' if t.conformal else '-'}")
    print(f"\nlambda = {LAM*1e3:.3f} mm")
    q, n = lib["knife_steel"].sample(LAM / 8)
    print(f"knife sampled: {len(q):,} pts  (target NN lambda/8)")
    q, n = lib["pistol"].sample(LAM / 8)
    print(f"pistol cloud : {len(q):,} pts")

"""Threat-skin multipath engine: the dihedral double-bounce the single-bounce model can't produce.

WHY THIS EXISTS
---------------
The core forward model (`model_gen/python/raforward.py`, imported as `rf`) is single-bounce:

    sig1[f,i] = Sum_tx Sum_t  s_t / (r1 r2) exp(-j k_f (r1 + r2))      # Tx -> t -> Rx_i

Every scattering point radiates independently; there is no point-to-point interaction. The one
mechanism it *structurally* cannot produce is the corner reflector formed where a hard, flat
threat surface beds against the body -- the dihedral DOUBLE bounce:

    sig2[f,i] = Sum_tx Sum_{a,b} K s_a s_b W1 W2 / (r1 rAB r3)
                                 exp(-j k_f (r1 + rAB + r3))          # Tx -> a -> b -> Rx_i
                a on threat, b nearby

  r1 = |Tx-a|,  rAB = |a-b|,  r3 = |b-Rx_i|.

This term is (1) displaced in range -- it images BEHIND the true surface, the classic artifact;
(2) retro-reflective -- a ~90 deg corner returns a fan of aspects coherently, bright and
angle-robust, not a fleeting specular flash; (3) a material discriminant -- it scales as
s_a * s_b, so metal-on-skin (1.0 x 0.6) dominates a dielectric benign object on skin.

The model is LINEAR in the field, so this is a purely additive coherent term on the SAME
(nf, nrx) measurement array:  sig_total = rf.forward(...)  +  double_bounce(...).

WHY OUR OWN COHERENT SUM (not rf's kernel via retro_sim's r1-injection)
-----------------------------------------------------------------------
The Tx is a single fixed point; the Rx is a 130x130 grid. The first-bounce facet lobe W1 (at a)
depends only on (Tx, a, b) -- fixed per pair. But the second-bounce lobe W2 (at b) depends on the
Rx direction, and THAT Rx-dependence is exactly the retroreflection (a 90 deg corner sends the
Tx->a->b ray back antiparallel, i.e. only Rx near the retro direction light up). rf's kernel takes
one scalar `s` per target and cannot carry a per-(pair, Rx) weight, so we sum coherently here.
We match rf's conventions exactly (k = wavenumbers 2 pi f / c; Tx summed; output (nf, nrx)) and
validate against it (validate_multipath.py): range-focus of the ghost, and K fixed so a dihedral
comes out sqrt(2)x an equal flat plate -- the textbook 8 pi vs 4 pi corner-vs-plate ratio.

CALIBRATION KNOBS (both documented, neither a hidden claim)
  gloss : Blinn-Phong exponent = the surface ROUGHNESS knob (DESIGN.md sec.5). Large = mirror-
          smooth, tight bright retro lobe; small = rough, broad and dim.
  K     : corner-coupling amplitude, pinned by calibrate_K() so a perfect dihedral is sqrt(2)x a
          flat plate of the same face size at the same range. One number, then reused on bodies.
"""
from __future__ import annotations
import numpy as np

C0 = 299_792_458.0


def _xp(use_gpu):
    if use_gpu:
        import cupy as cp
        return cp
    return np


# --------------------------------------------------------------------------------------------
# 1. Pair construction: which corners actually exist between threat A and body-shell B.
# --------------------------------------------------------------------------------------------
def seam_pairs(qA, nA, qB, nB, prune_dist, face_tol=0.05):
    """Concave corners between threat points A and body-shell points B.

    A pair (a,b) survives only if it forms a concave wedge the wave can rattle inside of:
      * separation |q_a - q_b| <= prune_dist            (a corner is local; far pairs are noise)
      * mutually facing (concave): n_a . (q_b-q_a) > face_tol*r  AND  n_b . (q_a-q_b) > face_tol*r
        i.e. the two normals point INTO the wedge. Convex edges (normals apart) don't retro.

    Returns index arrays (ia, ib) into A and B (one entry per surviving A->B corner). Both
    orderings (first bounce on threat vs first bounce on body) are materialised later by
    double_bounce_from_pairs; summing both is what turns a plate's 4 pi into the corner's 8 pi.
    """
    qA = np.asarray(qA, float); nA = np.asarray(nA, float)
    qB = np.asarray(qB, float); nB = np.asarray(nB, float)
    # KD-tree radius query: a corner is LOCAL, so only near neighbours can pair. This is
    # O(N log N + #pairs) instead of the O(Na*Nb) dense block, which is essential at body
    # sampling density (0.125 lambda) where the dense block would be tens of billions of cells.
    from scipy.spatial import cKDTree
    nbrs = cKDTree(qB).query_ball_point(qA, prune_dist)     # list per A of nearby B indices
    ia_all, ib_all = [], []
    for a, bl in enumerate(nbrs):
        if not bl:
            continue
        bl = np.asarray(bl)
        d = qB[bl] - qA[a]                                  # (m,3) = q_b - q_a
        r = np.linalg.norm(d, axis=1)
        faceA = d @ nA[a] > face_tol * r                    # a's front sees b
        faceB = np.einsum('mk,mk->m', -d, nB[bl]) > face_tol * r  # b's front sees a
        m = (r > 1e-6) & faceA & faceB
        if m.any():
            ia_all.append(np.full(int(m.sum()), a)); ib_all.append(bl[m])
    ia = np.concatenate(ia_all) if ia_all else np.array([], int)
    ib = np.concatenate(ib_all) if ib_all else np.array([], int)
    return ia, ib


def stationary_pairs(qA, nA, qB, nB, tx, *, max_perp=0.0015, max_path=0.05, step=0.001,
                     face_tol=0.02):
    """STATIONARY-PHASE pairing: one specular partner b per threat point a (O(N), convergent).

    Replaces seam_pairs' fixed-distance band (which sums O(pairs)~1/sp^4 and DIVERGES on shallow,
    broadly-in-phase contacts like a gun bedded flat against the belly). The physical double-bounce
    surface integral is dominated by the single (a,b) where the path Tx->a->b->Rx is stationary --
    specular at BOTH bounces. We pick that partner analytically instead of summing a band and hoping
    the phase cancels the rest (it doesn't, for a shallow corner). Result: pair count = N_a ~ 1/sp^2,
    the SAME scaling as the single bounce, so corner-vs-scene converges in sampling density.

    Construction, single Tx (this rig): reflect the Tx->a ray about n_a to get the only specular
    outgoing direction; b is the skin point ON that reflected ray. Validated by (i) perpendicular
    distance to the ray <= max_perp, (ii) along-ray t in (0, max_path], (iii) concave facing so it's
    a real inward corner. The second-bounce retro selectivity (does b re-aim at the aperture) is left
    to fast_double_bounce_from_pairs' W2 -- shallow contacts that don't retro fall out there.

    qA,nA : (Na,3) threat points + unit normals   qB,nB : (Nb,3) skin points + unit normals
    tx    : (1,3) single transmitter
    Returns (ia, ib): index arrays into A and B, one specular partnering per surviving a.
    """
    from scipy.spatial import cKDTree
    qA = np.asarray(qA, float); nA = np.asarray(nA, float)
    qB = np.asarray(qB, float); nB = np.asarray(nB, float)
    t = np.asarray(tx, float).reshape(-1, 3)[0]
    # max_perp is a FIXED physical corridor (~1.5 mm ~ 1.5 lambda), NOT tied to the sampling: a real
    # specular partner sits within a wavelength or so of the reflected ray. Keeping it density-
    # independent (a) makes the physics converge and (b) fixes nstep so the ray-march cost does not
    # blow up as the mesh refines (auto-scaling max_perp -> nstep ~ 1/spacing -> unusably slow).
    # first-bounce specular direction: reflect the incoming Tx->a ray about n_a
    din = qA - t[None, :]; din /= (np.linalg.norm(din, axis=1, keepdims=True) + 1e-12)
    drefl = din - 2.0 * np.sum(din * nA, axis=1, keepdims=True) * nA
    drefl /= (np.linalg.norm(drefl, axis=1, keepdims=True) + 1e-12)
    # b = where the reflected ray meets the skin. The hit can be cm down-ray, so MARCH the ray and
    # keep, per a, the sample whose nearest skin point is closest to the ray (min perpendicular
    # distance). A fixed short probe can't find an intersection that is not adjacent to a.
    tree = cKDTree(qB)
    Na = len(qA)
    nstep = max(6, int(np.ceil(max_path / max(0.5 * max_perp, 1e-4))))
    best_perp = np.full(Na, np.inf); best_ib = np.zeros(Na, np.int64)
    for tt in np.linspace(step, max_path, nstep):
        _, ib = tree.query(qA + tt * drefl, k=1)
        v = qB[ib] - qA
        tpar = np.sum(v * drefl, axis=1)
        perp = np.linalg.norm(v - tpar[:, None] * drefl, axis=1)
        better = (perp < best_perp) & (tpar > 1e-4) & (tpar <= max_path)
        best_perp[better] = perp[better]; best_ib[better] = ib[better]
    ib = best_ib
    # validate: b sits on the ray (perp <= a voxel), ahead of a, concave (mutually facing) wedge
    v = qB[ib] - qA
    tpar = np.sum(v * drefl, axis=1)
    r = np.linalg.norm(v, axis=1) + 1e-12
    faceA = np.sum(v * nA, axis=1) > face_tol * r          # a's front sees b
    faceB = np.sum(-v * nB[ib], axis=1) > face_tol * r     # b's front sees a (concave wedge)
    keep = (best_perp <= max_perp) & (tpar > 1e-4) & (tpar <= max_path) & faceA & faceB
    ia = np.flatnonzero(keep)
    return ia, ib[keep]


# --------------------------------------------------------------------------------------------
# 2. The coherent three-leg double-bounce sum, matching rf's conventions.
# --------------------------------------------------------------------------------------------
def double_bounce(qa, na, sa, qb, nb, sb, tx, rx, k, *,
                  K=1.0, gloss=200.0, w1_eps=1e-3, use_gpu=False, rx_chunk=4096, pair_chunk=8192):
    """Coherent monostatic-array double bounce over ORDERED corner pairs (Tx -> a -> b -> Rx).

    Elementwise-paired inputs: pair p bounces first at (qa[p], na[p], sa[p]) then at
    (qb[p], nb[p], sb[p]). Pass both orderings (see seam_pairs) for a symmetric corner.

    qa,na : (P,3) first-bounce points + unit normals   sa : (P,) amplitude reflectivity
    qb,nb : (P,3) second-bounce points + unit normals   sb : (P,)
    tx    : (T,3) transmitters (coherently summed, as rf does)
    rx    : (I,3) receivers
    k     : (F,)  WAVENUMBERS 2 pi f / c   (same as rf.Scene.k -- NOT Hz)

    SPECULAR PRUNE (exact, not an approximation): the first-bounce lobe W1 depends only on
    (Tx, a, b) -- it is Rx-INDEPENDENT because Tx is a single fixed point. With any real gloss
    the overwhelming majority of geometric pairs are far off-specular (W1 ~ 0) and contribute
    nothing, so we drop pairs with W1 <= w1_eps BEFORE the (Rx x freq) sum. This is what makes
    the engine scale: a body seam has millions of geometric corners but only thousands carry
    signal for a given Tx. Set w1_eps=0 to keep every pair (slow; only for cross-checks).

    Returns sig2 : (F, I) complex128, to be added onto the single-bounce measurement.
    """
    xp = _xp(use_gpu)
    qa = xp.asarray(qa, xp.float64); na = xp.asarray(na, xp.float64); sa = xp.asarray(sa, xp.float64)
    qb = xp.asarray(qb, xp.float64); nb = xp.asarray(nb, xp.float64); sb = xp.asarray(sb, xp.float64)
    tx = xp.asarray(tx, xp.float64); rx = xp.asarray(rx, xp.float64); k = xp.asarray(k, xp.float64)
    I, F = len(rx), len(k)

    uAB0 = qb - qa
    rAB0 = xp.linalg.norm(uAB0, axis=1)
    uAB0 = uAB0 / rAB0[:, None]

    sig2 = xp.zeros((F, I), dtype=xp.complex128)
    for txp in tx:
        d1 = qa - txp[None, :]                              # Tx -> a
        r1f = xp.linalg.norm(d1, axis=1)                    # (P,)
        u1 = d1 / r1f[:, None]                              # travel dir Tx->a
        # first-bounce lobe: specular half-vector between (toward Tx = -u1) and (toward b = uAB)
        h1 = (-u1) + uAB0
        h1 = h1 / (xp.linalg.norm(h1, axis=1, keepdims=True) + 1e-12)
        W1 = xp.clip(xp.einsum('pk,pk->p', h1, na), 0, None) ** gloss   # (P,) fixed per pair
        keep = W1 > w1_eps                                 # <-- Rx-independent specular prune
        if not bool(keep.any()):
            continue
        qbk = qb[keep]; nbk = nb[keep]
        uABk = uAB0[keep]; rABk = rAB0[keep]; r1k = r1f[keep]
        amp1k = K * sa[keep] * sb[keep] * W1[keep] / r1k   # (Pk,) everything Rx-independent
        Pk = len(qbk)

        # chunk over BOTH pairs and Rx so the (pair x rx) intermediates stay bounded -- at body
        # sampling Pk can be ~1e5 and a full (Pk, nrx) array would be tens of GB.
        for p0 in range(0, Pk, pair_chunk):
            sl = slice(p0, p0 + pair_chunk)
            qbp = qbk[sl]; nbp = nbk[sl]; uAB = uABk[sl]; rAB = rABk[sl]; r1 = r1k[sl]
            amp1 = amp1k[sl]
            for i0 in range(0, I, rx_chunk):
                rxb = rx[i0:i0 + rx_chunk]                  # (i,3)
                d3 = rxb[None, :, :] - qbp[:, None, :]      # (p,i,3) b -> Rx
                r3 = xp.linalg.norm(d3, axis=2)             # (p,i)
                u3 = d3 / r3[..., None]                     # travel dir b->Rx
                # second-bounce lobe: half-vector of (toward a = -uAB) and (toward Rx = u3)
                h2 = (-uAB)[:, None, :] + u3
                h2 = h2 / (xp.linalg.norm(h2, axis=2, keepdims=True) + 1e-12)
                W2 = xp.clip(xp.einsum('pik,pk->pi', h2, nbp), 0, None) ** gloss   # (p,i)

                amp = (amp1[:, None] * W2) / (rAB[:, None] * r3)           # (p,i) real
                L = r1[:, None] + rAB[:, None] + r3                        # (p,i) total path
                for f in range(F):                          # loop freq, no (F,p,i) tensor
                    sig2[f, i0:i0 + len(rxb)] += (amp * xp.exp(-1j * k[f] * L)).sum(0)
    if use_gpu:
        sig2 = sig2.get()
    return np.asarray(sig2)


def voxel_downsample(q, n, s, voxel):
    """Decimate a point cloud to ~one point per `voxel`-sized cell (mean position + normal + s).
    The single-bounce driver samples at lambda/8 to kill speckle, but a COHERENT double-bounce sum
    is fully resolved at ~lambda/2 -- lambda/8 just multiplies the pair count 256x for no new
    information. Decimating both clouds to lambda/2 before pairing is what makes the body case
    tractable; the single bounce is recomputed at the SAME spacing so the two terms stay on one
    scale (K is calibrated at this spacing too)."""
    ijk = np.round(q / voxel).astype(np.int64)
    # hash the 3 voxel indices into ONE int64 (offset so negatives are non-negative). np.unique on
    # a 1-D key is a fast radix; np.unique(axis=0) over tens of millions of rows is a lexsort and
    # is pathologically slow -- that distinction is the difference between seconds and never.
    off = 1 << 20
    key = ((ijk[:, 0] + off) | ((ijk[:, 1] + off) << 21) | ((ijk[:, 2] + off) << 42))
    _, inv = np.unique(key, return_inverse=True)
    m = int(inv.max()) + 1
    qd = np.zeros((m, 3)); nd = np.zeros((m, 3)); sd = np.zeros(m); cnt = np.zeros(m)
    np.add.at(qd, inv, q); np.add.at(nd, inv, n); np.add.at(sd, inv, s); np.add.at(cnt, inv, 1.0)
    qd /= cnt[:, None]; sd /= cnt
    nd /= (np.linalg.norm(nd, axis=1, keepdims=True) + 1e-12)
    return qd, nd, sd


def contact_band(qP, qQ, dist):
    """Indices of points in qP that lie within `dist` of ANY point in qQ (the contact strip).
    Restricting BOTH clouds to their mutual contact band before pairing is the first, cheapest
    cut -- a corner only forms where the two surfaces actually come close."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(qQ).query(qP, k=1)
    return np.flatnonzero(d <= dist)


def return_visible(pts, rx_center, occ, *, r_block=0.0012, self_skip=0.003, max_t=0.25, step=0.0008):
    """Line-of-sight test for the RETURN leg b->Rx of the double bounce.

    A skin point b that is tucked UNDER the threat is shadowed from the aperture: the metal body
    of the gun sits between b and Rx, so its second bounce never reaches the array. The pairing
    (stationary_pairs) has no such test -- it keeps any geometrically-facing wedge -- so those
    occluded points inflate the corner. Here we march the segment b->rx_center against the threat's
    OWN surface cloud `occ` (the gun, an opaque shell) and drop b if the ray pierces it.

    Only the gun is used as the occluder (not the body): the body near b is b's own surface and
    would self-shadow; the physically dominant blocker for an under-gun skin point is the gun. Rx is
    ~1 m off so we test visibility to the aperture centre (matches the W2-toward-centre convention).

    pts       : (M,3) points to test (the skin partners bqc[ib])
    rx_center : (3,) aperture centroid
    occ       : (G,3) opaque occluder cloud (the gun at pairing density)
    Returns a bool mask (M,) -- True = clear line of sight to the aperture.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, float); occ = np.asarray(occ, float)
    c = np.asarray(rx_center, float).reshape(3)
    d = c[None, :] - pts
    L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    dhat = d / L
    tree = cKDTree(occ)
    clear = np.ones(len(pts), bool)
    for tt in np.arange(self_skip, max_t, step):
        within = clear & (tt < (L[:, 0] - self_skip))     # still on the segment, still uncondemned
        if not within.any():
            break
        dist, _ = tree.query(pts + tt * dhat)             # nearest gun point to the marched position
        clear &= ~(within & (dist < r_block))             # pierced the gun shell -> occluded
    return clear


def double_bounce_from_pairs(qA, nA, sA, qB, nB, sB, ia, ib, tx, rx, k, **kw):
    """Convenience: expand seam_pairs() index arrays (both orderings) into elementwise pair
    arrays and run double_bounce. `s*` are per-point reflectivities of the two clouds."""
    qa = np.vstack([qA[ia], qB[ib]]); na = np.vstack([nA[ia], nB[ib]])
    sa = np.concatenate([sA[ia], sB[ib]])
    qb = np.vstack([qB[ib], qA[ia]]); nb = np.vstack([nB[ib], nA[ia]])
    sb = np.concatenate([sB[ib], sA[ia]])
    return double_bounce(qa, na, sa, qb, nb, sb, tx, rx, k, **kw)


# --------------------------------------------------------------------------------------------
# 2b. FAST path: reuse rf's CUDA kernel by injecting the first-leg path length (retro_sim trick).
# --------------------------------------------------------------------------------------------
def fast_double_bounce_from_pairs(qA, nA, sA, qB, nB, sB, ia, ib, tx, rx, k, custom_r1_fn, *,
                                  K=1.0, gloss=200.0, w1_eps=1e-3, precision="fp32"):
    """Same double bounce as double_bounce_from_pairs BUT ~100x faster, at one modelling cost.

    The optimised rf CUDA kernel computes  Sum_j s_j/(r1_j r2_j) exp(-j k (r1_j + r2_j))  with
    r2 = |q_j - Rx| in-kernel and r1 INJECTED per (tx, target). Place a virtual scatterer at the
    second-bounce point b, inject r1 = |Tx-a| + |a-b|, and fold the amplitude so the kernel's
    1/(r1 r2) becomes the true 1/(|Tx-a| |a-b| |b-Rx|):

        s_eff = K s_a s_b W1 (r1a + rAB) / (r1a rAB)

    The phase is then EXACT (r1a + rAB + r3) and range-faithful. The kernel carries one SCALAR per
    target, so the second-bounce lobe W2 cannot vary per-Rx -- but the aperture here is tiny
    (~6.7 cm at 1 m => +/-2 deg), so W2 barely changes across it. We therefore evaluate W2 ONCE per
    pair toward the APERTURE CENTRE and fold it (with W1) into s_eff. That RESTORES the
    geometry-dependent second-bounce directivity -- the thing that (correctly) makes an off-aimed
    corner weak -- to within a fraction of a dB of the per-Rx both-lobe engine, at ~100x the speed.
    (Isotropic re-radiation -- omitting W2 -- was wrong by ~40 dB where corners don't retro-aim at
    the array.) Use double_bounce() for the exact per-Rx proof; this for body surveys.

    custom_r1_fn := retro_sim.channel._forward_custom_r1  (kept as an arg so this module stays
    dependency-free). Single-Tx assumption (this rig has one) so W1/s_eff are unambiguous.
    """
    assert len(tx) == 1, "fast path assumes a single Tx (W1 and s_eff are per-Tx)"
    # both orderings: first bounce on threat then body, and vice-versa
    qa = np.vstack([qA[ia], qB[ib]]); na = np.vstack([nA[ia], nB[ib]])
    sa = np.concatenate([sA[ia], sB[ib]])
    qb = np.vstack([qB[ib], qA[ia]]); nb = np.vstack([nB[ib], nA[ia]])
    sb = np.concatenate([sB[ib], sA[ia]])

    t = np.asarray(tx[0], float)
    rxc = np.asarray(rx, float).mean(0)                    # aperture centre (W2 ~ constant across it)
    d1 = qa - t[None, :]; r1 = np.linalg.norm(d1, axis=1); u1 = d1 / r1[:, None]
    uAB = qb - qa; rAB = np.linalg.norm(uAB, axis=1); uAB = uAB / rAB[:, None]
    # first-bounce lobe at a (toward Tx and toward b)
    h1 = (-u1) + uAB; h1 = h1 / (np.linalg.norm(h1, axis=1, keepdims=True) + 1e-12)
    W1 = np.clip(np.einsum('pk,pk->p', h1, na), 0, None) ** gloss
    # second-bounce lobe at b (toward a and toward the aperture centre)
    u3c = rxc[None, :] - qb; u3c = u3c / (np.linalg.norm(u3c, axis=1, keepdims=True) + 1e-12)
    h2 = (-uAB) + u3c; h2 = h2 / (np.linalg.norm(h2, axis=1, keepdims=True) + 1e-12)
    W2 = np.clip(np.einsum('pk,pk->p', h2, nb), 0, None) ** gloss
    W = W1 * W2
    keep = W > w1_eps
    if not keep.any():
        return np.zeros((len(k), len(rx)), np.complex64)
    r1k = r1[keep]; rABk = rAB[keep]; r1inj = r1k + rABk
    s_eff = (K * sa[keep] * sb[keep] * W[keep] * r1inj / (r1k * rABk)).astype(np.float64)
    r1all = r1inj.reshape(1, -1)                            # tx-major (ntx=1, npairs)
    sig = custom_r1_fn(qb[keep].astype(np.float64), s_eff, np.asarray(tx, float),
                       np.asarray(rx, float), np.asarray(k, float), r1all, precision=precision)
    return np.asarray(sig)


# --------------------------------------------------------------------------------------------
# 3. Analytic reference: the dihedral corner reflector (for validate_multipath.py).
# --------------------------------------------------------------------------------------------
def plate_rcs(a, b, lam):
    """Peak monostatic RCS of a flat a x b conducting plate at normal incidence (m^2)."""
    return 4 * np.pi * a ** 2 * b ** 2 / lam ** 2


def dihedral_rcs(a, b, lam):
    """Peak monostatic RCS of a 90 deg dihedral of two a x b plates (m^2), at the corner bisector.
    Exactly twice the plate: the double bounce is what the multipath engine must reproduce."""
    return 8 * np.pi * a ** 2 * b ** 2 / lam ** 2


if __name__ == "__main__":
    # geometry/pruning smoke test: a right-angle corner of two facet grids -> pairs hug the seam.
    a = np.linspace(0.0, 0.10, 20)
    X, Y = np.meshgrid(a, a)
    q1 = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]; n1 = np.tile([0, 0, 1.0], (X.size, 1))
    X, Z = np.meshgrid(a, a)
    q2 = np.c_[X.ravel(), np.zeros(X.size), Z.ravel()]; n2 = np.tile([0, 1.0, 0], (X.size, 1))
    ia, ib = seam_pairs(q1, n1, q2, n2, prune_dist=0.03)
    print(f"corner pairs within 3 cm: {len(ia)}  (a band hugging the y=z=0 seam)")
    off = np.linalg.norm(q1[ia][:, 1:], axis=1)
    print(f"  plate-1 members lie {off.min()*100:.1f}-{off.max()*100:.1f} cm off the seam")

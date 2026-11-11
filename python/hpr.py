"""Port of utilities/hpr.m: Hidden-Point-Removal occlusion culling.

Algorithm from Katz, Tal, Basri - "Direct Visibility of Point Sets".
"""
import numpy as np
from scipy.spatial import ConvexHull


def hpr(p, C, param):
    """Return indices of points in `p` visible from camera position `C`.

    p: (N, 3) point cloud
    C: (3,) camera position
    param: HPR culling factor (spherical-flip radius exponent)
    """
    p = np.asarray(p, dtype=float) - np.asarray(C, dtype=float)

    normp = np.sqrt(np.sum(p * p, axis=1))

    R = normp.max() * (10.0 ** param)

    P = p + 2 * (R - normp)[:, None] * p / normp[:, None]

    hull = ConvexHull(P)
    return np.unique(hull.vertices)

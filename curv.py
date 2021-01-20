# -*- coding: utf-8 -*-
import numpy as np


def rotate(radians):
    M = np.eye(3)
    M[0, 0], M[1, 1] = np.cos(radians), np.cos(radians)
    M[0, 1], M[1, 0] = np.sin(radians), -np.sin(radians)

    return M


def reorient(points, theta, center):
    M = rotate(theta)
    points_trans = points - center
    points_aug = np.hstack((points_trans, np.ones((points.shape[0], 1))))
    points_trans = np.dot(M, points_aug.transpose())
    points_trans = points_trans.transpose()[:, :2]
    points_trans += center

    assert points_trans.ndim == 2, 'points_trans.ndim == %d != 2' % (points_trans.ndim)

    return points_trans


# contour: (n, 2) array of (x, y) points
def oriented_curvature(contour, radii):
    curvature = np.zeros((contour.shape[0], len(radii)), dtype=np.float32)
    # define the radii as a fraction of either the x or y extent
    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        dists = ((contour - center) ** 2).sum(axis=1)
        inside = dists[:, np.newaxis] <= (radii * radii)

        for j, _ in enumerate(radii):
            curve = contour[inside[:, j]]
            # sometimes only a single point lies inside the circle
            if curve.shape[0] == 1:
                curv = 0.5
            else:
                n = curve[-1] - curve[0]
                theta = np.arctan2(n[1], n[0])

                curve_p = reorient(curve, theta, center)
                center_p = np.squeeze(reorient(center[None], theta, center))
                r0 = center_p - radii[j]
                r1 = center_p + radii[j]
                r0[0] = max(curve_p[:, 0].min(), r0[0])
                r1[0] = min(curve_p[:, 0].max(), r1[0])

                area = np.trapz(curve_p[:, 1] - r0[1], curve_p[:, 0], axis=0)
                curv = area / np.prod(r1 - r0)
            curvature[i, j] = curv

    return curvature

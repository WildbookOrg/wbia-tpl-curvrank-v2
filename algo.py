# -*- coding: utf-8 -*-
from wbia_curvrank_v2.stitch import stitch
import cv2
import numpy as np
from sklearn.utils import shuffle

import torch
import torch.utils.data as data
from collections import defaultdict
from torch.utils.data import DataLoader


def reorient_normals(contour, normals):
    if len(normals) == 1:
        return normals

    for i in range(1, len(normals)):
        p = contour[i] - contour[i - 1]
        n = normals[i]
        if np.cross(p, n) < 0:
            normals[i] *= -1.0

    if np.dot(normals[1], normals[0]) < 0.0:
        normals *= -1.0

    return normals


def shift_contour_points(scores, max_offset):
    m, n = scores.shape
    lookup = np.zeros((m, n + 1), dtype=np.float32)
    offsets = np.full((m, n + 1), -1, dtype=np.int32)

    for col in range(0, n + 1):
        for row in range(0, m):
            offsets_idx = np.arange(-max_offset, max_offset + 1)
            offsets_val = np.full(offsets_idx.shape[0], -np.inf, dtype=np.float32)
            for i, idx in enumerate(offsets_idx):
                # Outside the grid.  Invalid move.
                if row + idx < 0 or row + idx >= m:
                    offsets_val[i] = -np.inf
                # First column.  All previous scores are zero.
                elif col == 0:
                    offsets_val[i] = scores[row, col]
                # Last column.  "Dummy" to determine endpoint.
                elif col == n:
                    offsets_val[i] = lookup[row + idx, col - 1]
                # General case.  Best cost thus far and current move.
                else:
                    offsets_val[i] = lookup[row + idx, col - 1] + scores[row, col]

            max_idx = offsets_val.argmax()
            lookup[row, col] = offsets_val[max_idx]
            offsets[row, col] = max_idx - max_offset

    end = lookup[:, n].argmax()
    cost = lookup[end, n]
    curr = end + offsets[end, n]
    path = []
    for i in range(n - 1, -1, -1):
        path.append((i, curr))
        curr = curr + offsets[curr, i]

    path = np.vstack(path)[::-1]

    return path, cost


def control_points(probs):
    _, probs_thresh = cv2.threshold(probs, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pad the thresholded probabilities to avoid peaks on the boundary
    # in the distance transform.
    probs_thresh = cv2.copyMakeBorder(probs_thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    dist = cv2.distanceTransform(probs_thresh, cv2.DIST_L2, cv2.DIST_MASK_5)

    nonzero_coords = np.vstack(np.where(dist > 0)).T

    smoothed = cv2.GaussianBlur(dist, (3, 3), 0.5)
    # Compute the normal vectors at each point on the contour.
    normals, hess, grad = normals_from_hessian(smoothed, nonzero_coords)

    # Localize the maxima to subpixel accuracy and do NMS.
    peaks_ij, is_max = subpixel_loc(smoothed, normals, nonzero_coords)

    # Undo the padding.
    peaks_ij = peaks_ij[1:-1, 1:-1] - np.array([1.0, 1.0])
    normals = normals[1:-1, 1:-1]
    is_max = is_max[1:-1, 1:-1]

    return peaks_ij, normals, is_max


class PatchSet(data.Dataset):
    def __init__(self, img, contour, patch_dims, patch_size):
        super(PatchSet, self).__init__()
        self.image = img
        self.contour = contour
        self.patch_dims = patch_dims
        self.patch_size = patch_size

    def __getitem__(self, index):
        pt = self.contour[index]
        img_crop_width = int(np.round(self.patch_dims[0]))
        img_crop_height = int(np.round(self.patch_dims[1]))
        crop = cv2.getRectSubPix(self.image, (img_crop_width, img_crop_height), tuple(pt))
        crop = cv2.resize(
            crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA
        )
        crop = crop.transpose(2, 0, 1) / 255.0

        return torch.FloatTensor(crop)

    def __len__(self):
        return self.contour.shape[0]


def refine_contour(
    img, cropped_img, bounding_box, contour, patch_dims, patch_size, patchnet, device
):
    using_gpu = torch.cuda.is_available()

    (x0, y0, x1, y1) = bounding_box

    patchset = PatchSet(img, contour, patch_dims, patch_size)

    batch_size = 64
    patch_iter = DataLoader(
        patchset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=batch_size // 8,
        pin_memory=using_gpu,
    )

    patch_probs = np.zeros((contour.shape[0], patch_size, patch_size), dtype=np.float32)

    for i, x in enumerate(patch_iter):
        if using_gpu:
            x = x.to(device)
        with torch.no_grad():
            _, y_hat = patchnet(x)
        p_vals = (
            y_hat.cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .reshape(-1, patch_size, patch_size, 2)
        )

        patch_probs[i * batch_size : (i + 1) * batch_size] = p_vals[:, :, :, 1]

    part_img = img[y0:y1, x0:x1]
    contour -= np.array([x0, y0])

    patch_width, patch_height = patch_dims
    cost = np.zeros(part_img.shape[0:2], dtype=np.float32)
    # cost = np.full(part_img.shape[0:2], 100., dtype=np.float32)
    contour = contour.astype(np.float32)
    weight = np.zeros(part_img.shape[0:2], dtype=np.int32)
    stitch(contour, patch_probs, patch_width, patch_height, cost, weight)
    stitched = np.divide(cost, weight, out=np.zeros_like(cost), where=weight > 0)
    return stitched


def link_points(peaks, normals, is_max):
    assert peaks.shape == normals.shape
    k = 3
    pad = k // 2
    peaks = cv2.copyMakeBorder(peaks, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    normals = cv2.copyMakeBorder(normals, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    is_max = cv2.copyMakeBorder(
        is_max.astype(np.int32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0
    )
    is_max = is_max.astype(np.bool)

    fwd_links = np.full(peaks.shape, -1, dtype=np.int32)
    bwd_links = np.full(peaks.shape, -1, dtype=np.int32)
    idx = np.argwhere(is_max)
    for (i, j) in idx:
        n = normals[i, j]
        pi, pj = peaks[i, j]

        i0, i1 = i - k // 2, i + k // 2 + 1
        j0, j1 = j - k // 2, j + k // 2 + 1
        max_idx = np.copy(is_max[i0:i1, j0:j1])
        max_idx[k // 2, k // 2] = False
        nbr_peaks = peaks[i0:i1, j0:j1][max_idx]
        nbr_index = np.argwhere(max_idx) + np.array([i0, j0])

        # if np.allclose(n[1], 0.):
        #    c0, c1, c2 = 0., -1., pj
        # else:
        #    m = n[0] / n[1]
        #    c0, c1, c2 = 1., -m, m * pj - pi
        # d = c0 * nbr_peaks[:, 0] + c1 * nbr_peaks[:, 1] + c2
        # assert np.all(np.sign(a) == np.sign(d)), (a, d, n)
        d = (nbr_peaks[:, 0] - pi) * n[1] - (nbr_peaks[:, 1] - pj) * n[0]
        posd, negd = d > 0, d < 0
        if np.any(posd):
            pdists = np.linalg.norm(peaks[i, j] - nbr_peaks[posd], axis=1)
            assert not np.allclose(nbr_index[posd][pdists.argmin()], (i, j))
            fwd_links[i, j] = nbr_index[posd][pdists.argmin()]
        else:
            fwd_links[i, j] = (-1, -1)

        if np.any(negd):
            ndists = np.linalg.norm(peaks[i, j] - nbr_peaks[negd], axis=1)
            bwd_links[i, j] = nbr_index[negd][ndists.argmin()]
        else:
            bwd_links[i, j] = (-1, -1)

    for (i, j) in idx:
        fi, fj = fwd_links[i, j]
        if np.all(fwd_links[fi, fj] == (i, j)) or np.all(bwd_links[fi, fj] == (i, j)):
            pass
        else:
            fwd_links[i, j] = (-1, -1)

        bi, bj = bwd_links[i, j]
        if np.all(fwd_links[bi, bj] == (i, j)) or np.all(bwd_links[bi, bj] == (i, j)):
            pass
        else:
            bwd_links[i, j] = (-1, -1)

    visited = np.ones(peaks.shape[0:2], dtype=np.bool)
    visited[idx[:, 0], idx[:, 1]] = False

    def merge(i, j):
        if i < 0 or j < 0 or visited[i, j]:
            return []
        else:
            visited[i, j] = True
            # One of these will be [] because terminal or has been visited.
            fi, fj = fwd_links[i, j]
            bi, bj = bwd_links[i, j]
            return merge(fi, fj) + merge(bi, bj) + [(i, j)]

    contours = []
    for (i, j) in idx:
        if not visited[i, j]:
            visited[i, j] = True

            ni, nj = fwd_links[i, j]
            fwd_contour = merge(ni, nj)

            ni, nj = bwd_links[i, j]
            bwd_contour = merge(ni, nj)

            # We use the same recursive function for searching both
            # directions, so need to reverse the backward one.
            contour = np.vstack(
                [x for x in fwd_contour + [(i, j)] + bwd_contour[::-1]]
            ) - np.array([pad, pad])

            contours.append(contour)

    return contours


def subpixel_loc(dist, normals, idx):
    merge_thresh = np.sqrt(2.0) / 4.0
    thetas = np.rad2deg(np.arctan2(normals[:, :, 0], normals[:, :, 1])) % 360

    subpixel = np.zeros((dist.shape[0], dist.shape[1], 2), dtype=np.float32)
    is_max = np.zeros(dist.shape[0:2], dtype=np.bool)
    idx = shuffle(idx)
    for pt_idx, (i, j) in enumerate(idx):
        n = normals[i, j]  # normal vector as (i, j)
        t = thetas[i, j]  # theta in degrees [0, 360)
        if 0 <= t < 45:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            i0, i1 = m * (j - 1) + c, m * (j + 1) + c
            d = np.linalg.norm((i - i0, 1.0))

            p0 = (i - i0) * dist[i, j - 1] + (i0 - i + 1) * dist[i - 1, j - 1]
            p1 = (i1 - i) * dist[i + 1, j + 1] + (i + 1 - i1) * dist[i, j + 1]
        elif 45 <= t < 90:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            j0, j1 = (i - 1 - c) / m, (i + 1 - c) / m
            d = np.linalg.norm((1.0, j - j0))

            p0 = (j - j0) * dist[i - 1, j - 1] + (j0 - j + 1) * dist[i - 1, j]
            p1 = (j1 - j) * dist[i + 1, j + 1] + (j + 1 - j1) * dist[i + 1, j]
        elif 90 <= t < 135:
            # Need to be careful of vertical lines!
            if np.allclose(n[1], 0.0):
                j0, j1 = j, j
            else:
                m = n[0] / n[1]  # slope of the line
                c = i - m * j  # intercept
                j0, j1 = (i - 1 - c) / m, (i + 1 - c) / m
            d = np.linalg.norm((1.0, j - j0))

            p0 = (j0 - j) * dist[i - 1, j + 1] + (j + 1 - j0) * dist[i - 1, j]
            p1 = (j - j1) * dist[i + 1, j - 1] + (j1 - j + 1) * dist[i + 1, j]
        elif 135 <= t < 180:
            m = n[0] / n[1]  # slope of the line
            c = i - (n[0] / n[1]) * j  # intercept
            i0, i1 = m * (j + 1) + c, m * (j - 1) + c
            d = np.linalg.norm((i - i0, 1.0))

            p0 = (i - i0) * dist[i - 1, j + 1] + (i0 - i + 1) * dist[i, j + 1]
            p1 = (i1 - i) * dist[i + 1, j - 1] + (i + 1 - i1) * dist[i, j - 1]
        elif 180 <= t < 225:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            i0, i1 = m * (j + 1) + c, m * (j - 1) + c
            d = np.linalg.norm((i0 - i, 1.0))

            p0 = (i0 - i) * dist[i + 1, j + 1] + (i + 1 - i0) * dist[i, j + 1]
            p1 = (i - i1) * dist[i - 1, j - 1] + (i1 - i + 1) * dist[i, j - 1]
        elif 225 <= t < 270:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            j0, j1 = (i + 1 - c) / m, (i - 1 - c) / m
            d = np.linalg.norm((1.0, j0 - j))

            p0 = (j0 - j) * dist[i + 1, j + 1] + (j + 1 - j0) * dist[i + 1, j]
            p1 = (j - j1) * dist[i - 1, j - 1] + (j1 - j + 1) * dist[i - 1, j]
        elif 270 <= t < 315:
            # Need to be careful of vertical lines!
            if np.allclose(n[1], 0.0):
                j0, j1 = j, j
            else:
                m = n[0] / n[1]  # slope of the line
                c = i - m * j  # intercept
                j0, j1 = (i + 1 - c) / m, (i - 1 - c) / m
            d = np.linalg.norm((1.0, j0 - j))

            p0 = (j - j0) * dist[i + 1, j - 1] + (j0 - j + 1) * dist[i + 1, j]
            p1 = (j1 - j) * dist[i - 1, j + 1] + (j + 1 - j1) * dist[i - 1, j]
        # Because of floating point precision 360 should be inclusive.
        elif 315 <= t <= 360:
            m = n[0] / n[1]  # slope of the line
            c = i - m * j  # intercept
            i0, i1 = m * (j - 1) + c, m * (j + 1) + c
            d = np.linalg.norm((i0 - i, 1.0))

            p0 = (i0 - i) * dist[i + 1, j - 1] + (i + 1 - i0) * dist[i, j - 1]
            p1 = (i - i1) * dist[i - 1, j - 1] + (i1 - i + 1) * dist[i, j + 1]
        else:
            assert False, '%.f' % t

        pc = dist[i, j]  # Peak of the pixel at (i, j).
        # This peak is a maximum, need to localize it so subpixel accuracy.
        if p0 <= pc >= p1:
            # Parabola: f(-d) = p0, f(0) = pc, f(d) = p1.  Scale to [-1, 1].
            # We are solving the system of equations:
            # c0 + c1 * x + c2 * x^2 = p0 at x = -1
            # c0 + c1 * x + c2 * x^2 = pc at x =  0
            # c0 + c1 * x + c2 * x^2 = p1 at x =  1
            # c0 = pc
            c1 = 0.5 * (p1 - p0)
            c2 = 0.5 * (p0 + p1) - pc
            assert c2 <= 0.0, '2nd derivative of parabola must be negative!'
            # Peak is where first derivative is zero.
            # TODO: check local min./max.?
            ds = -d * c1 / (2.0 * c2)

            # Shift the peak by ds along the normal direction.
            subpixel[i, j] = np.array([i, j]) + n * ds
            is_max[i, j] = True
        # This point is not a maximum and should be suppressed.
        else:
            subpixel[i, j] = np.array([i, j])
            is_max[i, j] = False

    # After localizing peaks to subpixel accuracy, some points may now be too
    # close together or even overlap exactly.  We need to merge these before
    # linking to avoid ambiguities in the contours.
    k = 3
    idx = np.argwhere(is_max)
    to_merge = defaultdict(list)
    for (i, j) in idx:
        i0, i1 = i - k // 2, i + k // 2 + 1
        j0, j1 = j - k // 2, j + k // 2 + 1
        max_idx = np.copy(is_max[i0:i1, j0:j1])
        max_idx[k // 2, k // 2] = False
        nbr_peaks = subpixel[i0:i1, j0:j1][max_idx]
        nbr_index = np.argwhere(max_idx) + np.array([i0, j0])

        # The distances from the peak at (i, j) to its neighbors.
        dists = np.linalg.norm(subpixel[i, j] - nbr_peaks, axis=1)
        if dists.size:
            too_close = dists < merge_thresh
            # assert too_close.sum() <= 1  # Sanity check...
            # For each point, we store the (i, j) location of the peak taht is
            # too close.  There is no need to store the reverse too.
            for ix in nbr_index[too_close]:
                ij = tuple(ix)
                if ij not in to_merge:
                    to_merge[(i, j)].append(ij)

    for (i, j) in to_merge:
        nbrs_ij = to_merge[i, j]
        # Get the subpixel peak locations to be merged.
        merge_peaks = np.vstack(
            (subpixel[i, j], [subpixel[ix, jx] for (ix, jx) in nbrs_ij])
        )
        # Get the normals to be merged and account for the sign ambiguity.
        merge_normals = np.vstack(
            (
                normals[i, j],
                [
                    normals[ix, jx]
                    if np.dot(normals[ix, jx], normals[i, j]) >= 0.0
                    else -1.0 * normals[ix, jx]
                    for (ix, jx) in nbrs_ij
                ],
            )
        )

        new_peak = merge_peaks.mean(axis=0)
        new_normal = merge_normals.mean(axis=0)

        # The first point is arbitrarily chosen as the subpixel location of
        # the merged peaks.
        new_normal = new_normal / np.linalg.norm(new_normal)
        subpixel[i, j] = new_peak
        # The points that have been merged are no longer peaks.
        for (ix, jx) in nbrs_ij:
            is_max[ix, jx] = False

    return subpixel, is_max


def normals_from_hessian(x, idx):
    # Kernels for the first and second derivatives in x and y.
    Sx = np.array([[0.0, 0.0, 0.0], [-0.5, 0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    Sy = np.array([[0.0, -0.5, 0.0], [0.0, 0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    Sxx = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    Syy = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    Sxy = np.array(
        [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]], dtype=np.float32
    )

    dx = cv2.filter2D(x, cv2.CV_32F, Sx)
    dy = cv2.filter2D(x, cv2.CV_32F, Sy)
    dxx = cv2.filter2D(x, cv2.CV_32F, Sxx)
    dyy = cv2.filter2D(x, cv2.CV_32F, Syy)
    dxy = cv2.filter2D(x, cv2.CV_32F, Sxy)

    normals = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
    hess = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
    grad = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float32)
    for (i, j) in idx:
        # Hessian matrix at the point (i,j).
        H = np.array([[dxx[i, j], dxy[i, j]], [dxy[i, j], dyy[i, j]]])
        # Derivative vector at the point (i, j).
        D = np.array([dx[i, j], dy[i, j]])

        # Choose the eigenvector corresponding to the most negative eigenvalue.
        w, v = np.linalg.eig(H)
        hv = v[:, w.argmin()][::-1]
        # Implicit scaling and weighting by 2-norm.
        dv = D[::-1]

        # Account for the sign ambiguity.
        if np.dot(dv, hv) < 0.0:
            dv = -1.0 * dv

        # Approx. the normal vector at a point as the weighted combination of
        # the first and second derivatives.
        nv = dv + np.linalg.norm(H) * hv
        normals[i, j] = nv / np.linalg.norm(nv)

        hess[i, j] = hv
        grad[i, j] = dv

    return normals, hess, grad

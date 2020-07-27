import cv2
import colorsys
import numpy as np

from scipy import interpolate


# Checks the contour directionality based on the Part type.  Returns true if
# the contour should be flipped to ensure consistency and false otherwise.
# contour: list of numpy arrays in matrix coordinates.
def reverse_contour_directionality(contour, type_):
    # Flukes: left to right.
    if type_.lower() == 'fluke':
        start, end = contour[0][0], contour[-1][-1]
        return start[1] > end[1]  # Compare j-coordinates.
    # Dorsal fins: left to right.
    elif type_.lower() == 'dorsal':
        start, end = contour[0][0], contour[-1][-1]
        return start[1] > end[1]  # Compare j-coordinates.
    # Left and right ears: top to bottom.
    elif type_.lower() in ('left ear', 'right ear'):
        start, end = contour[0][0], contour[-1][-1]
        return start[0] < end[0]  # Compare i-coodinates.
    else:
        raise ValueError('No consistent directionality defined for Part type:'
                         '%s.' % (type_))


# Applies resampling along the axis=0, treating all columns as
# independent functions of equidistant points.
def resample1d(input, length):
    interp = np.linspace(0, length, num=input.shape[0])
    f = interpolate.interp1d(interp, input, axis=0, kind='linear')

    return f(np.arange(length))


# Resamples a parametric curve f(t) = (x(t), y(t)), while assuming that
# initially the points are not necessarily equidistant.
def resample2d(input, length):
    dist = np.linalg.norm(np.diff(input, axis=0), axis=1)
    u = np.hstack((0., np.cumsum(dist)))
    t = np.linspace(0., u.max(), length)
    xn = np.interp(t, u, input[:, 0])
    yn = np.interp(t, u, input[:, 1])

    return np.vstack((xn, yn)).T


def pad_curvature_gaps(contour, curvature):
    # When padding we don't want to change the point density. Mostly this
    # should be 1, but may be up to sqrt(2) when we allow diagonal steps in A*.
    avg_step = np.mean([
        np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1)) for c in contour
        if c.shape[0] > 1  # Ignore single points.
    ])
    # Need to copy the contour because we're inserting into it.
    copy_contour = list(contour)
    num_scales = curvature[0].shape[1]
    dtype = curvature[0].dtype
    gaps_filled = 0
    # TODO: this may be simpler done backwards.
    for i in range(1, len(copy_contour)):
        prev, curr = copy_contour[i - 1], copy_contour[i]
        dist = np.linalg.norm(curr[0] - prev[-1])
        num_steps = int(np.ceil(dist / avg_step))
        # Inserting n points means n + 1 steps to get from A to B:
        # A x x x B
        step = dist / (1. + num_steps)
        normal = (curr[0] - prev[-1]) / dist
        # Don't duplicate the existing contour points, this breaks the
        # resampling.  So start/end one step away.
        cont_filler = prev[-1] + (
            normal * np.linspace(step, dist - step, num_steps)[:, None]
        )
        curv_filler = np.full((num_steps, num_scales), 0.5, dtype=dtype)
        contour.insert(i + gaps_filled, cont_filler)
        curvature.insert(i + gaps_filled, curv_filler)
        gaps_filled += 1

    return np.vstack(contour), np.vstack(curvature)


def random_colors(n):
    grc = 0.618033988749895
    h = np.random.random()
    colors = []
    for i in range(n):
        h += grc
        h %= 1
        r, g, b = colorsys.hsv_to_rgb(h, 0.99, 0.99)
        colors.append((255. * r, 255. * g, 255. * b))

    return colors


def points_to_mask(pts, radii, occluded, size):
    mask = np.zeros(size, dtype=np.uint8)
    for idx, (x, y) in enumerate(pts):
        r = radii[idx]
        if not occluded[idx]:
            cv2.circle(mask, (x, y), r, 255, -1)

    return mask


# Padding is expressed as a fraction of the width.
def crop_with_padding(image, x, y, w, h, pad):
    img_height, img_width = image.shape[0:2]
    if x >= 0 or y >= 0 or w >= 0 or h >= 0:
        x0 = max(0, x - int(pad * w))
        x1 = min(img_width, x + w + int(pad * w))
        y0 = max(0, y - int(pad * h))
        y1 = min(img_height, y + h + int(pad * h))
        crop = image[y0:y1, x0:x1]
    else:
        crop = image
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]

    return crop, (x0, y0, x1, y1)


# https://github.com/martinjevans/OpenCV-Rotate-and-Crop/blob/master/rotate_and_crop.py
def sub_image(image, center, theta, width, height,
              border_mode=cv2.BORDER_REPLICATE):
    """Extract a rectangle from the source image.

    image - source image
    center - (x,y) tuple for the centre point.
    theta - angle of rectangle.
    width, height - rectangle dimensions.
    """

    #if np.pi / 4. < theta <= np.pi / 2.:
    #    theta = theta - np.pi / 2.
    #    width, height = height, width

    #theta *= np.pi / 180  # convert to rad
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(image, mapping, (width, height),
                          flags=cv2.WARP_INVERSE_MAP,
                          borderMode=border_mode,
                          borderValue=0.)

def extend_line(p1, p2):
    m = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    new_p1 = p1 + 1000*m
    new_p2 = p2 - 1000*m
    return new_p1, new_p2
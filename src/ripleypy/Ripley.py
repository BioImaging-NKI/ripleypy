from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

N_SAMPLES = 256
angles = np.linspace(0, 2 * np.pi, N_SAMPLES, endpoint=False)
UX = np.cos(angles)
UY = np.sin(angles)


@dataclass
class PointOptions:
    class PointMode(Enum):
        CIRCLE = 0
        RECTANGLE = 1

    mode: PointMode = PointMode.CIRCLE
    radius: float = 0
    center: tuple[float, float] = (0, 0)
    width: float = 1
    height: float = 1


def generate_points(M: int, options: PointOptions) -> npt.NDArray[np.float64]:
    """
    Generate uniformly distributed 2D points.

    Parameters
    ----------
    M : int
        Number of points.
    options : PointOptions
    """

    if options.mode == options.PointMode.CIRCLE:
        radius = options.radius

        theta = np.random.uniform(0, 2 * np.pi, M)
        r = radius * np.sqrt(np.random.rand(M))
        cx, cy = options.center

        x = r * np.cos(theta) + cx
        y = r * np.sin(theta) + cy

    elif options.mode == options.PointMode.RECTANGLE:
        width = options.width
        height = options.height
        cx, cy = options.center

        x = np.random.uniform(cx - width / 2, cx + width / 2, M)
        y = np.random.uniform(cy - height / 2, cy + height / 2, M)

    else:
        raise ValueError(f"Unknown shape '{options.mode}'")

    return np.column_stack((x, y))


def circle_fraction(
    cx: npt.NDArray[np.float64], cy: npt.NDArray[np.float64], r: npt.NDArray[np.float64], mask: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Estimate the fraction of each circle circumference that lies on the
    positive mask.

    Parameters
    ----------
    cx, cy : (N,) arrays
        Circle centres.
    r : (N,) array
        Circle radii.
    mask : (H,W) bool array

    Returns
    -------
    fraction : (N,) float array
    """

    H, W = mask.shape

    xs = cx[:, None] + r[:, None] * UX
    ys = cy[:, None] + r[:, None] * UY

    ix = np.rint(xs).astype(np.int32)  # round to nearest integer
    iy = np.rint(ys).astype(np.int32)

    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    samples = np.zeros_like(xs, dtype=bool)
    samples[inside] = mask[iy[inside], ix[inside]]

    return samples.mean(axis=1)


def check_random_distances(
    points: npt.NDArray[np.float64], mask: npt.NDArray[np.float64], N: int, rmax: float
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """
    Measures the distance between two randomly chosen points.
    Calculates a weight based on the fraction of that distance radius inside the mask.
    Only distances <rmax are stored.
    When the number of distances exceeds N the function returns.
    It is not exactly N because we want to know the fraction of checked distances.
    :param points:
    :param mask:
    :param N:
    :param rmax:
    :return: distance matrix with two weights per distance, meta-data with the checked fraction
    """
    M = len(points)
    n_distances = M**2 - M
    order = np.random.permutation(M)

    x = points[:, 0]
    y = points[:, 1]

    rmax2 = rmax * rmax

    results = []

    checked = 0

    n_out = 0

    for stride in range(1, M):
        a = order[:-stride]
        b = order[stride:]

        dx = x[a] - x[b]
        dy = y[a] - y[b]
        d2 = dx * dx + dy * dy

        checked += d2.size

        mask_valid = d2 <= rmax2
        idx = np.flatnonzero(mask_valid)

        if idx.size == 0:
            continue

        r = np.sqrt(d2[idx])

        frac_a = circle_fraction(x[a[idx]], y[a[idx]], r, mask)
        frac_b = circle_fraction(x[b[idx]], y[b[idx]], r, mask)

        chunk = np.column_stack((d2[idx], frac_a, frac_b))

        results.append(chunk)
        n_out += chunk.shape[0]

        if n_out >= N:
            break

    out = np.vstack(results) if results else np.empty((0, 3))

    return out, {
        "checked_pairs": checked,
        "checked_fraction": checked / n_distances,
    }


def ripley(
    points: npt.NDArray[np.float64], mask: npt.NDArray[np.float64], n_distances_to_check: int, rmax: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, float]]:
    n_points = len(points)
    area = mask.sum()
    distances, metadata = check_random_distances(points, mask, n_distances_to_check, rmax)
    # mean of the reciprocal weights
    distances_mean = np.column_stack((np.sqrt(distances[:, 0]), 0.5 * (1 / distances[:, 1] + 1 / distances[:, 2])))
    # since we don't check every distance, each distance needs additional weight
    distances_mean[:, 1] = distances_mean[:, 1] / metadata["checked_fraction"]
    # cummulative sum of the sorted distances
    distances_sorted = distances_mean[np.argsort(distances_mean[:, 0])]
    d_cum = np.column_stack((distances_sorted[:, 0], np.cumsum(distances_sorted[:, 1])))

    # calculate the ripley functions
    r = d_cum[:, 0]
    k_function = (area / (n_points**2)) * d_cum[:, 1]
    l_function = np.sqrt(k_function / np.pi) - r
    return r, l_function, metadata

from pathlib import Path

import numpy as np
import shapely
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from ripleypy import generate_points_in_mask, ripley

SCALE = 5  # raw data units per mask pixel; the test data spans ~1360 x 910 units
PAD_REGION = 10
N_DISTANCES = int(1e4)
RMAX = 200
N_RANDOM_CURVES = 10
BAND_PERCENTILES = (12.5, 87.5)  # a "75% interval" around the median

data = np.loadtxt(Path(Path.cwd(), "testdata", "testdata.csv"), delimiter=",")

# Put at (0,0)
data[:, 0] -= data[:, 0].min()
data[:, 1] -= data[:, 1].min()

# Scale down
data[:, 0] /= SCALE
data[:, 1] /= SCALE

# Shift from (0,0)
data += PAD_REGION

# Create mask from Convex hull
hull = ConvexHull(data)
hull_pts = data[hull.vertices]
# Image dimensions
width = int(data[:, 0].max()) + PAD_REGION * 2
height = int(data[:, 1].max()) + PAD_REGION * 2
# Pixel centers
xs = np.arange(width)
ys = np.arange(height)
X, Y = np.meshgrid(xs, ys)
# Rasterize hull
poly = shapely.Polygon(hull_pts).buffer(PAD_REGION)
pts = shapely.points(X.ravel(), Y.ravel())
mask = shapely.contains(poly, pts).reshape(height, width)

rmax_pixels = RMAX / SCALE
r, l_function, meta = ripley(data, mask, N_DISTANCES, rmax_pixels)
r = np.concatenate(([0.0], r))
l_function = np.concatenate(([0.0], l_function))
l_minus_r = (l_function - r) * SCALE
r *= SCALE

# Random curves in the same mask, to show the noise floor as a median +/- percentile band
random_curves = []
for _ in range(N_RANDOM_CURVES):
    points = generate_points_in_mask(len(data), mask)
    r_rand, l_rand, _ = ripley(points, mask, N_DISTANCES, rmax_pixels)
    random_curves.append((r_rand * SCALE, l_rand * SCALE))

r_common = np.linspace(0, RMAX, 200)
interpolated_l = np.array(
    [np.interp(r_common, np.concatenate(([0.0], r_i)), np.concatenate(([0.0], l_i))) for r_i, l_i in random_curves]
)
interpolated = interpolated_l - r_common
median = np.median(interpolated, axis=0)
lo, hi = np.percentile(interpolated, BAND_PERCENTILES, axis=0)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.imshow(mask, origin="lower", extent=(0, width * SCALE, 0, height * SCALE), interpolation="nearest")
ax1.scatter(data[:, 0] * SCALE, data[:, 1] * SCALE, s=5, c="r")
ax1.set_aspect("equal", adjustable="box")

band_width = BAND_PERCENTILES[1] - BAND_PERCENTILES[0]
ax2.fill_between(r_common, lo, hi, color="grey", alpha=0.3, label=f"random {band_width:.0f}% interval")
ax2.plot(r_common, median, color="grey", linestyle="--", label="random median")
ax2.plot(r, l_minus_r, color="C0", label="observed")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlim(0, RMAX)
ax2.set_xlabel("r")
ax2.set_ylabel("L(r) - r")
ax2.legend()

fig.suptitle(f"real world data (n={len(data)})")

out_path = Path(Path.cwd(), "img", "example2_ripley.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path)
print(out_path)

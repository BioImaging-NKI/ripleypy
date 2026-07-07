from pathlib import Path

import numpy as np
import shapely
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from ripleypy import generate_points_in_mask, ripley

SCALE = 50
PAD_REGION = 10
N_DISTANCES = int(1e4)
RMAX = 10000

data = np.loadtxt(Path(Path.cwd(), "testdata", "testdata.csv"), delimiter=",")

# Put at (0,0)
data[:, 0] -= data[:, 0].min()
data[:, 1] -= data[:, 1].min()

# Scale down
data[:, 0] /= SCALE
data[:, 1] /= SCALE

# Shift from (0,0)
data += PAD_REGION

data = np.loadtxt(Path(Path.cwd(), "testdata", "testdata.csv"), delimiter=",")

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

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
# Display data
ax1.imshow(mask, origin="lower", extent=(0, width * SCALE, 0, height * SCALE), interpolation="nearest")
ax1.scatter(data[:, 0] * SCALE, data[:, 1] * SCALE, s=5, c="r")

r, l_function, meta = ripley(data, mask, N_DISTANCES, RMAX / SCALE)
r *= SCALE
l_function *= SCALE

ax2.plot(r, l_function - r)
out_path = Path(Path.cwd(), "img", "example2_ripley.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path)
print(out_path)
fig.clf()

# Generate random curves in mask
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
for i in range(10):
    points = generate_points_in_mask(len(data), mask)
    if i == 0:
        ax1.imshow(mask, origin="lower", extent=(0, width * SCALE, 0, height * SCALE), interpolation="nearest")
        ax1.scatter(points[:, 0] * SCALE, points[:, 1] * SCALE, s=5, c="r")
    r, l_function, meta = ripley(points, mask, N_DISTANCES, RMAX / SCALE)
    r *= SCALE
    l_function *= SCALE
    ax2.plot(r, l_function - r)

out_path = Path(Path.cwd(), "img", "example2_ripley_random.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path)
print(out_path)
fig.clf()

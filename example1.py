from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ripleypy import generate_points, ripley
from ripleypy.Ripley import PointOptions

# Mode
mode = "random_example"

# Test parameters
N_POINTS = 2000
BOUNDS = (200, 100)
N_DISTANCES = 20000
RMAX = 50
N_RANDOM_CURVES = 10
BAND_PERCENTILES = (12.5, 87.5)  # a "75% interval" around the median
FRACTION = 0.0
if mode == "clustered_example":
    FRACTION = 0.2

mask = np.ones((BOUNDS[1], BOUNDS[0]), dtype=np.bool)

# Generate some testdata
options = PointOptions(
    mode=PointOptions.PointMode.RECTANGLE, width=BOUNDS[0], height=BOUNDS[1], center=(BOUNDS[0] / 2, BOUNDS[1] / 2)
)
points1 = generate_points(N_POINTS - int(N_POINTS * FRACTION), options)
options = PointOptions(mode=PointOptions.PointMode.CIRCLE, radius=10, center=(BOUNDS[0] / 2, BOUNDS[1] / 2))
points2 = generate_points(int(N_POINTS * FRACTION), options)
points = np.concatenate((points1, points2), axis=0)

r, l_function, _ = ripley(points, mask, N_DISTANCES, RMAX)
l_minus_r = l_function - r

# Random curves in the same mask, to show the noise floor as a median +/- percentile band
random_curves = []
for _ in range(N_RANDOM_CURVES):
    r_rand, l_rand, _ = ripley(points, mask, N_DISTANCES, RMAX, do_random=True)
    random_curves.append((r_rand, l_rand - r_rand))

r_common = np.linspace(0, RMAX, 200)
interpolated = np.array([np.interp(r_common, r_i, l_i) for r_i, l_i in random_curves])
median = np.median(interpolated, axis=0)
lo, hi = np.percentile(interpolated, BAND_PERCENTILES, axis=0)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.imshow(mask, origin="lower", extent=(0, BOUNDS[0], 0, BOUNDS[1]), interpolation="nearest")
ax1.plot(points[:, 0], points[:, 1], "o", markersize=1, color="r")
ax1.set_aspect("equal", adjustable="box")

band_width = BAND_PERCENTILES[1] - BAND_PERCENTILES[0]
ax2.fill_between(r_common, lo, hi, color="grey", alpha=0.3, label=f"random {band_width:.0f}% interval")
ax2.plot(r_common, median, color="grey", linestyle="--", label="random median")
ax2.plot(r, l_minus_r, color="C0", label=mode)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlim(0, RMAX)
ax2.set_xlabel("r")
ax2.set_ylabel("L(r) - r")
ax2.legend()

fig.suptitle(f"{mode} (n={len(points)})")

# Store figure
out_path = Path(Path.cwd(), "img", "example1_" + mode + ".png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path)

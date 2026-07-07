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
FRACTION = 0.0
if mode == "clustered_example":
    FRACTION = 0.2

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
for i in range(10):
    # Generate some testdata
    mask = np.ones((BOUNDS[1], BOUNDS[0]), dtype=np.bool)
    options = PointOptions(
        mode=PointOptions.PointMode.RECTANGLE, width=BOUNDS[0], height=BOUNDS[1], center=(BOUNDS[0] / 2, BOUNDS[1] / 2)
    )
    points1 = generate_points(N_POINTS - int(N_POINTS * FRACTION), options)
    options = PointOptions(mode=PointOptions.PointMode.CIRCLE, radius=10, center=(BOUNDS[0] / 2, BOUNDS[1] / 2))
    points2 = generate_points(int(N_POINTS * FRACTION), options)
    points = np.concatenate((points1, points2), axis=0)

    r, l_function, _ = ripley(points, mask, N_DISTANCES, RMAX)

    if i == 0:
        ax1.plot(points[:, 0], points[:, 1], "o", markersize=1)
        ax1.set_aspect("equal", adjustable="box")
    ax2.plot(r, l_function - r)

# Store figure
out_path = Path(Path.cwd(), "img", "example1_" + mode + ".png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path)

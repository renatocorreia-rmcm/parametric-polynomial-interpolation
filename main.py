import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from interpolation.aitken_interpolation import aitken_interpolation
from resample import resample

points: npt.NDArray[float] = np.array([
    (0.0, 0.0, 0.0),
    (2.0, 1.0, 1.0),
    (0.0, 3.0, 2.0),
    (0.0, 1.0, 3.0),
    (-2.0, 3.0, 4.0)
])


aitken_resampled_points = resample(
    interpolation_points=points,
    interpolation_function=aitken_interpolation
)

fig, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1])
ax.plot(aitken_resampled_points[:, 0], aitken_resampled_points[:, 1])
plt.savefig("output/aitken.png")



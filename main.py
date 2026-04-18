import numpy as np
import numpy.typing as npt

import matplotlib as mpl
import matplotlib.pyplot as plt

from aitken_interpolation import aitken_interpolation
from lagrange_interpolation import lagrange_interpolation

from resample import resample


points: npt.NDArray[float] = np.array([
    (0.0, 0.0, 0.0),
    (2.0, 1.0, 1.0),
    #(1.79538260e+00,  1.35094255e+00,  1.10344828e+00),  # aligned point  # test cancelation error
    (0.0, 3.0, 2.0),
    (0.0, 1.0, 3.0),
    (-1, 2.7, 3.5),
    (-2.0, 3.0, 4.0),
    # (0.0, 0.0, 5.0)  # close fig, but discontinuos
])


def plot_experiment(
        interpolation_points: npt.NDArray[float],
        interpolation_function,
        experiment_name: str = "experiment"
):

    resampled_points = resample(
        interpolation_points=interpolation_points,
        interpolation_function=interpolation_function,
    )

    def build_segments_and_lengths(points: np.ndarray):
        # Pairwise segments
        p0 = points[:-1]
        p1 = points[1:]
        segments = np.stack([p0, p1], axis=1)

        # Euclidean lengths
        lengths = np.linalg.norm(p1 - p0, axis=1)

        return segments, lengths

    segments, lengths = build_segments_and_lengths(resampled_points)

    norm = mpl.colors.Normalize(vmin=lengths.min(), vmax=lengths.max())
    cmap = plt.get_cmap('plasma')

    fig, ax = plt.subplots()
    ax.set_facecolor('black')

    for segment, length in zip(segments, lengths):
        color = cmap(norm(length))
        ax.plot(segment[:, 0], segment[:, 1], color=color)

    ax.scatter(interpolation_points[:, 0], interpolation_points[:, 1], color='white', zorder=2, marker='.')
    # ax.plot(interpolation_points[:, 0], interpolation_points[:, 1], color='white', zorder=2, marker='o')

    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Speed")
    plt.savefig(f"output/{experiment_name}.png", dpi=300)



plot_experiment(
    interpolation_points=points,
    interpolation_function=lagrange_interpolation,
    experiment_name="lagrange"
)

plot_experiment(
    interpolation_points=points,
    interpolation_function=aitken_interpolation,
    experiment_name="aitken"
)


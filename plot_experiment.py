import numpy as np
import numpy.typing as npt

import matplotlib as mpl
import matplotlib.pyplot as plt


# todo: INTERFACE EM ANYWIDGET ?


def plot_experiment(
        original_points: npt.NDArray[float],  # [(t, x, y)]
        resampled_points: npt.NDArray[float],  # [(t, x, y)]
        experiment_name: str = "test"
):
    """
    plot interpolated curve over scattered original points

    """
    # todo: parametized color as
    #  - speed,
    #  - t,
    #  - higher order derivatives ?

    # todo: allow multiple curves
    #   single color colormap for each

    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.set_aspect('equal', adjustable='datalim')

    if len(resampled_points) > 1:
        def build_segments_and_lengths(points: np.ndarray):
            """
            decompose curve in 2 point curves
            allow multicolored curves
            """
            # Pairwise segments
            p0 = points[:-1]
            p1 = points[1:]
            segments = np.stack([p0, p1], axis=1)

            # Euclidean lengths
            lengths = np.linalg.norm(p1[:, 1:] - p0[:, 1:], axis=1)

            return segments, lengths

        segments, lengths = build_segments_and_lengths(resampled_points)

        # PLOT

        vmin, vmax = lengths.min(), lengths.max()
        if vmin == vmax:
            vmax = vmin + 1e-12  # avoid zero range
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        cmap = plt.get_cmap('plasma')
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, label="speed")

        for segment, length in zip(segments, lengths):
            color = cmap(norm(length))  # speed = ds/dt  # currently is using only ds, but ok since all dt are equal
            ax.plot(
                segment[:, 1],
                segment[:, 2],
                color=color,
                linewidth=1.5,
                solid_capstyle='round'
            )

    # scatter original points over interpolated curve
    ax.scatter(original_points[1:-1, 1], original_points[1:-1, 2], color='white', zorder=2, marker='.')

    ax.scatter(original_points[0, 1], original_points[0, 2], color='white', zorder=2, marker='o')  # make start marker bigger
    ax.scatter(original_points[0, 1], original_points[0, 2], color='black', zorder=2, marker='.')  # make start marker bigger

    ax.scatter(original_points[-1, 1], original_points[-1, 2], color='white', zorder=2, marker='x')  # make start marker bigger

    plt.savefig(f"output/{experiment_name}.svg", dpi=300)

    plt.close(fig)

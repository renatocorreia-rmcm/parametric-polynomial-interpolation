import numpy as np
import numpy.typing as npt

import matplotlib as mpl
import matplotlib.pyplot as plt

# todo: INTERFACE EM ANYWIDGET ?


color_settings = [  # pairs Colormap / Color
    ("GnBu_r", "yellow"),
    ("YlGn_r", "magenta"),
    ("OrRd_r", "cyan"),
]
color_modes = [
    "parameter",
    "speed",
    # todo: parametized color as higher order derivatives
]


def plot_experiment(
        original_points: npt.NDArray[float],  # [(t, x, y)]
        resampled_points: npt.NDArray[float],  # [(t, x, y)]
        color_setting=0,
        color_mode="parameter",
        experiment_name: str = "test",
):
    """
    plot interpolated curve over scattered original points

    """

    assert 0 <= color_setting < len(color_settings)
    assert color_mode in color_modes

    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.set_aspect('equal', adjustable='datalim')

    colormap, contrast_color = color_settings[color_setting]

    if len(resampled_points) > 1:
        def build_segments_and_colors(points: np.ndarray):
            """
            decompose curve in 2 point curves
            allow multicolored curves
            """
            # Pairwise segments
            p0 = points[:-1]
            p1 = points[1:]
            segments = np.stack([p0, p1], axis=1)

            colors: npt.NDArray = np.array([])
            if color_mode == "speed":
                # Euclidean length of current segment is directly proportional to avg speed in that segment
                colors = np.linalg.norm(p1[:, 1:] - p0[:, 1:], axis=1)
            if color_mode == "parameter":
                # parameters
                colors = segments[:, 1, 0]

            return segments, colors

        segments, lengths = build_segments_and_colors(resampled_points)

        # PLOT

        vmin, vmax = lengths.min(), lengths.max()
        if vmin == vmax:
            vmax = vmin + 1e-12  # avoid zero range
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        cmap = plt.get_cmap(colormap)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, label=color_mode)

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
    ax.scatter(original_points[1:-1, 1], original_points[1:-1, 2], color=contrast_color, zorder=2, marker='.')
    # make start marker bigger
    ax.scatter(original_points[0, 1], original_points[0, 2], color=contrast_color, zorder=2, marker='o')
    ax.scatter(original_points[0, 1], original_points[0, 2], color='black', zorder=2, marker='.')
    # make end marker bigger
    ax.scatter(original_points[-1, 1], original_points[-1, 2], color=contrast_color, zorder=2, marker='x')

    plt.savefig(f"output/{experiment_name}.svg", dpi=300)

    plt.close(fig)

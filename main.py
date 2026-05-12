import numpy as np
import numpy.typing as npt

import vandermond
from sample_polynomials import sample_polynomials
from plot_experiment import plot_experiment

from parametize import parametize


def main(
        interpolation_points: npt.NDArray,
        experiment_name: str = None,
        color_setting: int = 0,
        color_mode: str = "parameter"
):
    # GET POLYNOMIALS

    x_polynomial = vandermond.coefficients(
        interpolation_points=interpolation_points[:, [0, 1]]
    )
    y_polynomial = vandermond.coefficients(
        interpolation_points=interpolation_points[:, [0, 2]]
    )

    # RESAMPLE POLYNOMIALS

    resampled_points = sample_polynomials(
        parameters=interpolation_points[:, 0],
        polynomials=np.array([x_polynomial, y_polynomial])
    )

    # PLOT

    plot_experiment(
        original_points=interpolation_points,
        resampled_points=resampled_points,
        color_setting=color_setting,
        color_mode=color_mode,
        experiment_name=experiment_name
    )


if __name__ == "__main__":

    points: npt.NDArray = np.array([  # todo: test cancelation error: include resampled points here (aligned points)
        # [t, x, y]
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
        [2.0, 0.0, 3.0],
        [3.0, 0.0, 1.0],
        [3.5, -1.0, 2.7],
        [4.0, -2.0, 3.0],
        [5.0, -1.0, -6.0]
    ])

    main(
        interpolation_points=parametize(points),
        experiment_name="manual",
        color_setting=0,
        color_mode="parameter"
    )

    main(
        interpolation_points=parametize(points, exponent=0),
        experiment_name="uniform",
        color_setting=1,
        color_mode="parameter"
    )

    main(
        interpolation_points=parametize(points, exponent=0.5),
        experiment_name="centripetal",
        color_setting=2,
        color_mode="parameter"
    )

    main(
        interpolation_points=parametize(points, exponent=1),
        experiment_name="chordal",
        color_setting=2,
        color_mode="parameter"

    )

    #
    # main(
    #     interpolation_points=parametize(points, exponent=0.25),
    #     experiment_name="0.25",
    #     color_setting=4
    # )

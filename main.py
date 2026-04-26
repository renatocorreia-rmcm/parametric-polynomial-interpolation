import numpy as np
import numpy.typing as npt

import vandermond
from sample_polynomials import sample_polynomials
from plot_experiment import plot_experiment


def main(
        interpolation_points: npt.NDArray[float],
        experiment_name: str = None
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
        experiment_name=experiment_name
    )


if __name__ == "__main__":

    points: npt.NDArray[float] = np.array([  # todo: test cancelation error: include resampled points here (aligned points)
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 1.0),
            (2.0, 0.0, 3.0),
            (3.0, 0.0, 1.0),
            (3.5, -1.0, 2.7),
            (4.0, -2.0, 3.0),
            (5.0, -1.0, -6.0)
    ])

    main(
        interpolation_points=points,
        experiment_name="vandermonde_interpolation"
    )

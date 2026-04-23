import numpy as np
import numpy.typing as npt
from evaluate_polynomial import evaluate_polynomial


def sample_polynomials(
        parameters: npt.NDArray[float],  # common array of parameters
        polynomials: npt.NDArray[float],  # arrays of coefficients for each polynomial
        relative_sample_rate: int = 2 * 6
) -> npt.NDArray[float]:
    """
    resample parametric 2D points using the given interpolation
    """

    # set samples

    samples_amount: int = relative_sample_rate * len(parameters)

    parameters_samples: npt.NDArray[float] = np.linspace(
        np.min(parameters),
        np.max(parameters),
        samples_amount,
        dtype=float
    )  # assume parameter is not necessairly monotonic  # if it was, could use only [0] and [-1] instead

    # compute samples

    values_samples: npt.NDArray[float] = np.array([
        evaluate_polynomial(
            parameters_samples=parameters_samples,
            coefficients=polynomial
        )
        for polynomial in polynomials
    ])

    # combine samples into points

    resampled_points: npt.NDArray[float] = np.column_stack(
        [parameters_samples, values_samples.T]
    )

    return resampled_points

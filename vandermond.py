import numpy as np
import numpy.typing as npt

import householder


def coefficients(
        interpolation_points: npt.NDArray[float],  # [(t, x, y)]
) -> npt.NDArray[float]:
    """
    return the coefficients of the polynomial that interpolates the given points, in increasing order of degree

    solve
    T*cx=x and T*cy=y
    where T is the Vandermond matrix of the t's
    """

    t, x, y = interpolation_points.T

    polynomial_degree = len(interpolation_points) - 1
    amount_of_coefficients = polynomial_degree + 1

    # setting T by vectorized approach T = [(t_i)^j]
    exponents = np.arange(amount_of_coefficients)
    T = t[:, None] ** exponents

    # solve T*[cx cy]=[x y]
    coefficients_x, coefficients_y = householder.solve_xy(A=T, x=x, y=y)

    return np.array([coefficients_x, coefficients_y])

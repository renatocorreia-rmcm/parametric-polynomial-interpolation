import numpy as np
import numpy.typing as npt

import householder


def coefficients(
        interpolation_points: npt.NDArray[float],  # [(t, y)]
) -> npt.NDArray[float]:
    """
    return the coefficients of the polynomial that interpolates the given points, in increasing order of degree

    y = T * a
    """

    t, y = interpolation_points.T

    polynomial_degree = len(interpolation_points) - 1
    amount_of_coefficients = polynomial_degree + 1

    T: npt.NDArray[float] = np.zeros(shape=(amount_of_coefficients, amount_of_coefficients), dtype=float)

    # setting T by vectorized approach T_ij = (t_i)^j
    exponents = np.arange(amount_of_coefficients)
    T = t[:, None] ** exponents

    # setting T by collumn
    """
    for index_column in range(amount_of_coefficients):
        T[:, index_column] = t ** index_column

    """

    # setting T by line
    """
    exponents = np.arange(amount_of_coefficients)
    print(exponents)
    for line in range(amount_of_coefficients):
        T[line] = interpolation_points[line, 0] ** exponents
    """

    T_inv = householder.inversion(T)

    coefficients = T_inv @ y

    return coefficients

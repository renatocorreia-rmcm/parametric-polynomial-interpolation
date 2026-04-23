import numpy as np
import numpy.typing as npt


def get_vandermonde_coefficients(
        interpolation_points: npt.NDArray[float],  # (t, y)
) -> npt.NDArray[float]:
    """
    y = T * a
    """

    t, y = interpolation_points.T

    polynomial_degree = len(interpolation_points) - 1
    amount_of_coefficients = polynomial_degree + 1

    T: npt.NDArray[float] = np.zeros(shape=(amount_of_coefficients, amount_of_coefficients), dtype=float)

    # setting T by collumn
    for index_column in range(amount_of_coefficients):
        T[:, index_column] = t ** index_column

    # setting T by line
    # exponents = np.arange(amount_of_coefficients)
    # print(exponents)
    # for line in range(amount_of_coefficients):
    #     T[line] = interpolation_points[line, 0] ** exponents

    coefficients = np.linalg.solve(T, y)  # todo: implement QR decomposition + inversion

    return coefficients

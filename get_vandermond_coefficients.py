import numpy as np
import numpy.typing as npt


points: npt.NDArray[float] = np.array([
    (0.0, 0.0),
    (1.0, 1.0),
    (3.0, 2.0),
    (1.0, 3.0),
    (2.7, 3.5),
    (-3.0, 4.0),
])


def get_vandermonde_coefficients(
        interpolation_points: npt.NDArray[float],
        t: int = None
) -> npt.NDArray[float]:
    """
    get array of points in format (y, t)

    p = T * a
    """

    p = interpolation_points[:, 0]

    polynomial_degree = len(interpolation_points) - 1
    amount_of_coefficients = polynomial_degree + 1

    T: npt.NDArray[float] = np.zeros(shape=(amount_of_coefficients, amount_of_coefficients), dtype=float)

    # setting T by collumn
    for column in range(amount_of_coefficients):
        T[:, column] = np.pow(interpolation_points[:, -1], column)

    # setting T by line
    # exponents = np.arange(amount_of_coefficients)
    # print(exponents)
    # for line in range(amount_of_coefficients):
    #     T[line] = interpolation_points[line, 0] ** exponents

    T_inv = np.linalg.inv(T)

    a = T_inv @ p

    return a

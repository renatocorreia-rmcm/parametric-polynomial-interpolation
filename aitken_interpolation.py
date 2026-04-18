import numpy as np
import numpy.typing as npt


def aitken_interpolation(
        interpolation_points: npt.NDArray[float],
        t: int,
        interpolation_indices: tuple[int, int] = None
) -> npt.NDArray[float]:
    # todo: optimize recurssion redundance -> return array of polynomial coefficients "get_aitken_coefficients"
    """

    :return: F(t) for F being the interpolated polynomial of points interpolation_points
    """

    # SET INTERPOLATION POINTS
    i: int
    j: int
    if interpolation_indices is None:
        i = 0
        j = len(interpolation_points)-1
    else:
        i, j = interpolation_indices

    # CASO BASE
    if i == j:
        y = interpolation_points[:, 0]
        return y[i]

    # CASO RECURSIVO
    else:
        parameters = interpolation_points[:, -1]
        t_i = parameters[i]
        t_j = parameters[j]

        return (
            (t_j - t) * aitken_interpolation(  # P_i_(j-1)
                interpolation_points=interpolation_points,
                t=t,
                interpolation_indices=(i, j-1)
            ) +
            (t - t_i) * aitken_interpolation(  # P_(i+1)_j
                interpolation_points=interpolation_points,
                t=t,
                interpolation_indices=(i+1, j)
            )
        ) / (t_j - t_i)

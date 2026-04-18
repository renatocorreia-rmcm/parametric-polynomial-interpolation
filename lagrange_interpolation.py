import numpy as np
import numpy.typing as npt


def lagrange_interpolation(
        interpolation_points: npt.NDArray[float],
        t: int
) -> npt.NDArray[float]:
    # todo: optimize redundance -> return array of polynomial coefficients "get_lagrange_coefficients"
    # todo: decompose each pa, so can visualize algorithm operation
    """

    :return: F(t) for F being the interpolated polynomial of points interpolation_points
    """

    p_t = 0

    for y_a, t_a in interpolation_points:
        pa_t = y_a

        for y_b, t_b in interpolation_points:
            if t_a != t_b:  # todo: solve: bad approach. use index
                pa_t *= (t - t_b)/(t_a-t_b)

        p_t += pa_t

    return p_t
import numpy as np
import numpy.typing as npt

relative_sample_rate = 6


def resample(
        interpolation_points: npt.NDArray[float],
        interpolation_function,
        samples_amount: int = None
) -> npt.NDArray[float]:
    """
    resample parametric 2D points using the given interpolation

    :param interpolation_points:
    :param interpolation_function:
    :param samples_amount:
    :return:
    """

    if samples_amount is None:
        samples_amount = relative_sample_rate*len(interpolation_points)

    parameters = interpolation_points[:, -1]

    t_samples = np.linspace(
        parameters[0],
        parameters[-1],
        samples_amount
    )  # assumes parameter is monotonic in points array  # otherwise use np.min/max instead of [0/-1]

    x_samples = np.array([
        interpolation_function(interpolation_points=interpolation_points[:, [0, -1]], t=t) for t in t_samples
    ])
    y_samples = np.array([
        interpolation_function(interpolation_points=interpolation_points[:, [1, -1]], t=t) for t in t_samples
    ])

    resampled_points: npt.NDArray[float] = np.stack([x_samples, y_samples, t_samples], axis=1)

    return resampled_points


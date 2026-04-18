import numpy as np
import numpy.typing as npt

def resample(
        interpolation_points: npt.NDArray[float],
        interpolation_function,
        relative_sample_rate: int = 2*6
) -> npt.NDArray[float]:
    """
    resample parametric 2D points using the given interpolation
    """

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


def resample_vandermond(
        interpolation_points: npt.NDArray[float],
        interpolation_function_x,
        interpolation_function_y,
        relative_sample_rate: int = 2*6
) -> npt.NDArray[float]:
    """
    resample parametric 2D points using the given interpolation
    """

    samples_amount = relative_sample_rate*len(interpolation_points)

    parameters = interpolation_points[:, -1]

    t_samples = np.linspace(
        parameters[0],
        parameters[-1],
        samples_amount
    )  # assumes parameter is monotonic in points array  # otherwise use np.min/max instead of [0/-1]

    x_samples = np.array([
        interpolation_function_x(t=t) for t in t_samples
    ])
    y_samples = np.array([
        interpolation_function_y(t=t) for t in t_samples
    ])

    resampled_points: npt.NDArray[float] = np.stack([x_samples, y_samples, t_samples], axis=1)

    return resampled_points


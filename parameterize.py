import numpy as np
import numpy.typing as npt


def parameterize(points: npt.NDArray, exponent: float = None) -> npt.NDArray:

    if exponent is None:
        assert points.shape[1] == 3  # must already have the parameters
        return points

    else:
        assert points.shape[1] in [2, 3]
        points = points[:, -2:]  # ignore the parameters

    """
    ti+1 := ti + di
    di := ||Pi+1 - Pi||^exponent
    """

    parameters = np.zeros(shape=len(points))

    for i in range(1, len(points)):
        parameters[i] = (
                parameters[i - 1] +
                np.linalg.norm(points[i] - points[i - 1]) ** exponent
        )

    return np.column_stack((parameters, points))


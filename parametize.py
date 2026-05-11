import numpy as np
import numpy.typing as npt

methods = [
    "manual",
    "uniform",
    "chordal",
    "centripetal",
    "foley-nielson"
]

def parametize(points: npt.NDArray, method: str):
    
    assert method in methods
    
    if method=="manual":
        assert points.shape[1] == 3  # must already have the parameters
    else:
        assert points.shape[1] == 2
    
    
    if method == "manual":
        return points
    
    if method == "uniform":
        parameters = np.arange(0, len(points))

        return np.column_stack((points, parameters))


    if method == "chordal":
            return points
    
    if method == "centripetal":
            return points
    
    if method == "foley-nielson":
            return points

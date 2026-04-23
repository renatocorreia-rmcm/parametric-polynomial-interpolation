import numpy as np
import numpy.typing as npt


def my_pow(x: float, e: int):
    r = 1
    for i in range(e): r *= x
    return r


# todo: implemment horner's method
# todo: use JIT
def evaluate_polynomial(coefficients: npt.NDArray[float], parameters_samples: npt.NDArray[float]) -> npt.NDArray[float]:
    results = np.zeros_like(parameters_samples)

    for i, t in enumerate(parameters_samples):
        result = 0
        for deg, coef in enumerate(coefficients):
            if coef != 0:
                result += coef * my_pow(t, deg)
        results[i] = result

    return results


# todo: cant parallelize horners method, but can parallelize [horner(t) for t in [...]]
#   (make horner a task) and create a top level for applying it over several points

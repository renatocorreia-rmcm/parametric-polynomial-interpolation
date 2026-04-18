import numpy as np
import numpy.typing as npt


def my_pow(x: float, e: int):
    r = 1
    for i in range(e): r*=x
    return r


def evaluate_polynomial(coefficients: npt.NDArray[float], t: float):

    result = 0

    for deg, coef in enumerate(coefficients):
        if coef != 0:
            result += coef * my_pow(t, deg)

    return result

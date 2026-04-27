import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular


def reflection(a_i: npt.NDArray[float], i: int) -> npt.NDArray[float]:
    """
    returns the Householder reflection matrix H_i such that H_i @ a_i = norm(a_i) * e_i
    """

    n = len(a_i)

    norm_a_i = np.linalg.norm(a_i)
    e_i = np.zeros(shape=n)
    e_i[i] = 1

    sign_a_i = np.sign(a_i[0]) if a_i[0] != 0 else 1.0
    v_i = a_i + sign_a_i * norm_a_i * e_i

    if np.linalg.norm(v_i) == 0: return np.identity(n)
    u_i = v_i / np.linalg.norm(v_i)

    H_i = np.identity(n) - 2 * np.outer(u_i, u_i)
    return H_i


def decomposition(A: npt.NDArray[float]) -> (npt.NDArray[float], npt.NDArray[float]):
    """
    returns Q, R such that A = Q @ R where Q is orthogonal and R is upper triangular
    """
    assert len(A.shape) == 2  # 2D matrix
    assert A.shape[0] >= A.shape[1]  # more rows than columns = more equations than variables

    n = A.shape[0]

    Q = np.identity(n)
    R = A.copy()

    for i in range(A.shape[1]):  # only up to number of columns
        a_i = R[i:, i]

        # build reflection for the subvector
        H_sub = reflection(a_i=a_i, i=0)  # local index starts at 0

        # embed into full matrix
        H_i = np.eye(n)
        H_i[i:, i:] = H_sub

        Q = Q @ H_i
        R = H_i @ R

    return Q, R


def inversion(A: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    returns the inverse of A using Householder decomposition
    """

    Q, R = decomposition(A)

    Q_inv = Q.T

    R_inv = solve_triangular(  # back substitute in O(n^2)
        R,
        np.identity(R.shape[0]),
        lower=False
    )

    return R_inv @ Q_inv

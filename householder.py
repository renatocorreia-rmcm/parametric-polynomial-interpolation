import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular


def decomposition(A: npt.NDArray[float]) -> (npt.NDArray[float], npt.NDArray[float]):
    """
    returns Q, R such that A = Q @ R where Q is orthogonal and R is upper triangular
    """
    assert len(A.shape) == 2  # 2D matrix
    assert A.shape[0] >= A.shape[1]  # more rows than columns = more equations than variables

    Q = np.identity(len(A))
    R = A.copy()

    for i in range(A.shape[1]):  # only up to number of columns
        a_i = R[i:, i]

        norm_a_i = np.linalg.norm(a_i)
        e_i = np.zeros(shape=a_i.shape)
        e_i[0] = 1

        sign_a_i = np.sign(a_i[0]) if a_i[0] != 0 else 1.0
        v_i = a_i + sign_a_i * norm_a_i * e_i

        u_i = v_i / np.linalg.norm(v_i)

        R[i:, i:] -= 2 * np.outer(u_i, u_i @ R[i:, i:])
        Q[:, i:] -= 2 * np.outer(Q[:, i:] @ u_i, u_i)

    return Q, R

def solve(QR, b):
    """
    QRx=b
    Rx=(Q^t)b  backsubstitute here
    """

    Q, R = QR

    return solve_triangular(R, Q.T @ b)


def solve_xy(A, x, y):
    """
    Decompose A once, solve both x and y

    Ax=b
    QRx=b
    Rx=(Q^t)b  backsubstitute here
    """

    QR = decomposition(A)
    coefficients_x = solve(QR=QR, b=x)
    coefficients_y = solve(QR=QR, b=y)

    return coefficients_x, coefficients_y

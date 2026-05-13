import numpy as np
import numpy.typing as npt


def generate_samples(
        ts: npt.NDArray[float],  # array of parameters
        relative_sample_rate: int,
        relative_extrapolation_rate: float,
):
    t_min = np.min(ts)
    t_max = np.max(ts)

    # need unique, sorted t values
    if len(np.unique(ts)) != len(ts):
        return None, False

    # set samples amount
    samples_amount: int = relative_sample_rate * (len(ts) - 1) + 1

    # set extrapolation
    t_span = t_max - t_min
    extrapolation_amount = relative_extrapolation_rate * t_span

    # set sample bounds
    t_start = t_min - extrapolation_amount
    t_end = t_max + extrapolation_amount

    # generate samples
    parameter_samples: npt.NDArray[float] = np.linspace(
        t_start,
        t_end,
        samples_amount,
        dtype=float
    )  # assume parameter is not necessairly monotonic  # if it was, could use only [0] and [-1] instead

    return parameter_samples


def sample_polynomials(
        parameter_samples: npt.NDArray[float],  # common array of parameters
        polynomials: npt.NDArray[float],  # arrays of coefficients for each polynomial
) -> npt.NDArray[float]:
    """
    sample multiple polynomials at the same parameters
    """

    # compute samples
    values_samples = np.polynomial.polynomial.polyval(
        x=parameter_samples,
        c=polynomials.T
    )

    # combine samples into points
    resampled_points: npt.NDArray[float] = np.column_stack(
        [parameter_samples, values_samples.T]
    )

    return resampled_points




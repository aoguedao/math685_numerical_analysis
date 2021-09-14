import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

images_path = Path("./images/hwk01")

# --- Exercise 5 ---

def log_taylor_approx(x, x0, n):
    """ Taylor expansion approximation of natural logarithm.

    Parameters
    ----------
    x : array_like
        Where is going to be evaluate.
    x0 : float
        Fixed point about expansion.
    n : int
        Expansion order.

    Returns
    -------
    array_like

    Raises
    ------
    AttributeError
        If order is non-negative or not int.
    """   
    
    if not isinstance(n, int) or (n < 0):
        raise AttributeError("order 'n' must be an non-negative integer")
    taylor_term = lambda x, x0, k: ((-1) ** (k -1) * ((x - x0) / x0) ** k) / k
    value = (
        np.log(x0)
        + np.sum([taylor_term(x, x0, k) for k in range(1, n + 1)], axis=0)
    )
    return value


def relative_error(real, pred):
    """ Computes relative error

    Parameters
    ----------
    real : array_like
        Real value
    pred : arra_like
        Predicted value

    Returns
    -------
    array_like
    """    
    return np.abs((pred - real) / real)
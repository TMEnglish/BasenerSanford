# ASSUME: import numpy as np
# ASSUME: from mpmath import mp
import math
from os import mkdir
    

# Make some mpmath functions into quasi-ufuncs taking either scalar or
# array arguments.
#
mp_float = np.frompyfunc(mp.mpf, 1, 1)
erf = np.frompyfunc(mp.erf, 1, 1)
erfc = np.frompyfunc(mp.erfc, 1, 1)


def fsum(a):
    """
    Returns an accurate sum of elements of `a`.
    
    If `a[0]` is a multiprecision float, then `mpmath.fsum(a)` is
    returned. Otherwise, `math.fsum(a)` is returned.
    """
    if type(a[0]) is mp.mpf:
        return mp.fsum(a)
    return math.fsum(a)


def linspace(a, b, n):
    """
    Tries to return numpy.linspace(); falls back on mpmath version.
    """
    try:
        return np.linspace(a, b, n)
    except:
        return np.array(mp.linspace(a, b, n))


def bias_exponents(array, max_exponent, current_max=None):
    """
    Scales elements in `array` by an integer power of 2.

    On return, the internal representation of the maximum element of
    `array` has `max_exponent` as its exponent. The given `array` is
    returned.
    """
    if current_max is None:
        _, current_max = math.frexp(array.max())
    array *= 2.0 ** (max_exponent - current_max)
    return array


def mean_var(frequency, x):
    """
    Returns mean and variance for `frequency` distribution over `x`.
    
    Sums are calculated accurately using `fsum()`.
    """
    if type(frequency[0]) is mp.mpf:
        if not type(x[0]) is mp.mpf:
            x = mp_float(x)
    elif type(x[0]) is mp.mpf:
        frequency = mp_float(frequency)
    norm = fsum(frequency)
    mom1 = fsum(frequency * x) 
    mom2 = fsum(frequency * x**2)
    var = (mom2 - mom1**2 / norm) / norm
    mean = mom1 / norm
    return mean, var
   

def relative_error(actual, desired):
    """
    Returns `(actual - desired) / desired`, with 0/0 treated as 0.
    """
    if np.shape(actual) != np.shape(desired):
        raise ValueError('Arguments are not identical in shape')
    if not isinstance(desired, np.ndarray):
        desired = np.array(desired)
    result = np.subtract(actual, desired)
    unequal = result != 0
    undefined = np.logical_and(unequal, desired == 0)
    defined = np.logical_and(unequal, desired != 0)
    result[undefined] = np.inf
    result[defined] /= desired[defined]
    return result


def trim(array, threshold):
    """
    Zeroes sub-`threshold` elements at left and right ends of `array`.
    
    Returns pair `(left, right)` indicating how many elements were
    zeroed at the left and right ends of the given 1-D `array`.
    """
    zero = array[0] * 0
    last = len(array) - 1
    left = 0
    while left <= last and array[left] < threshold:
        array[left] = zero
        left += 1
    right = last
    while right >= 0 and array[right] < threshold:
        array[right] = zero
        right -= 1
    return left, last - right


def slice_to_support(p):
    """
    Returns a slice excluding zeros in the tails of distribution `p`.
    
    Assumption: At least one element of `p` is nonzero.
    """
    w, = np.nonzero(p)
    return slice(w[0], w[-1] + 1)


def ensure_directory_exists(path):
    """
    Create directory with given `path` if it does not exist already.
    """
    try:
        mkdir(path)
    except FileExistsError:
        pass
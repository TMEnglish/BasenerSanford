import math
    
# Make some mpmath functions into quasi-ufuncs taking either
# scalar or array arguments.
#
mp_float = np.frompyfunc(mp.mpf, 1, 1)
erfc = np.frompyfunc(mp.erfc, 1, 1)


def fsum(a):
    """
    Returns an accurate sum of elements of iterable `a`.
    
    Iterable `a` is converted to a NumPy array if it is not one 
    already. The array must be one-dimensional. If the array contains
    links to objects, then the array is assumed to contain a link to at
    least one multiprecision float, and the multiprecision sum is
    calculated using `mpmath.fsum(a)`. Otherwise, the 64-bit floating-
    point sum is calculated by `math.fsum(a)`.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if a.dtype is np.dtype('O'):
        return mp.fsum(a)
    return math.fsum(a)


def bias_exponents(a, max_exponent):
    """
    Scales array `a` of floating-point numbers by an integer power of 2.
    
    Returns the integer power `n` of the scalar `2**n`.

    On return, the maximum element of `a` has `max_exponent` as its
    exponent. If, before and after the scaling operation, no element
    of `a` is subnormal (with leading zeros in its mantissa), then the
    mantissas of all elements are unchanged, and the operation is
    precisely invertible. The elements of `a` may be multiprecision
    floats.
    """
    a_max = np.max(a)
    basetype = type(a_max)
    unused_mantissa, current_max_exponent = mp.frexp(a_max)
    power = max_exponent - current_max_exponent
    if power != 0.0:
        a *= basetype(2.0)**power
    return power


def mean_var(frequency, x):
    """
    Returns mean and variance for `frequency` distribution over `x`.
    
    It is assumed that all elements of `frequency` are non-negative
    numbers (not necessarily integers). All calculations are performed
    with multiprecision floats. 
    """
    x = mp_float(x)
    frequency = mp_float(frequency)
    norm = mp.fsum(frequency)
    mean = mp.fsum(frequency * x) / norm
    var = mp.fsum(frequency * (x - mean)**2) / norm
    return mean, var
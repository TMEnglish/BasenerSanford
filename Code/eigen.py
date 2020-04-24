import numpy.random as random


def equilibrium(W, max_error=1e-14, maxiter=10**5):
    """
    Calculates dominant eigenvalue and eigenvector of matrix `W`.
    
    Returns 
        * the dominant eigenvalue of square array `W`,
        * an eigenvector `v` corresponding to the eigenvalue, and
        * the error `eig_error(W, v)` of the calculated eigenpair.
    
    The calculation proceeds until the error is no greater than
    `max_error`, or the number of iterations exceeds `max_iterations`.
    """
    # Create an instance of `Solver` with random initial frequencies
    # (all positive), step size of one year, and threshold relative
    # frequency of zero.
    solver = Solver(W, random.rand(len(W)) + 1e-15, 1, threshold=0)
    #
    # Calculate the error of the initial guess of the eigenvector.
    error, e_value = eigen_error(W, solver.s)
    #
    # Run the solver repeatedly, 1000 years per iteration, until the
    # solution converges to an eigenvector of `W` or the maximum of
    # iterations is exceeded.
    while error > max_error and len(solver) <= maxiter:
        solver(1000)
        error, e_value = eigen_error(W, solver.s)
    return e_value, solver.s, error


def eigen_error(W, v):
    """
    Evaluates an approximate eigenvector `v` of square matrix `W`.
    
    The corresponding approximate eigenvalue `e_value` is calculated.
    The error of the approximate eigenvector `v` of `W` is returned,
    along with the corresponding approximate eigenvalue `e_value`. The
    error is the maximum absolute error of `e_value * v` relative to
    the matrix product of `W` and `v`.
    
    Assumption: The elements of the matrix product of `W` and `v` are
    all positive.
    """
    # Set `e_value` to the Rayleigh quotient R(W, v).
    product = W @ v
    e_value = np.dot(v, product) / np.dot(v, v)
    #
    # Calculate the maximum absolute relative error, assuming that
    # elements of `product` are strictly positive. NumPy will issue a
    # warning if any of the elements are zero.
    relative_errors = (e_value * v - product) / product
    mare = np.max(np.abs(relative_errors))
    return mare, e_value
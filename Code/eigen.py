# %load ./Code/eigen.py
from scipy import linalg


def equilibrium(W, max_error=1e-14, maxiter=10**5):
    """
    Finds the equilibrium distribution for derivative operator `W`.
    
    Returns 
        * the largest real eigenvalue of square array `W`,
        * an eigenvector `v` corresponding to the eigenvalue, and
        * the error `eig_error(W, v)` of the calculated eigenpair.
    
    The equilibrium is initially approximated using a function in the
    SciPy package. Then the approximation is iteratively improved. The
    calculation proceeds until the error is no greater than `max_error`,
    or the number of iterations exceeds `max_iterations`.
    """
    # Get a first approximation of an eigenvector corresponding to the
    # largest real eigenvalue of `W`, using the `eig` function of 
    # SciPy's linear algebra package. The elements of the eigenvector
    # are complex numbers, possibly with negative real components. Thus
    # we replace the elements with their absolute values.
    e_values, e_vectors = linalg.eig(W)
    which = np.argmax(e_values.real)
    e_vector = np.abs(e_vectors[:,which])
    #
    # Create an instance of `Solver` with threshold relative frequency
    # of zero, step size of one year, and initial frequencies equal to
    # the first approximation of the eigenvector.
    solver = Solver(W, e_vector, 1, threshold=0)
    #
    # Run the solver repeatedly, 1000 iterations per run, until the
    # solution converges to an eigenvector of `W` or the maximum 
    # number of iterations is exceeded.
    error, e_value = eigen_error(W, solver.s)
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
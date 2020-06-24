from scipy import linalg
import warnings


def largest_real_eig(W):
    """
    Returns largest real eigenvalue of `W` and an associated eigenvector.
    
    The eigenpair is obtained using the `eig` function in the SciPy
    linear algebra package, which requires that square matrix `W`
    contain 64-bit floats. The largest-magnitude element of the returned
    eigenvector is positive.
    """
    # Use the `eig` function of SciPy's linear algebra package to obtain
    # all eigenvalues and eigenvectors of `W`. Ignore eigenvalues with
    # nonzero imaginary parts, and also their associated eigenvectors.
    e_values, e_vectors = linalg.eig(W)
    real = e_values.imag == 0
    e_values = e_values[real]
    e_vectors = e_vectors[:,real]
    #
    # Select the largest real eigenvalue and its associated eigenvector.
    largest = np.argmax(e_values.real)
    e_value = e_values[largest].real
    e_vector = e_vectors[:,largest].real
    #
    # Negate eigenvector if its largest-magnitude element is negative.
    max_mag = np.argmax(np.abs(e_vector))
    if e_vector[max_mag] < 0:
        e_vector = -e_vector
    return e_value, e_vector


def equilibrium(W, n_iterations=10):
    """
    Finds the equilibrium distribution for derivative operator `W`.
        
    Returns 
    * the largest real eigenvalue `e_value` of square array `W`,
    * the corresponding eigenvector `v` with elements summing to 1, and
    * the maximum absolute error of `e_value * v` relative to `W @ v`.
        
    The eigenpair is initially approximated using `largest_real_eig`.
    The approximation is subsequently improved by performing at most
    `n_iterations` iterations of the inverse power method. 
    """
    # Get a first approximation of the largest real eigenvalue and an
    # associated eigenvector.
    e_value, e_vector = largest_real_eig(W)
    #
    # All elements of the correct eigenvector are positive. Zero the 
    # negative elements of the approximate eigenvector, and recalculate
    # the eigenvalue, along with the approximation error.
    e_vector[e_vector < 0.0] = 0.0
    e_vector /= fsum(e_vector)
    best_e_vector = e_vector
    best_e_value, best_error = rayleigh_quotient(W, best_e_vector)
    #
    # Improve the approximate eigenpair by the inverse power method
    # (see Wikipedia, https://en.wikipedia.org/wiki/Inverse_iteration).
    # The method fails if subtracting the approximate eigenvalue from
    # the main diagonal of `W` produces a singular matrix. 
    A = np.array(W)
    diag_indices = np.diag_indices(A.shape[0])
    A[diag_indices] -= best_e_value
    for _ in range(n_iterations):
        # Solve for a better approximation of the eigenvector. Ignore
        # warnings about possible numerical inaccuracy, and quit the
        # iteration on exception (presumably due to singular `A`).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                e_vector = linalg.solve(A, e_vector)
            except:
                break
        # Calculate the associated eigenvalue and approximation error
        # for the new approximation of the eigenvector.
        e_vector /= fsum(e_vector)
        e_value, error = rayleigh_quotient(W, e_vector)
        #
        # If the approximation error is the least so far, then use the
        # eigenvalue to recalculate `A`.
        if error < best_error:
            best_error = error
            best_e_value = e_value
            best_e_vector = e_vector
            A[diag_indices] = W[diag_indices] - best_e_value
    return best_e_value, best_e_vector, best_error


def rayleigh_quotient(W, v):
    """
    Calculates the Rayleigh quotient of square matrix `W` and vector `v`.
    
    Returns
    * the Rayleigh quotient `e_value` (the approximate eigenvalue
      corresponding to approximation `v` of an eigenvector of `W`)
    * the maximum absolute error of `e_value * v` relative to `W @ v`,
      the matrix product of `W` and `v`
    
    Assumption: The elements of `W @ v` are all nonzero.
    """
    # Numerical accuracy is crucial, so we calculate the vector dot
    # product using the stable `fsum` to sum the elements of the
    # pointwise product of two vectors.
    def dot(u, v):
        return fsum(u * v)
    #
    # The Rayleigh quotient is (v.T @ W @ v) / (v.T @ v). 
    Wv_product = np.array([dot(row, v) for row in W])
    e_value = dot(v, Wv_product) / dot(v, v)
    #
    # Calculate the maximum absolute relative error, assuming that all
    # elements of `Wv_product` are nonzero. NumPy will issue a warning
    # if any of the elements are zero.
    relative_errors = (e_value*v - Wv_product) / Wv_product
    mare = np.max(np.abs(relative_errors))
    return e_value, mare
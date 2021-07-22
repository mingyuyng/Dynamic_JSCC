import numpy as np
import scipy.optimize

__all__ = ['nnls']

MAX_MEM_BLOCK = 2**8 * 2**10
def _nnls_obj(x, shape, A, B):
    '''Compute the objective and gradient for NNLS'''

    # Scipy's lbfgs flattens all arrays, so we first reshape
    # the iterate x
    x = x.reshape(shape)

    # Compute the difference matrix
    diff = np.dot(A, x) - B

    # Compute the objective value
    value = 0.5 * np.sum(diff ** 2)

    # And the gradient
    grad = np.dot(A.T, diff)

    # Flatten the gradient
    return value, grad.flatten()


def _nnls_lbfgs_block(A, B, x_init=None, **kwargs):
    '''Solve the constrained problem over a single block
    Parameters
    ----------
    A : np.ndarray [shape=(m, d)]
        The basis matrix
    B : np.ndarray [shape=(m, N)]
        The regression targets
    x_init : np.ndarray [shape=(d, N)]
        An initial guess
    kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`
    Returns
    -------
    x : np.ndarray [shape=(d, N)]
        Non-negative matrix such that Ax ~= B
    '''

    # If we don't have an initial point, start at the projected
    # least squares solution
    if x_init is None:
        x_init = np.linalg.lstsq(A, B, rcond=None)[0]
        np.clip(x_init, 0, None, out=x_init)

    # Adapt the hessian approximation to the dimension of the problem
    kwargs.setdefault('m', A.shape[1])

    # Construct non-negative bounds
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape

    # optimize
    x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(_nnls_obj, x_init,
                                                             args=(shape, A, B),
                                                             bounds=bounds,
                                                             **kwargs)
    # reshape the solution
    return x.reshape(shape)


def nnls(A, B, **kwargs):
    '''Non-negative least squares.
    Given two matrices A and B, find a non-negative matrix X
    that minimizes the sum squared error:
        err(X) = sum_i,j ((AX)[i,j] - B[i, j])^2
    Parameters
    ----------
    A : np.ndarray [shape=(m, n)]
        The basis matrix
    B : np.ndarray [shape=(m, N)]
        The target matrix.
    kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`
    Returns
    -------
    X : np.ndarray [shape=(n, N), non-negative]
        A minimizing solution to |AX - B|^2
    See Also
    --------
    scipy.optimize.nnls
    scipy.optimize.fmin_l_bfgs_b
    Examples
    --------
    Approximate a magnitude spectrum from its mel spectrogram

    '''

    # If B is a single vector, punt up to the scipy method
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]

    n_columns = int(MAX_MEM_BLOCK // (A.shape[-1] * A.itemsize))

    # Process in blocks:
    if B.shape[-1] <= n_columns:
        return _nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)

    x = np.linalg.lstsq(A, B, rcond=None)[0].astype(A.dtype)
    np.clip(x, 0, None, out=x)
    x_init = x

    for bl_s in range(0, x.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, B.shape[-1])
        x[:, bl_s:bl_t] = _nnls_lbfgs_block(A, B[:, bl_s:bl_t],
                                            x_init=x_init[:, bl_s:bl_t],
                                            **kwargs)
    return x
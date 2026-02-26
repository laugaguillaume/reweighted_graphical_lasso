from benchopt.utils.safe_import import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from numba import njit

    from benchmark_utils.utils import loss, neg_llh


class GraphicalIsta():
    def __init__(self,
                 alpha=1.,
                 gamma_max=1.,
                 back_track_const=0.9,
                 max_back_track=100,
                 max_iter=100,
                 tol=1e-8):

        self.alpha = alpha
        self.gamma_max = gamma_max
        self.back_track_const = back_track_const
        self.max_back_track = max_back_track
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, S):

        p = S.shape[0]

        W = S.copy()
        W *= 0.95
        diagonal = S.flat[:: p + 1]
        W.flat[:: p + 1] = diagonal
        Theta = np.linalg.pinv(W)

        Theta, W = gista_fit(Theta,
                             W,
                             S,
                             self.alpha,
                             self.gamma_max,
                             self.back_track_const,
                             self.max_back_track,
                             self.max_iter)

        self.precision_, self.covariance_ = Theta, W
        return self


@njit
def gista_fit(Theta, W, S, alpha, gamma_max, back_track_const, max_back_track, max_iter):

    gamma = gamma_max
    for it in range(max_iter):
        Theta, W, gamma = line_search(
            Theta, S, W, gamma, alpha, gamma_max, back_track_const, max_back_track)
        # print(gamma)
    # else:
    #     print(
    #         f"Not converged at epoch {it + 1}, "
    #         f"diff={norm(Theta - Theta_old):.2e}"
    #     )

    return Theta, W


@njit
def line_search(Theta, S, W, gamma, alpha, gamma_max, back_track_const, max_back_track):
    """ Perform backtracking line-search and return correct Theta_next"""

    for back_track in range(max_back_track):
        Theta_next = gista_iter(Theta, S, W, gamma, alpha)

        try:
            L = np.linalg.cholesky(Theta_next)
            L_inv = np.linalg.solve(L, np.eye(L.shape[0]))
        # except np.linalg.LinAlgError: # not supported by numba
        except:
            gamma *= back_track_const
            continue
        if neg_llh(Theta_next, S) > quad_approx(Theta_next, Theta, W, S, gamma):
            gamma *= back_track_const
            continue

        # Use cholesky to compute the inverse and the loss, instead
        W_next = L_inv.T @ L_inv
        gamma = compute_gamma_init(Theta_next, Theta, W_next, W)

        Theta = Theta_next
        W = W_next
        return Theta, W, gamma

    else:
        print(
            f"Reached max backtracking iters {back_track}, taking safe step-size.")
        gamma_safe = np.linalg.eigvalsh(
            Theta).min()**2  # This can be sped-up ?
        Theta = gista_iter(Theta, S, W, gamma_safe, alpha)
        gamma = gamma_max
        # This can be computed with Cholesky also ?
        W = np.linalg.pinv(Theta)

    return Theta, W, gamma


@njit
def gista_iter(Theta, S, W, gamma, alpha):
    """ An iteration of GISTA """
    return ST_off_diag(Theta - gamma*(S - W), alpha*gamma)


@njit
def compute_gamma_init(Theta_next, Theta, W_next, W):
    """ Compute Barzilai-Borwein step """
    trace_num = ((Theta_next - Theta)**2).sum()
    trace_denom = ((Theta_next - Theta) * (W - W_next)).sum()
    return trace_num/(trace_denom + 1e-12)


@njit
def quad_approx(Theta_next, Theta, W, S, gamma):
    """ Quadratic approximation of loss around current iterate"""

    Q = (neg_llh(Theta, S) +
         ((Theta_next - Theta) * (S - W)).sum() +
         np.linalg.norm(Theta_next - Theta)**2 / (2*gamma + 1e-12))

    return Q


@njit
def ST_off_diag(x, tau):
    off_diag = np.sign(x) * np.maximum(np.abs(x) - tau, 0)
    diag = np.diag(x)
    np.fill_diagonal(off_diag, diag)
    return off_diag

from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from sklearn.utils.validation import check_random_state
    from sklearn.linear_model import _cd_fast as cd_fast

    from numba import njit
    import scipy


class GraphicalLasso():
    def __init__(self,
                 alpha=1.,
                 weights=None,
                 algo="dual",
                 pen="l1",
                 inner_anderson=False,
                 outer_anderson=False,
                 max_iter=100,
                 tol=1e-8,
                 warm_start=False,
                 inner_tol=1e-4,
                 ):
        self.alpha = alpha
        self.weights = weights
        self.algo = algo
        self.pen = pen,
        self.inner_anderson = inner_anderson
        self.outer_anderson = outer_anderson
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.inner_tol = inner_tol

    def fit(self, S):
        p = S.shape[-1]
        indices = np.arange(p)

        if self.weights is None:
            Weights = np.ones((p, p))
        else:
            Weights = self.weights
            if not np.allclose(Weights, Weights.T):
                raise ValueError("Weights should be symmetric.")

        if self.warm_start and hasattr(self, "precision_"):
            if self.algo == "dual":
                raise ValueError(
                    "dual does not support warm start for now.")
            Theta = self.precision_
            W = self.covariance_

        else:
            W = S.copy()
            W *= 0.95
            diagonal = S.flat[:: p + 1]
            W.flat[:: p + 1] = diagonal
            Theta = scipy.linalg.pinvh(W)

        W_11 = np.copy(W[1:, 1:], order="C")
        eps = np.finfo(np.float64).eps
        it = 0
        Theta_old = Theta.copy()
        for it in range(self.max_iter):
            Theta_old = Theta.copy()
            if self.outer_anderson:
                K = 4
                buffer_filler = 0
                anderson_mem = np.zeros(
                    (Theta.shape[0], Theta.shape[0], K+1))  # p x (p-1) x (K+1)
            for col in range(p):
                if self.algo == "primal":
                    indices_minus_col = np.concatenate(
                        [indices[:col], indices[col + 1:]])
                    _11 = indices_minus_col[:, None], indices_minus_col[None]
                    _12 = indices_minus_col, col
                    _21 = col, indices_minus_col
                    _22 = col, col

                elif self.algo == "dual":
                    if col > 0:
                        di = col - 1
                        W_11[di] = W[di][indices != col]
                        W_11[:, di] = W[:, di][indices != col]
                    else:
                        W_11[:] = W[1:, 1:]

                s_12 = S[col, indices != col]

                # penalty.weights = Weights[_12]
                if self.algo == "dual":
                    beta_init = (Theta[indices != col, col] /
                                 (Theta[col, col] + 1000 * eps))
                    # Xw_init = W_11 @ w_init
                    Q = W_11

                elif self.algo == "primal":
                    inv_Theta_11 = (W[_11] -
                                    np.outer(W[_12],
                                             W[_12])/W[_22])
                    Q = inv_Theta_11
                    beta_init = Theta[indices != col, col] * S[col, col]
                else:
                    raise ValueError(f"Unsupported algo {self.algo}")

                beta = cd_gram(
                    Q,
                    s_12,
                    x=beta_init,
                    alpha=self.alpha,
                    weights=Weights[indices != col, col],
                    pen=self.pen,
                    anderson=self.inner_anderson,
                    anderson_buffer=4,
                    tol=self.inner_tol,
                    max_iter=self.max_iter,
                )

                if self.algo == "dual":
                    w_12 = -np.dot(W_11, beta)
                    W[col, indices != col] = w_12
                    W[indices != col, col] = w_12

                    Theta[col, col] = 1 / \
                        (W[col, col] + np.dot(beta, w_12))
                    Theta[indices != col, col] = beta*Theta[col, col]
                    Theta[col, indices != col] = beta*Theta[col, col]

                else:  # primal
                    Theta[indices != col, col] = beta / S[col, col]
                    Theta[col, indices != col] = beta / S[col, col]
                    Theta[col, col] = (1/S[col, col] +
                                       Theta[col, indices != col] @
                                       inv_Theta_11 @
                                       Theta[indices != col, col])
                    W[col, col] = (1/(Theta[col, col] -
                                      Theta[indices != col, col] @
                                      inv_Theta_11 @
                                      Theta[indices != col, col]))
                    W[indices != col, col] = (-W[col, col] *
                                              inv_Theta_11 @
                                              Theta[indices != col, col])
                    W[col, indices != col] = (-W[col, col] *
                                              inv_Theta_11 @
                                              Theta[indices != col, col])
                    # Maybe W_11 can be done smarter ?
                    W[_11] = (inv_Theta_11 +
                              np.outer(W[indices != col, col],
                                       W[indices != col, col])/W[col, col])

            if self.outer_anderson:
                if buffer_filler <= K:
                    if self.algo == "dual":
                        anderson_mem[:, :, buffer_filler] = W
                    elif self.algo == "primal":
                        anderson_mem[:, :, buffer_filler] = Theta
                    buffer_filler += 1
                else:
                    try:
                        U = np.diff(anderson_mem)
                        c = np.linalg.solve(np.dot(U.T, U), np.ones(K))
                        C = c / np.sum(c)
                        if self.algo == "dual":
                            W = np.dot(np.ascontiguousarray(
                                anderson_mem[:, :, 1:]), C)
                        elif self.algo == "primal":
                            Theta = np.dot(np.ascontiguousarray(
                                anderson_mem[:, :, 1:]), C)
                        buffer_filler = 0
                    except np.linalg.LinAlgError:
                        print(f"linalg err at iter {it}")
                        pass

            if norm(Theta - Theta_old) < self.tol:
                print(f"Weighted Glasso converged at CD epoch {it + 1}")
                break
        else:
            print(
                f"Not converged at epoch {it + 1}, "
                f"diff={norm(Theta - Theta_old):.2e}"
            )
        self.precision_, self.covariance_ = Theta, W
        self.n_iter_ = it + 1

        return self


@njit
def ST(x, tau):
    """ Prox of ell_1 penalty """
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


@njit
def prox_MCP(v, lambd, gamma):
    scale = 1.0 / (1.0 - 1.0 / gamma)
    if isinstance(v, float) or isinstance(v, int):
        abs_v = abs(v)
        if abs_v <= lambd:
            return 0.0
        elif abs_v <= gamma * lambd:
            return scale * np.sign(v) * (abs_v - lambd)
        else:
            return v
    else:
        prox = np.empty_like(v)
        for i in range(v.shape[0]):
            abs_vi = abs(v[i])
            if abs_vi <= lambd:
                prox[i] = 0.0
            elif abs_vi <= gamma * lambd:
                prox[i] = scale * np.sign(v[i]) * (abs_vi - lambd)
            else:
                prox[i] = v[i]
        return prox


@njit
def prox_SCAD(v, lambd, gamma=3.7, eta=1.0):
    scale = 1.0 / (1.0 - eta / (gamma - 1))

    if isinstance(v, float) or isinstance(v, int):
        abs_v = abs(v)
        sign_v = 1.0 if v >= 0 else -1.0

        if abs_v <= lambd:
            return 0.0
        elif abs_v <= gamma * lambd:
            return sign_v * (abs_v - eta * lambd) * scale
        else:
            return v
    else:
        prox = np.empty_like(v)
        for i in range(v.shape[0]):
            abs_vi = abs(v[i])
            sign_vi = 1.0 if v[i] >= 0 else -1.0

            if abs_vi <= lambd:
                prox[i] = 0.0
            elif abs_vi <= gamma * lambd:
                prox[i] = sign_vi * (abs_vi - eta * lambd) * scale
            else:
                prox[i] = v[i]
        return prox


@njit
def prox_l_05(x, u):
    t = (3./2.) * u ** (2./3.)
    if np.abs(x) < t:
        return 0.
    return x * (2./3.) * (1 + np.cos((2./3.) * np.arccos(
        -(3.**(3./2.)/4.) * u * np.abs(x)**(-3./2.))))


@njit
def _r2(x, alpha, eps):
    # compute r2 as in (eq. 7), ref [1] in `prox_log_sum`
    return (x - eps) / 2. + np.sqrt(((x + eps) ** 2) / 4 - alpha)


@njit
def _log_sum_prox_val(x, z, alpha, eps):
    # prox objective of log-sum `log(1 + abs(x) / eps)`
    return ((x - z) ** 2) / (2 * alpha) + np.log1p(np.abs(x) / eps)


@njit
def _r(x, alpha, eps):
    # compute r as defined in (eq. 9), ref [1] in `prox_log_sum`
    r_z = _log_sum_prox_val(_r2(x, alpha, eps), x, alpha, eps)
    r_0 = _log_sum_prox_val(0, x, alpha, eps)
    return r_z - r_0


@njit
def _find_root_by_bisection(a, b, alpha, eps, tol=1e-8):
    # find root of function func in interval [a, b] by bisection."""
    while b - a > tol:
        c = (a + b) / 2.
        if _r(a, alpha, eps) * _r(c, alpha, eps) < 0:
            b = c
        else:
            a = c
    return c


@njit
def prox_log_sum(x, alpha, eps):
    if np.sqrt(alpha) <= eps:
        if abs(x) <= alpha / eps:
            return 0.
        else:
            return np.sign(x) * _r2(abs(x), alpha, eps)
    else:
        a = 2 * np.sqrt(alpha) - eps
        b = alpha / eps
        # f is continuous and f(a) * f(b) < 0, the root can be found by bisection
        x_star = _find_root_by_bisection(a, b, alpha, eps)
        if abs(x) <= x_star:
            return 0.
        else:
            return np.sign(x) * _r2(abs(x), alpha, eps)


@njit
def cd_gram(H, q, x, alpha, weights, pen="l1",
            anderson=False, anderson_buffer=0, max_iter=100, tol=1e-4):
    """
    Solve min .5 * x.T H x + q.T @ x + alpha * norm(x, 1) with(out) extrapolation.

    H must be symmetric.
    """
    if anderson == True:
        K = anderson_buffer
        buffer_filler = 0
        anderson_mem = np.zeros((x.shape[0], K+1))

    dim = H.shape[0]
    lc = np.zeros(dim)
    for j in range(dim):
        lc[j] = H[j, j]

    # Hx = H @ x
    Hx = np.dot(H, x)
    for epoch in range(max_iter):
        max_delta = 0  # max coeff change

        for j in range(dim):
            x_j_prev = x[j]
            if pen[0] == "l1":
                x[j] = ST(x[j] - (Hx[j] + q[j]) / lc[j],
                          alpha*weights[j] / lc[j])

            elif pen[0] == "mcp":
                gamma = 3.
                x[j] = prox_MCP(x[j] - (Hx[j] + q[j]) / lc[j],
                                alpha*weights[j] / lc[j], gamma)

            elif pen[0] == "scad":
                x[j] = prox_SCAD(x[j] - (Hx[j] + q[j]) / lc[j],
                                 alpha*weights[j] / lc[j], gamma)

            elif pen[0] == "sqrt":
                x[j] = prox_l_05(x[j] - (Hx[j] + q[j]) / lc[j],
                                 alpha*weights[j] / lc[j])

            elif pen[0] == "log":
                eps = 1.
                x[j] = prox_log_sum(x[j] - (Hx[j] + q[j]) / lc[j],
                                    alpha*weights[j] / lc[j], eps=eps)

            else:
                print("Unknown penalty")

            max_delta = max(max_delta, np.abs(x_j_prev - x[j]))

            if x_j_prev != x[j]:
                Hx += (x[j] - x_j_prev) * H[j]
        # if max_delta <= tol:
        #     break

        if anderson:
            if buffer_filler <= K:
                anderson_mem[:, buffer_filler] = x
                buffer_filler += 1

            else:
                try:
                    U = np.diff(anderson_mem)
                    c = np.linalg.solve(np.dot(U.T, U), np.ones(K))
                    C = c / np.sum(c)
                    x = np.dot(np.ascontiguousarray(anderson_mem[:, 1:]), C)
                    buffer_filler = 0
                except:
                    # print(f"no accel at epoch {epoch}")
                    buffer_filler = 0
    return x

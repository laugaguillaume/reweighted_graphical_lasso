from benchopt.utils.safe_import import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numba import njit


class AdaptiveGistaSolver:
    def __init__(
        self,
        alpha=1.0,
        strategy="log",
        n_reweights=5,
        gamma_max=1.0,
        back_track_const=0.9,
        max_back_track=100,
        max_iter=100,
        tol=1e-8,
    ):
        self.alpha = alpha
        self.strategy = strategy
        self.n_reweights = n_reweights
        self.gamma_max = gamma_max
        self.back_track_const = back_track_const
        self.max_back_track = max_back_track
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, S, X=None):
        p = S.shape[0]

        W = S.copy()
        W *= 0.95
        diagonal = S.flat[:: p + 1]
        W.flat[:: p + 1] = diagonal
        Theta = np.linalg.pinv(W)

        for _ in range(self.n_reweights):
            if self.strategy == "log":
                weights = self.alpha / (np.abs(Theta) + 1e-10)
            elif self.strategy == "sqrt":
                weights = self.alpha / (2 * np.sqrt(np.abs(Theta)) + 1e-10)
            elif self.strategy == "mcp":
                gamma = 3.0
                weights = np.zeros_like(Theta)
                mask = np.abs(Theta) < gamma * self.alpha
                weights[mask] = self.alpha - np.abs(Theta[mask]) / gamma
            elif self.strategy == "scad":
                a = 3.7
                abs_theta = np.abs(Theta)
                weights = np.zeros_like(Theta)
                mask1 = abs_theta <= self.alpha
                weights[mask1] = self.alpha
                mask2 = (abs_theta > self.alpha) & (abs_theta <= a * self.alpha)
                weights[mask2] = (a * self.alpha - abs_theta[mask2]) / (a - 1)
            else:
                raise ValueError(f"Unknown strategy {self.strategy}")

            Theta, W = gista_fit_weighted(
                Theta,
                W,
                S,
                weights,
                self.gamma_max,
                self.back_track_const,
                self.max_back_track,
                self.max_iter,
            )

        self.precision_, self.covariance_ = Theta, W
        return self


@njit
def gista_fit_weighted(
    Theta,
    W,
    S,
    weights,
    gamma_max,
    back_track_const,
    max_back_track,
    max_iter,
):
    gamma = gamma_max
    for _ in range(max_iter):
        if (not np.isfinite(gamma)) or gamma <= 0:
            gamma = gamma_max
        Theta, W, gamma = line_search(
            Theta,
            S,
            W,
            gamma,
            weights,
            gamma_max,
            back_track_const,
            max_back_track,
        )

    return Theta, W


@njit
def line_search(
    Theta,
    S,
    W,
    gamma,
    weights,
    gamma_max,
    back_track_const,
    max_back_track,
):
    for back_track in range(max_back_track):
        Theta_next = gista_iter(Theta, S, W, gamma, weights)

        try:
            L = np.linalg.cholesky(Theta_next)
            L_inv = np.linalg.solve(L, np.eye(L.shape[0]))
        except:
            gamma *= back_track_const
            continue

        if neg_llh(Theta_next, S) > quad_approx(Theta_next, Theta, W, S, gamma):
            gamma *= back_track_const
            continue

        W_next = L_inv.T @ L_inv
        gamma = compute_gamma_init(Theta_next, Theta, W_next, W)
        if (not np.isfinite(gamma)) or gamma <= 0 or gamma > gamma_max:
            gamma = gamma_max

        Theta = Theta_next
        W = W_next
        return Theta, W, gamma

    print(f"Reached max backtracking iters {back_track}, taking safe step-size.")
    gamma_safe = np.linalg.eigvalsh(Theta).min() ** 2
    Theta = gista_iter(Theta, S, W, gamma_safe, weights)
    gamma = gamma_max
    W = np.linalg.pinv(Theta)

    return Theta, W, gamma


@njit
def gista_iter(Theta, S, W, gamma, weights):
    return ST_off_diag(Theta - gamma * (S - W), gamma * weights)


@njit
def compute_gamma_init(Theta_next, Theta, W_next, W):
    trace_num = ((Theta_next - Theta) ** 2).sum()
    trace_denom = ((Theta_next - Theta) * (W - W_next)).sum()
    return trace_num / (trace_denom + 1e-12)


@njit
def quad_approx(Theta_next, Theta, W, S, gamma):
    Q = (
        neg_llh(Theta, S)
        + ((Theta_next - Theta) * (S - W)).sum()
        + np.linalg.norm(Theta_next - Theta) ** 2 / (2 * gamma + 1e-12)
    )
    return Q


@njit
def neg_llh(Theta, S):
    return (-np.linalg.slogdet(Theta)[1] + (Theta * S).sum())


@njit
def ST_off_diag(x, tau):
    off_diag = np.sign(x) * np.maximum(np.abs(x) - tau, 0)
    diag = np.diag(x)
    np.fill_diagonal(off_diag, diag)
    return off_diag

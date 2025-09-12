from benchopt.utils import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from benchmark_utils.glasso_solver import GraphicalLasso
    import scipy


class AdaptiveGraphicalLasso():
    def __init__(
        self,
        alpha=1.,
        strategy="log",
        n_reweights=5,
        max_iter=1000,
        tol=1e-8,
        warm_start=False,
        # verbose=False,
    ):
        self.alpha = alpha
        self.strategy = strategy
        self.n_reweights = n_reweights
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, S):

        glasso = GraphicalLasso(
            alpha=self.alpha,
            algo="primal",
            max_iter=self.max_iter,
            tol=self.tol,
            warm_start=True)

        self.n_iter_ = []

        # Initialize Theta
        p = S.shape[-1]
        W = S.copy()
        W *= 0.95
        diagonal = S.flat[:: p + 1]
        W.flat[:: p + 1] = diagonal
        glasso.precision_ = scipy.linalg.pinvh(W)
        glasso.covariance_ = W.copy()

        for it in range(self.n_reweights):

            Theta = glasso.precision_  # initialiser le theta_init ici

            if self.strategy == "log":
                Weights = self.alpha / (np.abs(Theta) + 1e-10)

            elif self.strategy == "sqrt":
                Weights = self.alpha / (2*np.sqrt(np.abs(Theta)) + 1e-10)

            elif self.strategy == "mcp":
                gamma = 3.0
                Weights = np.zeros_like(Theta)
                mask = np.abs(Theta) < gamma * self.alpha
                Weights[mask] = self.alpha - np.abs(Theta[mask]) / gamma

            elif self.strategy == "scad":
                a = 3.7
                abs_theta = np.abs(Theta)
                Weights = np.zeros_like(Theta)

                # Region 1: |x| ≤ alpha
                mask1 = abs_theta <= self.alpha
                Weights[mask1] = self.alpha

                # Region 2: alpha < |x| ≤ a * alpha
                mask2 = (abs_theta > self.alpha) & (abs_theta <= a * self.alpha)
                Weights[mask2] = (a * self.alpha - abs_theta[mask2]) / (a - 1)

            else:
                raise ValueError(f"Unknown strategy {self.strategy}")

            glasso.weights = Weights
            glasso.fit(S)  # fit et weights descendent en dessous

            self.n_iter_.append(glasso.n_iter_)

            glasso.covariance_ = np.linalg.pinv(Theta, hermitian=True)

        self.precision_ = glasso.precision_
        self.covariance_ = glasso.covariance_
        return self

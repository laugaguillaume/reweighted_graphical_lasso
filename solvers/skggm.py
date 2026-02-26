from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    # You need to clone and install https://github.com/skggm/skggm
    from inverse_covariance import QuicGraphicalLasso


class Solver(BaseSolver):
    name = 'skggm'

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, S, alpha, X):
        self.S = S
        self.alpha = alpha
        self.X = X

        # sklearn doesnt' accept tolerance 0
        self.tol = 1e-18
        self.model = QuicGraphicalLasso(lam=self.alpha,
                                        mode="default",
                                        auto_scale=False,
                                        init_method="cov",
                                        tol=self.tol,
                                        )

    def run(self, n_iter):

        self.model.max_iter = n_iter
        self.model.fit(self.X)
        self.Theta = self.model.precision_
        self.W = self.model.covariance_
    def get_result(self):
        return dict(Theta=self.Theta,W = self.W)

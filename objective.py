from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import f1_score


class Objective(BaseObjective):

    name = "GLasso"

    url = "https://github.com/Perceptronium/benchmark_graphical_lasso"

    # alpha is the regularization hyperparameter
    parameters = {
        'reg': [0.5],
    }

    requirements = ["numpy", "scikit-learn", "numba"]

    min_benchopt_version = "1.5"

    def set_data(self, S, Theta_true, X):

        self.S = S
        self.Theta_true = Theta_true
        self.alpha = self.reg * self._get_alpha_max(S)
        self.X = X

    def evaluate_result(self, Theta):

        neg_llh = (-np.linalg.slogdet(Theta)[1] +
                   (Theta * self.S).sum())

        pen = self.alpha * np.sum(np.abs(Theta))
        # diagonal is not penalized:
        pen -= self.alpha * np.trace(np.abs(Theta))

        return dict(
            value=neg_llh + pen,
            neg_log_likelihood=neg_llh,
            penalty=pen,
            sparsity=1 - np.count_nonzero(Theta) / (Theta.shape[0]**2),
            NMSE=np.linalg.norm(Theta - self.Theta_true)**2 /
            np.linalg.norm(self.Theta_true)**2,
            F1_score=f1_score(Theta.flatten() != 0.,
                              self.Theta_true.flatten() != 0.)
        )

    def get_one_result(self):
        return dict(Theta=np.eye(self.S.shape[0]))

    def get_objective(self):
        return dict(
            S=self.S,
            alpha=self.alpha,
            X=self.X,
        )

    def _get_alpha_max(self, S):
        S_cpy = np.copy(S)
        np.fill_diagonal(S_cpy, 0.)
        return np.max(np.abs(S_cpy))

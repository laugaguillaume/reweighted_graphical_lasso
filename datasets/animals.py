from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "animals"

    parameters = {
        'random_state': [0],
    }

    # requirements = ["sklearn"]

    def get_data(self):
        X = np.loadtxt('./data/animals.txt', delimiter=',').T
        X = X-X.mean(0)  # center the data, check what work best

        S = np.cov(X, bias=True, rowvar=False)

        return dict(S=S,
                    Theta_true=np.eye(S.shape[0]),
                    X=X,
                    )

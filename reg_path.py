import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state

from benchmark_utils import GraphicalLasso
from benchmark_utils import AdaptiveGraphicalLasso

data_type = "synthetic"
# data_type = "animals"


def synthetic_data(p=100, n=1000, alpha=0.9):
    rng = check_random_state(0)
    Theta_true = make_sparse_spd_matrix(
        p,
        alpha=alpha,
        random_state=rng)

    Theta_true += 0.1*np.eye(p)
    Sigma_true = np.linalg.pinv(Theta_true, hermitian=True)
    X = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma_true,
        size=int(n*2),
    )

    X_train = X[:n]
    X_test = X[n:]

    S_train = np.cov(X_train, bias=True, rowvar=False)
    S_train_cpy = np.copy(S_train)
    np.fill_diagonal(S_train_cpy, 0.)
    alpha_max = np.max(np.abs(S_train_cpy))

    S_test = np.cov(X_test, bias=True, rowvar=False)

    return alpha_max, S_train, S_test, Theta_true


def animals_data():
    X = np.loadtxt('./data/animals.txt', delimiter=',').T

    train_size = int(2 * X.shape[0] / 3)
    X_train = X[:train_size]
    X_test = X[train_size:]

    X = X-X.mean(0)

    S_train = np.cov(X_train, bias=True, rowvar=False)
    S_cpy = np.copy(S_train)
    np.fill_diagonal(S_cpy, 0.)
    alpha_max = 2*np.max(np.abs(S_cpy))

    S_test = np.cov(X_test, bias=True, rowvar=False)

    return alpha_max, S_train, S_test, X_train.shape[1], train_size


if data_type == "synthetic":
    p = 100
    n = 1000
    sparsity = 0.9

    smallest_alpha_scaler = 1e-3

    grid_size = 30
    # grid_size = 5

    alpha_max, S_train, S_test, Theta_true = synthetic_data(p=p, n=n, alpha=sparsity)

    alphas = alpha_max*np.geomspace(1, smallest_alpha_scaler, num=grid_size)

elif data_type == "animals":
    alpha_max, S_train, S_test, p, n = animals_data()

    smallest_alpha_scaler = 1e-3

    sparsity = "Unknown"

    alphas = 5*alpha_max*np.geomspace(1, smallest_alpha_scaler, num=30)

else:
    raise "Unknown data type"

penalties = [
    "L1",
    "MCP",
    "SCAD",
    "L0_5",
    "Log",

    "L1_",
    "R-Log",
    "R-MCP",
    "R-SCAD",
    "R-L0.5",
]

n_reweights = 20

models_tol = 1e-4
models = [
    GraphicalLasso(algo="primal",
                   pen="l1",
                   warm_start=True,
                   tol=models_tol),

    GraphicalLasso(algo="primal",
                   pen="mcp",
                   warm_start=True,
                   tol=models_tol),

    GraphicalLasso(algo="primal",
                   pen="scad",
                   warm_start=True,
                   tol=models_tol),

    GraphicalLasso(algo="primal",
                   pen="sqrt",
                   warm_start=True,
                   tol=models_tol),

    GraphicalLasso(algo="primal",
                   pen="log",
                   warm_start=True,
                   tol=models_tol),
    #####
    GraphicalLasso(algo="primal",
                   pen="l1",
                   warm_start=True,
                   tol=models_tol),

    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="log",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=100),

    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="mcp",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=100),

    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="scad",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=100),

    AdaptiveGraphicalLasso(warm_start=True,
                           strategy="sqrt",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=100),
]

glasso_nmses = {penalty: [] for penalty in penalties}
glasso_f1_scores = {penalty: [] for penalty in penalties}
glasso_holdout_llhs = {penalty: [] for penalty in penalties}
glasso_sparsity_degrees = {penalty: [] for penalty in penalties}


for i, (penalty, model) in enumerate(zip(penalties, models)):
    print(penalty)
    for alpha_idx, alpha in enumerate(alphas):
        print(f"======= alpha {alpha_idx+1}/{len(alphas)} =======")

        model.alpha = alpha
        model.fit(S_train)

        Theta = model.precision_

        if data_type == "synthetic":
            nmse = norm(Theta - Theta_true)**2 / norm(Theta_true)**2
            f1 = f1_score(Theta.flatten() != 0.,
                          Theta_true.flatten() != 0.)
            print(f"NMSE: {nmse:.3f}")
            print(f"F1  : {f1:.3f}")
            glasso_nmses[penalty].append(nmse)
            glasso_f1_scores[penalty].append(f1)

        else:
            sparsity_degree = 1 - (np.count_nonzero(Theta) - p) / (p**2)
            glasso_sparsity_degrees[penalty].append(sparsity_degree)

        holdout_likelihood = np.linalg.slogdet(Theta)[1] - (Theta * S_test).sum()
        glasso_holdout_llhs[penalty].append(holdout_likelihood)


plt.close('all')
if data_type == "synthetic":
    fig, ax = plt.subplots(3, 2, sharex=True,
                           figsize=([9.7, 8.29]),
                           layout="constrained")
else:
    fig, ax = plt.subplots(2, 2, sharex=True,
                           figsize=([8.22, 6.85]),
                           layout="constrained")

for i, penalty in enumerate(penalties):

    if penalty[:2] == "L1":
        linestyle = "solid"
        marker = 'X' if penalty[-1] == "_" else 'd'
    elif penalty[0] == "R":
        linestyle = "solid"
        marker = 'X'
    else:
        linestyle = "solid"
        marker = 'd'

    if penalty[:2] == "L1":
        color = "k"
        label = r"$\ell_1$"
    elif (penalty == "MCP") or (penalty == "R-MCP"):
        color = "tab:orange"
        label = "MCP"
    elif (penalty == "SCAD") or (penalty == "R-SCAD"):
        color = "tab:green"
        label = "SCAD"
    elif (penalty == "L0_5") or (penalty == "R-L0.5"):
        color = "tab:red"
        label = r"$\ell_{0.5}$"
    elif (penalty == "R-Log") or (penalty == "Log"):
        color = "tab:purple"
        label = "log"

    if data_type == "synthetic":
        col = 1 if ((penalty[0] == "R") or (penalty == "L1_")) else 0

        # LLH
        ax[0, col].semilogx(alphas/alpha_max,
                            glasso_holdout_llhs[penalty],
                            linewidth=2.,
                            color=color,
                            linestyle=linestyle)
        max_llh = np.argmax(glasso_holdout_llhs[penalty])
        ax[0, col].vlines(
            x=alphas[max_llh] / alphas[0],
            ymin=np.min(glasso_holdout_llhs[penalty]),
            ymax=np.max(glasso_holdout_llhs[penalty]),
            linestyle="dotted",
            alpha=0.5,
            color=color)
        line2 = ax[0, col].plot(
            [alphas[max_llh] / alphas[0]],
            np.min(glasso_holdout_llhs[penalty]),
            clip_on=False,
            marker=marker,
            alpha=0.5,
            markersize=12,
            color=color)

        # NMSE
        ax[1, col].semilogx(alphas/alpha_max,
                            glasso_nmses[penalty],
                            color=color,
                            linewidth=2.,
                            label=label,
                            linestyle=linestyle)
        min_nmse = np.argmin(glasso_nmses[penalty])
        ax[1, col].vlines(
            x=alphas[min_nmse] / alphas[0],
            ymin=0,
            ymax=np.min(glasso_nmses[penalty]),
            linestyle="dotted",
            alpha=0.5,
            color=color)
        line0 = ax[1, col].plot(
            [alphas[min_nmse] / alphas[0]],
            0,
            clip_on=False,
            marker=marker,
            alpha=0.5,
            linestyle=linestyle,
            color=color,
            markersize=12)

        # F1 score
        ax[2, col].semilogx(alphas/alpha_max,
                            glasso_f1_scores[penalty],
                            linewidth=2.,
                            color=color,
                            linestyle=linestyle)
        max_f1 = np.argmax(glasso_f1_scores[penalty])
        ax[2, col].vlines(
            x=alphas[max_f1] / alphas[0],
            ymin=0,
            ymax=np.max(glasso_f1_scores[penalty]),
            linestyle="dotted",
            alpha=0.5,
            color=color)
        line1 = ax[2, col].plot(
            [alphas[max_f1] / alphas[0]],
            0,
            clip_on=False,
            marker=marker,
            alpha=0.5,
            markersize=12,
            color=color)

    elif data_type == "animals":
        col = 1 if ((penalty[0] == "R") or (penalty == "L1_")) else 0

        ax[0, col].semilogx(alphas/alphas[0],
                            glasso_holdout_llhs[penalty],
                            linewidth=2.,
                            color=color,
                            label=label,
                            linestyle=linestyle)
        max_llh = np.argmax(glasso_holdout_llhs[penalty])
        ax[0, col].vlines(
            x=alphas[max_llh] / alphas[0],
            ymin=np.min(list(glasso_holdout_llhs.values()))-10,
            ymax=np.max(glasso_holdout_llhs[penalty]),
            linestyle="dotted",
            alpha=0.5,
            color=color)
        line0 = ax[0, col].plot(
            [alphas[max_llh] / alphas[0]],
            np.min(list(glasso_holdout_llhs.values())),
            clip_on=False,
            marker=marker,
            alpha=0.5,
            markersize=12,
            color=color)

        ax[1, col].plot(alphas/alphas[0],
                        glasso_sparsity_degrees[penalty],
                        linewidth=2.,
                        color=color,
                        linestyle=linestyle)
        max_sparsity = np.argmax(glasso_sparsity_degrees[penalty])
        ax[1, col].vlines(
            x=alphas[max_llh] / alphas[0],
            ymin=-1.,
            ymax=2.,
            linestyle="dotted",
            alpha=0.5,
            color=color)

if data_type == "synthetic":
    for col in range(2):
        strategy = "Proximal" if col == 0 else "Reweighting"
        ax[0, col].set_title(f"{p=}, {n=}\n{strategy}", fontsize=18)
        ax[0, col].grid(which='both', alpha=0.9)

        ax[1, col].legend(fontsize=14, ncol=1)
        ax[1, col].grid(which='both', alpha=0.9)
        ax[2, col].set_xlabel(r"$\lambda / \lambda_\mathrm{{max}}$",  fontsize=18)
        ax[2, col].grid(which='both', alpha=0.9)

    for row in range(2):
        ax[row, 1].set_yticklabels([])

    ax[0, 0].set_ylabel(
        r"$\log \det(\hat{{\mathbf{{\Theta}}}}) - \langle \mathbf{S}_{\text{test}}, \hat{{\mathbf{{\Theta}}}} \rangle $", fontsize=12)
    ax[1, 0].set_ylabel("NMSE", fontsize=18)
    ax[2, 0].set_ylabel("F1 score", fontsize=18)

elif data_type == "animals":
    for col in range(2):
        strategy = "Proximal" if col == 0 else "Reweighting"
        ax[0, col].set_title(f"{p=}, {n=}\n{strategy}", fontsize=18)
        ax[0, col].grid(which='both', alpha=0.9)
        ax[0, col].set_ylim([np.min(list(glasso_holdout_llhs.values())),
                             np.max(list(glasso_holdout_llhs.values())) + 3])

        ax[1, col].set_ylim([0, 1])
        ax[1, col].grid(which='both', alpha=0.9)
        ax[1, col].set_xlabel(r"$\lambda / \lambda_\mathrm{{max}}$",  fontsize=18)

    ax[0, col].legend(fontsize=14, ncol=1)
    for row in range(2):
        ax[row, 1].set_yticklabels([])

    ax[0, 0].set_ylabel(
        r"$\log \det(\hat{{\mathbf{{\Theta}}}}) - \langle \mathbf{S}_{\text{test}}, \hat{{\mathbf{{\Theta}}}} \rangle $", fontsize=12)
    ax[0, 0].set_ylabel(
        r"$\log \det(\hat{{\mathbf{{\Theta}}}}) - \langle \mathbf{S}_{\text{test}}, \hat{{\mathbf{{\Theta}}}} \rangle $", fontsize=12)
    ax[1, 0].set_ylabel(
        f"Sparsity degree", fontsize=12)

plt.savefig(
    f"./reg_path_results/p{p}_n{n}_alpha{sparsity}_{data_type}_corrected_again.pdf")
plt.show()

for penalty in penalties:
    f1_at_optimal_heldout = glasso_f1_scores[penalty][np.argmax(
        glasso_holdout_llhs[penalty])]

    nmse_at_optimal_heldout = glasso_nmses[penalty][np.argmax(
        glasso_holdout_llhs[penalty])]

    max_llh = np.argmax(glasso_holdout_llhs[penalty])
    max_f1 = np.argmax(glasso_f1_scores[penalty])

    # discrepancy = np.abs((alphas[max_f1] - alphas[max_llh]) / alphas[0])
    print(f"{penalty}: F1 score irl: {f1_at_optimal_heldout:.2f}, NMSE irl: {nmse_at_optimal_heldout:.2f}")

to_save = [glasso_nmses,
           glasso_f1_scores,
           glasso_holdout_llhs]

for i, elem in enumerate(to_save):
    np.save(f"./reg_path_results/p{p}_{i}_{data_type}_corrected_again.npy", elem)

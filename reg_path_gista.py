import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import check_random_state

from benchmark_utils import GraphicalLasso
from benchmark_utils import gista_solver
from benchmark_utils import AdaptiveGistaSolver
import time
import pandas as pd
import argparse


data_type = "synthetic"
parser = argparse.ArgumentParser()
parser.add_argument("--max_iter_reweights", type=int, default = 10)
args = parser.parse_args()

max_iter_reweights = args.max_iter_reweights
def synthetic_data(p=50, n=1000, alpha=0.9):
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

# Synthetic data parameters
p = 75
n = 1000
sparsity = 0.9

# Grid search parameters
smallest_alpha_scaler = 1e-3
grid_size = 20
number_of_runs = 8

# Generate synthetic data
alpha_max, S_train, S_test, Theta_true = synthetic_data(p=p, n=n, alpha=sparsity)
alphas = alpha_max*np.geomspace(1, smallest_alpha_scaler, num=grid_size)

# Number of reweights
n_reweights = 20

penalties = [
    # "L1",
    # "MCP",
    # "SCAD",
    # "L0_5",
    # "Log",

    "L1_",
    "R-Log",
    "R-MCP",
    # "R-SCAD",
    "R-L0.5",
]


models_tol = 1e-4
models = [
    # GraphicalLasso(algo="primal",
    #                pen="l1",
    #                warm_start=True,
    #                tol=models_tol),

    # GraphicalLasso(algo="primal",
    #                pen="mcp",
    #                warm_start=True,
    #                tol=models_tol),

    # GraphicalLasso(algo="primal",
    #                pen="scad",
    #                warm_start=True,
    #                tol=models_tol),

    # GraphicalLasso(algo="primal",
    #                pen="sqrt",
    #                warm_start=True,
    #                tol=models_tol),

    # GraphicalLasso(algo="primal",
    #                pen="log",
    #                warm_start=True,
    #                tol=models_tol),
    #####
    GraphicalLasso(algo="primal",
                   pen="l1",
                   warm_start=True,
                   tol=models_tol),

    AdaptiveGistaSolver(strategy="log",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=max_iter_reweights),

    AdaptiveGistaSolver(strategy="mcp",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=max_iter_reweights),

    # AdaptiveGistaSolver(strategy="scad",
    #                        n_reweights=n_reweights,
    #                        tol=models_tol,
    #                        max_iter=max_iter_reweights),

    AdaptiveGistaSolver(strategy="sqrt",
                           n_reweights=n_reweights,
                           tol=models_tol,
                           max_iter=max_iter_reweights),
]

glasso_fit_times = {penalty: [] for penalty in penalties}
glasso_nmses = {penalty: [] for penalty in penalties}
glasso_f1_scores = {penalty: [] for penalty in penalties}
glasso_holdout_llhs = {penalty: [] for penalty in penalties}
glasso_sparsity_degrees = {penalty: [] for penalty in penalties}


for i, (penalty, model) in enumerate(zip(penalties, models)):
    print(penalty)
    for alpha_idx, alpha in enumerate(alphas):
        print(f"======= alpha {alpha_idx+1}/{len(alphas)} =======")

        model.alpha = alpha
        fit_times = []
        for k in range(number_of_runs):
            t0 = time.perf_counter()
            model.fit(S_train)
            t1 = time.perf_counter()
            fit_times.append(t1-t0)

        Theta = model.precision_

        if data_type == "synthetic":
            med_run_time = np.median(fit_times[1:])
            nmse = norm(Theta - Theta_true)**2 / norm(Theta_true)**2
            f1 = f1_score(Theta.flatten() != 0.,
                          Theta_true.flatten() != 0.)
            print(f"NMSE: {nmse:.3f}")
            print(f"F1  : {f1:.3f}")
            print(f"Median run time : {med_run_time}")
            glasso_nmses[penalty].append(nmse)
            glasso_f1_scores[penalty].append(f1)
            glasso_fit_times[penalty].append(med_run_time)

        else:
            sparsity_degree = 1 - (np.count_nonzero(Theta) - p) / (p**2)
            glasso_sparsity_degrees[penalty].append(sparsity_degree)

        holdout_likelihood = np.linalg.slogdet(Theta)[1] - (Theta * S_test).sum()
        glasso_holdout_llhs[penalty].append(holdout_likelihood)


plt.close('all')
if data_type == "synthetic":
    fig, ax = plt.subplots(4, 2, sharex=True,
                           figsize=([9.7, 8.29]),
                           layout="constrained")
else:
    fig, ax = plt.subplots(2, 2, sharex=True,
                           figsize=([8.22, 6.85]),
                           layout="constrained")
csv_rows= []
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
        # Median run times
        ax[3, col].semilogx(alphas/alpha_max,
                            glasso_fit_times[penalty],
                            linewidth=2.,
                            color=color,
                            linestyle=linestyle)
        min_time = np.argmin(glasso_fit_times[penalty])
        ax[3, col].vlines(
            x=alphas[min_time] / alphas[0],
            ymin=0,
            ymax=np.max(glasso_fit_times[penalty]),
            linestyle="dotted",
            alpha=0.5,
            color=color)
        line1 = ax[3, col].plot(
            [alphas[min_time] / alphas[0]],
            0,
            clip_on=False,
            marker=marker,
            alpha=0.5,
            markersize=12,
            color=color)
        csv_rows.append({
            "penalty": penalty,
            "llh_run_time": float(glasso_fit_times[penalty][int(max_llh)]),
            "max_llh": float(glasso_holdout_llhs[penalty][int(max_llh)]),
            "lambda_at_max_llh": float(alphas[int(max_llh)]),
            "mmse_run_time": float(glasso_fit_times[penalty][int(min_nmse)]),
            "min_nmse": float(glasso_nmses[penalty][int(min_nmse)]),
            "lambda_at_min_nmse": float(alphas[int(min_nmse)]),
            "f1_score_run_time": float(glasso_fit_times[penalty][int(max_f1)]),
            "max_f1": float(glasso_f1_scores[penalty][int(max_f1)]),
            "lambda_at_max_f1_score": float(alphas[int(max_f1)]),
        })
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
    f"./bench_results/gista/p{p}_n{n}_alpha{sparsity}_{data_type}_max_iter_reweights_{max_iter_reweights}_gridsize_{grid_size}_.pdf")
plt.show(block=False)
pd.DataFrame(csv_rows).to_csv(f"./bench_results/gista/p{p}_n{n}_alpha{sparsity}_{data_type}_max_iter_reweights_{max_iter_reweights}_gridsize_{grid_size}_glasso_summary.csv", index=False)

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

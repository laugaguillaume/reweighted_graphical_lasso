import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASEDIR = Path(__file__).resolve().parent / "bench_results" / "pglasso"
EXCLUDE = {"L1", "MCP", "SCAD", "L0_5", "Log", "R-SCAD"}  # penalties to drop

files = sorted(BASEDIR.glob("p75_*_alpha0.9_*_max_iter_reweights_*_gridsize_20_glasso_summary.csv"))
if not files:
    raise SystemExit(f"No CSVs in {BASEDIR.resolve()} matching pattern")

def parse_reweights(p: Path):
    m = re.search(r"max_iter_reweights_(\d+)", p.name)
    return int(m.group(1)) if m else None

rows = []
for fp in files:
    k = parse_reweights(fp)
    if k is None:
        continue
    df = pd.read_csv(fp)
    if df.empty:
        continue

    df.columns = df.columns.str.strip()
    # tolerate header variants if needed:
    # df = df.rename(columns={"mmse": "max_f1", "best_f1": "max_f1"})

    need = {"penalty", "min_nmse"}
    if not need.issubset(df.columns):
        continue

    df = df[~df["penalty"].isin(EXCLUDE)]  # exclude here
    if df.empty:
        continue

    df["reweights"] = k
    rows.append(df[["penalty", "min_nmse", "reweights"]])

if not rows:
    raise SystemExit("No valid rows after exclusions.")

full = pd.concat(rows, ignore_index=True).sort_values(["penalty", "reweights"])
full.to_csv(BASEDIR / "combined_min_mmse_by_reweights_excluded.csv", index=False)

# Plot
plt.figure(figsize=(7,5))
penalties = ["L1", "L0_5", "Log", "MCP"]
k =0
for pen, g in full.groupby("penalty"):
    g = g.sort_values("reweights")
    plt.plot(g["reweights"], g["min_nmse"], marker="o", linestyle="-", label=penalties[k])
    k = k+1

plt.xlabel("Number of iterations per reweighting")
plt.xscale("log")  
plt.ylabel("min NMSE")
plt.xticks(sorted(full["reweights"].unique()))
plt.grid(True, linewidth=0.5, alpha=0.4)
plt.legend(title="penalty", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(
    f"bench_results/PGLASSO_reweights_p_75_n_1000_alpha_0.9_minMMSE.pdf")
plt.show()

plt.figure(figsize=(8,5))
penalties = ["L1", "L0_5", "Log", "MCP"]
k =0
for pen, g in full.groupby("penalty"):
    g = g.sort_values("reweights")
    plt.plot(20*g["reweights"], g["min_nmse"], marker="o", linestyle="-", label=penalties[k])
    k = k+1

plt.xlabel("Number of total iterations")
plt.ylabel("min NMSE")
plt.grid(True, linewidth=0.5, alpha=0.4)
plt.legend(title="penalty", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(
    f"bench_results/PGLASSO_iterations_p_75_n_1000_alpha_0.9_minMMSE.pdf")
plt.show()

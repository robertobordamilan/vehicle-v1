import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import build_vehicle_graph, laplacian_and_alpha

OUTDIR = "outputs"
T = 120

def entropy(psi):
    s = float(psi.sum())
    if s <= 0:
        return np.nan
    p = psi / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)

    G, coords = build_vehicle_graph(dims=(2,2,4))
    nodes = list(G.nodes())
    node_to_i = {n:i for i,n in enumerate(nodes)}
    L, alpha = laplacian_and_alpha(G, nodes)

    def energy(psi):
        return float(psi @ (L @ psi))

    sources = [("CUBE", ci) for ci in range(16)]

    H_all = []
    E_all = []
    M_all = []
    report_rows = []

    for ci, s in enumerate(sources):
        psi = np.zeros(len(nodes), float)
        psi[node_to_i[s]] = 1.0
        H = [entropy(psi)]
        E = [energy(psi)]
        M = [psi.sum()]

        for _ in range(T):
            psi = psi - alpha * (L @ psi)
            psi[psi < 0] = 0.0
            H.append(entropy(psi))
            E.append(energy(psi))
            M.append(psi.sum())

        H = np.array(H); E = np.array(E); M = np.array(M)
        H_all.append(H); E_all.append(E); M_all.append(M)

        tol = 1e-10
        report_rows.append({
            "source_cube_index": ci,
            "source_coord": str(coords[ci]),
            "H0": float(H[0]),
            "H_final": float(H[-1]),
            "E0": float(E[0]),
            "E_final": float(E[-1]),
            "H_monotone_nondecreasing": bool(np.all(np.diff(H) >= -tol)),
            "E_monotone_nonincreasing": bool(np.all(np.diff(E) <= tol)),
            "alpha": alpha,
            "T_steps": T,
        })

    H_mat = np.vstack(H_all)
    E_mat = np.vstack(E_all)
    M_mat = np.vstack(M_all)

    k = np.arange(T+1)
    pd.DataFrame(H_mat, columns=[f"k{kk}" for kk in k]).to_csv(f"{OUTDIR}/exp2_entropy_H.csv", index=False)
    pd.DataFrame(E_mat, columns=[f"k{kk}" for kk in k]).to_csv(f"{OUTDIR}/exp2_energy_E.csv", index=False)
    pd.DataFrame(M_mat, columns=[f"k{kk}" for kk in k]).to_csv(f"{OUTDIR}/exp2_mass.csv", index=False)
    pd.DataFrame(report_rows).to_csv(f"{OUTDIR}/exp2_monotonicity_report.csv", index=False)

    H_mean = H_mat.mean(axis=0)
    E_mean = E_mat.mean(axis=0)

    plt.figure(figsize=(8,5), dpi=180)
    plt.plot(k, H_mean)
    plt.xlabel("k (event steps)")
    plt.ylabel("Entropy H(k)")
    plt.title("Experiment 2 (Vehicle v1.0, diffusion): mean entropy vs k")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/exp2_entropy_mean_plot.png")
    plt.close()

    plt.figure(figsize=(8,5), dpi=180)
    plt.plot(k, E_mean)
    plt.xlabel("k (event steps)")
    plt.ylabel("Energy E(k) = psi^T L psi")
    plt.title("Experiment 2 (Vehicle v1.0, diffusion): mean energy vs k")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/exp2_energy_mean_plot.png")
    plt.close()

    print("Experiment 2 complete. Outputs in ./outputs/")

if __name__ == "__main__":
    main()

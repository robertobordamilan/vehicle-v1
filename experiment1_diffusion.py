import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import build_vehicle_graph, laplacian_and_alpha

OUTDIR = "outputs"
T = 60

def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)

    G, coords = build_vehicle_graph(dims=(2,2,4))
    nodes = list(G.nodes())
    node_to_i = {n:i for i,n in enumerate(nodes)}
    L, alpha = laplacian_and_alpha(G, nodes)

    import networkx as nx
    center_node = nx.center(G)[0]

    time = np.arange(T+1)

    def diffuse_from(source_node):
        psi = np.zeros(len(nodes), float)
        psi[node_to_i[source_node]] = 1.0
        y = [psi[node_to_i[center_node]]]
        mass = [psi.sum()]
        for _ in range(T):
            psi = psi - alpha * (L @ psi)
            y.append(psi[node_to_i[center_node]])
            mass.append(psi.sum())
        return np.array(y), np.array(mass)

    sources = [("CUBE", ci) for ci in range(16)]
    Y = []
    M = []
    for s in sources:
        y, m = diffuse_from(s)
        Y.append(y); M.append(m)
    Y = np.vstack(Y)
    M = np.vstack(M)

    dmat = np.zeros((len(sources), len(sources)))
    for i in range(len(sources)):
        for j in range(len(sources)):
            dmat[i,j] = float(np.linalg.norm(Y[i]-Y[j]))

    df_y = pd.DataFrame(Y, columns=[f"t{t}" for t in time])
    df_y.insert(0, "source_cube_index", list(range(len(sources))))
    df_y.insert(1, "source_coord", [str(coords[i]) for i in range(len(sources))])
    df_y.insert(2, "center_node", str(center_node))
    df_y.insert(3, "alpha", alpha)

    df_mass = pd.DataFrame(M, columns=[f"t{t}" for t in time])
    df_mass.insert(0, "source_cube_index", list(range(len(sources))))
    df_mass.insert(1, "source_coord", [str(coords[i]) for i in range(len(sources))])
    df_mass.insert(2, "alpha", alpha)

    df_d = pd.DataFrame(dmat, columns=[f"src{j}" for j in range(len(sources))])
    df_d.insert(0, "src_i", list(range(len(sources))))

    df_y.to_csv(f"{OUTDIR}/exp1_diffusion_center_readout_timeseries.csv", index=False)
    df_mass.to_csv(f"{OUTDIR}/exp1_diffusion_mass_check.csv", index=False)
    df_d.to_csv(f"{OUTDIR}/exp1_diffusion_center_readout_pairwise_L2.csv", index=False)

    plt.figure(figsize=(8,5), dpi=180)
    for i in [0,1,5,10,15]:
        plt.plot(time, Y[i], label=f"src {i} {coords[i]}")
    plt.xlabel("k (steps)")
    plt.ylabel("center readout y(k)")
    plt.title("Experiment 1 (Vehicle v1.0, diffusion): center readout")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/exp1_diffusion_center_readout_plot.png")
    plt.close()

    summary = pd.DataFrame({
        "source_cube_index": list(range(len(sources))),
        "source_coord": [str(coords[i]) for i in range(len(sources))],
        "y_peak": Y.max(axis=1),
        "k_peak": Y.argmax(axis=1),
        "y_final": Y[:, -1],
        "mass_final": M[:, -1],
        "center_node": str(center_node),
        "alpha": alpha,
        "T_steps": T,
    })
    summary.to_csv(f"{OUTDIR}/exp1_diffusion_summary.csv", index=False)

    print("Experiment 1 complete. Outputs in ./outputs/")

if __name__ == "__main__":
    main()

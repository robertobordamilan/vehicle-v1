import numpy as np
import pandas as pd

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
    nbrs = sorted(list(G.neighbors(center_node)))
    readout_nodes = [center_node] + nbrs
    readout_idx = [node_to_i[n] for n in readout_nodes]
    k_readout = len(readout_nodes)

    def diffuse_readout(source_node):
        psi = np.zeros(len(nodes), float)
        psi[node_to_i[source_node]] = 1.0
        Y = np.zeros((T+1, k_readout), float)
        Y[0,:] = psi[readout_idx]
        for t in range(T):
            psi = psi - alpha * (L @ psi)
            Y[t+1,:] = psi[readout_idx]
        return Y

    sources = [("CUBE", ci) for ci in range(16)]
    feats = np.vstack([diffuse_readout(s).reshape(-1) for s in sources])

    feats_round = np.round(feats, 12)
    keys = [tuple(r) for r in feats_round]
    groups = {}
    for i,k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    groups_2plus = [idxs for idxs in groups.values() if len(idxs) >= 2]

    report = pd.DataFrame([{
        "readout": "center + 1-hop neighbors",
        "center_node": str(center_node),
        "k_readout_nodes": k_readout,
        "T_steps": T,
        "num_sources": len(sources),
        "num_unique_signatures": len(groups),
        "num_indistinguishable_groups": len(groups_2plus),
        "max_group_size": max(len(v) for v in groups.values()),
        "alpha": alpha,
    }])
    report.to_csv(f"{OUTDIR}/exp1B_report.csv", index=False)

    groups_df = pd.DataFrame([{
        "group_size": len(idxs),
        "sources": idxs,
        "coords": [str(coords[i]) for i in idxs],
    } for idxs in groups_2plus])
    groups_df.to_csv(f"{OUTDIR}/exp1B_groups.csv", index=False)

    print("Experiment 1B complete. Outputs in ./outputs/")

if __name__ == "__main__":
    main()

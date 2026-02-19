import numpy as np
import networkx as nx
from scipy.sparse import diags

def make_centers(dims, gap=1.2, a=1.0):
    nx_, ny_, nz_ = dims
    coords = []
    centers = []
    idx_map = {}
    idx = 0
    for i in range(nx_):
        for j in range(ny_):
            for k in range(nz_):
                c = np.array([(i-(nx_-1)/2)*a*gap,
                              (j-(ny_-1)/2)*a*gap,
                              (k-(nz_-1)/2)*a*gap], float)
                coords.append((i,j,k))
                centers.append(c)
                idx_map[(i,j,k)] = idx
                idx += 1
    return np.array(centers), coords, idx_map

def neighbors_of(dims):
    nx_, ny_, nz_ = dims
    conn = []
    for i in range(nx_):
        for j in range(ny_):
            for k in range(nz_):
                if i+1 < nx_: conn.append(((i,j,k),(i+1,j,k)))
                if j+1 < ny_: conn.append(((i,j,k),(i,j+1,k)))
                if k+1 < nz_: conn.append(((i,j,k),(i,j,k+1)))
    return conn

def build_vehicle_graph(dims=(2,2,4)):
    # Vehicle v1.0 topology used for diffusion experiments.
    centers, coords, idx_map = make_centers(dims)
    conn = neighbors_of(dims)
    axes = ["X","Y","Z"]

    G = nx.Graph()
    # cube centers
    for ci in range(len(centers)):
        G.add_node(("CUBE", ci))

    # internal nodes & edges
    for axis in axes:
        for ci in range(len(centers)):
            G.add_node((axis, "CENTER", ci))
            G.add_edge((axis, "CENTER", ci), ("CUBE", ci))
            for k in range(4):
                G.add_node((axis, "CORNER", ci, k))
                G.add_edge((axis, "CORNER", ci, k), ("CUBE", ci))

    # lattice edges
    for u,v in conn:
        iu, iv = idx_map[u], idx_map[v]
        G.add_edge(("CUBE", iu), ("CUBE", iv))
        for axis in axes:
            G.add_edge((axis, "CENTER", iu), (axis, "CENTER", iv))
            for k in range(4):
                G.add_edge((axis, "CORNER", iu, k), (axis, "CORNER", iv, k))
    return G, coords

def laplacian_and_alpha(G, nodelist):
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=float, weight=None, format="csr")
    deg = np.array(A.sum(axis=1)).reshape(-1)
    L = diags(deg, 0) - A
    alpha = 0.95 / float(deg.max())
    return L, alpha

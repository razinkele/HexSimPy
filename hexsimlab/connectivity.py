raise NotImplementedError(
    "hexsimlab.connectivity is an incomplete prototype. "
    "The 'neighbors' function is undefined. Use salmon_ibm.hexsim instead."
)

import networkx as nx


def build_graph(grid):

    G = nx.Graph()

    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if grid[r, c] < 0:
                continue

            node = (r, c)

            for n in neighbors(r, c):
                G.add_edge(node, n)

    return G

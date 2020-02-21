import networkx as nx
import numpy as np
import unittest

import csrgraph

class TestGraph(unittest.TestCase):
    def test_norm(self):
        wg = nx.generators.classic.wheel_graph(10)
        A = nx.adj_matrix(wg)
        nodes = list(map(str, list(wg)))
        G = csrgraph.Graph(A.data, A.indptr, A.indices, nodes)
        G.normalize_weights()
        G[1]


if __name__ == '__main__':
    unittest.main()
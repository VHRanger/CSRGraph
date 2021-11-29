import io
import networkx as nx
import numpy as np
import pandas as pd
import random
from scipy import sparse
from sklearn import cluster, manifold, metrics
import string
import unittest
import warnings

import csrgraph as cg

# This is a markov chain with an absorbing state
# Over long enough, every walk ends stuck at node 1
absorbing_state_graph = sparse.csr_matrix(np.array([
    [1, 0., 0., 0., 0.], # 1 point only to himself
    [0.5, 0.3, 0.2, 0., 0.], # everyone else points to 1
    [0.5, 0, 0.2, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.1, 0.2],
    ], dtype=np.float32)
)

absorbing_state_graph_2 = sparse.csr_matrix(np.array([
    [0.5, 0.5, 0., 0., 0.], # 1 and 2 are an absorbing state
    [0.5, 0.5, 0., 0., 0.], # everyone else can get absorbed
    [0.5, 0.2, 0., 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.2, 0.1],
    [0.5, 0.1, 0.1, 0.1, 0.2],
    ], dtype=np.float32)
)

# This graph is two disconnected subgraphs
# No walk can go from one to the other
disconnected_graph = sparse.csr_matrix(np.array([
    [0.5, 0.5, 0., 0., 0.], # 1 and 2 together
    [0.5, 0.5, 0., 0., 0.],
    [0., 0., 0.7, 0.2, 0.1], # 3-5 together
    [0., 0., 0.1, 0.2, 0.7],
    [0., 0., 0.1, 0.7, 0.2],
    ], dtype=np.float32)
)

class TestGGVec(unittest.TestCase):
    def test_given_disjoint_graphs_embeddings_cluster(self):
        """
        Embedding disjoint subgraphs should cluster correctly
        """
        n_clusters = 5
        graph_size = 150
        G = nx.complete_graph(graph_size)
        for i in range(1, n_clusters):
            G = nx.disjoint_union(G, nx.complete_graph(graph_size))
        labels = []
        for i in range(n_clusters):
            labels.append([i] * graph_size)
        labels = sum(labels, [])
        # Embed Graph and cluster around it
        G = cg.csrgraph(G)
        v = G.ggvec(
            n_components=4,
            tol=0.1,
            max_epoch=550,
            learning_rate=0.05, 
            max_loss=1.,
            verbose=False
        )
        cluster_hat = cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='cosine', 
            linkage='average'
        ).fit(v).labels_
        r1 = metrics.adjusted_mutual_info_score(cluster_hat, labels)
        r2 = metrics.adjusted_rand_score(cluster_hat, labels)
        r3 = metrics.fowlkes_mallows_score(cluster_hat, labels)
        self.assertGreaterEqual(r1, 0.65)
        self.assertGreaterEqual(r2, 0.65)
        self.assertGreaterEqual(r3, 0.65)

class TestGraph(unittest.TestCase):
    def test_api(self):
        wg = nx.generators.classic.wheel_graph(10)
        A = nx.adjacency_matrix(wg)
        nodes = list(map(str, list(wg)))
        G = cg.csrgraph(wg)
        G.normalize()
        G = cg.csrgraph(A, nodes)
        G.normalize()
        G.random_walk_resample()

    def test_data_stealing(self):
        """normal ctor points to underlying passed data"""
        wg = nx.generators.classic.wheel_graph(10)
        A = sparse.csr_matrix(nx.adjacency_matrix(wg))
        nodes = list(map(str, list(wg)))
        G = cg.csrgraph(wg)
        self.assertIs(G.src, G.mat.indptr)
        self.assertIs(G.dst, G.mat.indices)
        self.assertIs(G.weights, G.mat.data)
        G = cg.csrgraph(A, nodenames=nodes, copy=False)
        self.assertIs(G.src, A.indptr)
        self.assertIs(G.dst, A.indices)
        self.assertIs(G.weights, A.data)

    def test_row_normalizer(self):
        # Multiplying by a constant returns the same
        test1 = cg.csrgraph(disconnected_graph * 3)
        test2 = cg.csrgraph(absorbing_state_graph_2 * 6)
        test3 = cg.csrgraph(absorbing_state_graph * 99)
        # Scipy.sparse uses np.matrix which throws warnings
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        np.testing.assert_array_almost_equal(
            test1.normalize().mat.toarray(),
            disconnected_graph.toarray(),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            test2.normalize().mat.toarray(),
            absorbing_state_graph_2.toarray(),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            test3.normalize().mat.toarray(),
            absorbing_state_graph.toarray(),
            decimal=3
        )
        testzeros = cg.csrgraph(np.array([
                [0,0,0,0,0,0], # all 0 row
                [1,2,3,4,5,6],
                [1,0,0,1,0,1],
                [1,1,1,1,1,1],
                [1,0,0,0,0,0],
                [0,10,0,1,0,0.1]
            ])
        ).normalize().mat
        self.assertAlmostEqual(testzeros[0].sum(), 0.)
        warnings.resetwarnings()


class TestSparseUtilities(unittest.TestCase):
    """
    Test graph embeddings sub methods
    """
    def test_given_disconnected_graph_walks_dont_cross(self):
        walks1 = cg.csrgraph(disconnected_graph).random_walks(
            start_nodes=[0,1],
            walklen=10
        )
        walks2 = cg.csrgraph(disconnected_graph).random_walks(
            start_nodes=[2,3,4],
            walklen=10
        )
        end_state1 = walks1[:, -1]
        end_state2 = walks2[:, -1]
        self.assertTrue(np.isin(end_state1, [0,1]).all(),
            f"Walks: {walks1} \nEndStates: {end_state1}\n"
        )
        self.assertTrue(np.isin(end_state2, [3,4,2]).all(),
            f"Walks: {walks2} \nEndStates: {end_state2}\n"
        )


    def test_random_walk_uniform_dist(self):
        n_nodes = 15
        n_walklen = 100
        fully_connected = cg.csrgraph(np.ones((n_nodes,n_nodes))).normalize()
        t1 = fully_connected.random_walks(
            walklen=n_walklen,
            epochs=10)
        expected_val = (n_nodes - 1) / 2
        self.assertTrue(np.abs(np.mean(t1) - expected_val) < 0.3)


    def test_given_absorbing_graph_walks_absorb(self):
        # Walks should be long enough to avoid flaky tests
        walks1 = cg.csrgraph(absorbing_state_graph).random_walks(
            walklen=80
        )
        walks2 = cg.csrgraph(absorbing_state_graph).random_walks(
            walklen=80
        )
        walks3 = cg.csrgraph(absorbing_state_graph).random_walks(
            walklen=50, epochs=80
        )
        walks4 = cg.csrgraph(absorbing_state_graph).random_walks(
            walklen=50, epochs=80
        )
        end_state1 = walks1[:, -1]
        end_state2 = walks2[:, -1]
        end_state3 = walks3[:, -1]
        end_state4 = walks4[:, -1]
        self.assertTrue(np.isin(end_state1, [0]).all(),
            f"Walks: {walks1} \nEndStates: {end_state1}\n"
        )
        self.assertTrue(np.isin(end_state2, [0, 1]).all(),
            f"Walks: {walks2} \nEndStates: {end_state2}\n"
        )


    def test_changing_n_threads_works(self):
        """
        force recompile with different # threads
        """
        walks = cg.csrgraph(absorbing_state_graph, threads=3).random_walks(
            walklen=50, 
            epochs=80
        )
        self.assertTrue(True, "Should get here without issues")
        walks = cg.csrgraph(absorbing_state_graph, threads=0).random_walks(
            walklen=50, 
            epochs=80
        )
        self.assertTrue(True, "Should get here without issues again")

class TestFileInput(unittest.TestCase):
    """
    Test 
    """
    def test_karate(self):
        fname = "./data/karate_edges.txt"
        G = cg.read_edgelist(fname)
        m = G.mat.todense()
        df = pd.read_csv(fname, sep="\t", header=None)
        df.columns = ['src', 'dst']
        for i in range(len(df)):
            s = df.iloc[i].src
            d = df.iloc[i].dst
            if m[s-1, d-1] != 1:
                raise ValueError(f"For src {s}, dst {d}, error {m}")
                self.assertEqual(m[s-1, d-1], 1)
        # Only those edges are present
        self.assertTrue(m.sum() == 154)

    def test_string_karate(self):
        N_NODES = 35
        STR_LEN = 10
        fname = "./data/karate_edges.txt"
        df = pd.read_csv(fname, sep="\t", header=None)
        # string node names for each node ID
        new_names = [
            ''.join(random.choice(string.ascii_uppercase) 
                    for _ in range(STR_LEN))
            for i in range(N_NODES)
        ]
        # Map node ID -> new node name
        name_dict = dict(zip(np.arange(N_NODES), new_names))
        for c in df.columns:
            df[c] = df[c].map(name_dict)
        # Pass this new data to read_edgelist
        data = io.StringIO(df.to_csv(index=False, header=False))
        G = cg.read_edgelist(data, sep=',')
        # re-read original graph
        df2 = pd.read_csv(fname, sep="\t", header=None)
        # re-map IDs to string node names
        for c in df2.columns:
            df2[c] = df2[c].map(name_dict)
        df2.columns = ['src', 'dst']
        for i in range(len(df2)):
            s = df2.iloc[i].src
            d = df2.iloc[i].dst
            # addressing graph by __getitem__ with str
            # should return list of str node names
            self.assertTrue(d in G[s])
        # Only those edges are present
        m = G.mat.todense()
        self.assertTrue(m.sum() == 154)

    def test_float_weights_reading(self):
        fname = "./data/karate_edges.txt"
        df = pd.read_csv(fname, sep="\t", header=None)
        df['weights'] = np.random.rand(df.shape[0])
        data = io.StringIO(df.to_csv(index=False, header=False))
        G = cg.read_edgelist(data, sep=',')
        self.assertTrue((G.weights < 1).all())
        self.assertTrue((G.weights > 0).all())

    def test_int_weights_reading(self):
        WEIGHT_VALUE = 5
        fname = "./data/karate_edges.txt"
        df = pd.read_csv(fname, sep="\t", header=None)
        df['weights'] = np.ones(df.shape[0]) * WEIGHT_VALUE
        data = io.StringIO(df.to_csv(index=False, header=False))
        G = cg.read_edgelist(data, sep=',')
        self.assertTrue((G.weights == WEIGHT_VALUE).all())
        self.assertTrue((G.weights == WEIGHT_VALUE).all())

    def test_largenumbererror(self):
        fname = "./data/largenumbererror.csv"
        G = cg.read_edgelist(fname, sep=',')
        self.assertTrue(len(G.nodes()) == 4)
        # These two have no edges
        self.assertTrue(len(G[44444444444444]) == 0)
        self.assertTrue(len(G[222222222222]) == 0)
        # These two have one edge
        self.assertTrue(len(G[333333333333333]) == 1)
        self.assertTrue(len(G[1111111111111]) == 1)

    def test_unfactored_edgelist_directed(self):
        fname = "./data/unfactored_edgelist.csv"
        G = cg.read_edgelist(fname, directed=True, sep=',')
        nxG = cg.csrgraph(
            nx.read_edgelist(
                fname, 
                delimiter=',',
                create_using=nx.DiGraph(),
            )
        )
        self.assertEqual(G.src.size, nxG.src.size)
        self.assertEqual(G.dst.size, nxG.dst.size)
        self.assertEqual(G.weights.size, nxG.weights.size)
        self.assertEqual(G.weights.sum(), nxG.weights.sum())
        # The number of edges on source nodes should have same statistics
        Gdiff = np.diff(G.src)
        nxGdiff = np.diff(nxG.src)
        self.assertEqual(int(Gdiff.mean() * 1000), int(nxGdiff.mean() * 1000))
        self.assertEqual(int(Gdiff.std() * 1000), int(nxGdiff.std() * 1000))
        self.assertEqual(Gdiff.min(), nxGdiff.min())
        self.assertEqual(Gdiff.max(), nxGdiff.max())
        for i in range(1, 10):
            self.assertEqual(np.quantile(Gdiff, i / 10), np.quantile(nxGdiff, i / 10))

    def test_unfactored_edgelist_undirected(self):
        """
        Undirected edgelist reading works
        Even on disconnected graphs
        """
        fname = "./data/unfactored_edgelist.csv"
        ### FIX FIX FIX ###
        G = cg.read_edgelist(fname, directed=False, sep=',')
        nxG = cg.csrgraph(
            nx.read_edgelist(
                fname, 
                delimiter=',',
                create_using=nx.Graph(),
            )
        )
        self.assertEqual(G.src.size, nxG.src.size)
        self.assertEqual(G.dst.size, nxG.dst.size)
        self.assertEqual(G.weights.size, nxG.weights.size)
        self.assertEqual(G.weights.sum(), nxG.weights.sum())
        # The number of edges on source nodes should have same statistics
        Gdiff = np.diff(G.src)
        nxGdiff = np.diff(nxG.src)
        self.assertEqual(int(Gdiff.mean() * 1000), int(nxGdiff.mean() * 1000))
        self.assertEqual(int(Gdiff.std() * 1000), int(nxGdiff.std() * 1000))
        self.assertEqual(Gdiff.min(), nxGdiff.min())
        self.assertEqual(Gdiff.max(), nxGdiff.max())
        for i in range(1, 10):
            self.assertEqual(np.quantile(Gdiff, i / 10), np.quantile(nxGdiff, i / 10))


class TestNodeWalks(unittest.TestCase):
    """
    Test that Node2Vec walks do as they should
    """
    def test_return_weight_inf_loops(self):
        """
        if return weight ~inf, should loop back and forth
        """
        n_nodes = 5
        n_epoch = 2
        walklen=10
        fully_connected = np.ones((n_nodes,n_nodes))
        np.fill_diagonal(fully_connected, 0)
        fully_connected = cg.csrgraph(fully_connected, threads=1).normalize()
        t1 = fully_connected.random_walks(
            walklen=walklen,
            epochs=n_epoch,
            return_weight=99999,
            neighbor_weight=1.)
        # Neighbor weight ~ 0 should also loop 
        t2 = fully_connected.random_walks(
            walklen=walklen,
            epochs=n_epoch,
            return_weight=1.,
            neighbor_weight=0.0001)
        self.assertTrue(t1.shape == (n_nodes * n_epoch, walklen))
        # even columns should be equal (always returning)
        np.testing.assert_array_equal(t1[:, 0], t1[:, 2])
        np.testing.assert_array_equal(t1[:, 0], t1[:, 4])
        np.testing.assert_array_equal(t1[:, 0], t1[:, 6])
        np.testing.assert_array_equal(t2[:, 0], t2[:, 2])
        np.testing.assert_array_equal(t2[:, 0], t2[:, 4])
        np.testing.assert_array_equal(t2[:, 0], t2[:, 6])
        # same for odd columns
        np.testing.assert_array_equal(t1[:, 1], t1[:, 3])
        np.testing.assert_array_equal(t1[:, 1], t1[:, 5])
        np.testing.assert_array_equal(t1[:, 1], t1[:, 7])
        np.testing.assert_array_equal(t2[:, 1], t2[:, 3])
        np.testing.assert_array_equal(t2[:, 1], t2[:, 5])
        np.testing.assert_array_equal(t2[:, 1], t2[:, 7])

    def test_no_loop_weights(self):
        """
        if return weight ~0, should never return
        """
        n_nodes = 5
        n_epoch = 2
        walklen=10
        fully_connected = np.ones((n_nodes,n_nodes))
        np.fill_diagonal(fully_connected, 0)
        fully_connected = cg.csrgraph(fully_connected, threads=1).normalize()
        t1 = fully_connected.random_walks(
            walklen=walklen,
            epochs=n_epoch,
            return_weight=0.0001,
            neighbor_weight=1.)
        # Neighbor weight ~inf should also never return 
        t2 = fully_connected.random_walks(
            walklen=walklen,
            epochs=n_epoch,
            return_weight=1.,
            neighbor_weight=99999)
        self.assertTrue(t1.shape == (n_nodes * n_epoch, walklen))
        # Test that it doesn't loop back
        # Difference between skips shouldnt be 0 anywhere
        tres1 = ((t1[:, 0] - t1[:, 2]) != 0)
        tres2 = ((t1[:, 1] - t1[:, 3]) != 0)
        tres3 = ((t1[:, 2] - t1[:, 4]) != 0)
        tres4 = ((t1[:, 3] - t1[:, 5]) != 0)
        for i in [tres1, tres2, tres3, tres4]:
            if not i.all():
                print(f"ERROR in walks\n\n {t1}")
            self.assertTrue(i.all())
        # Second by neighbor weight
        tres1 = ((t2[:, 0] - t2[:, 2]) != 0)
        tres2 = ((t2[:, 1] - t2[:, 3]) != 0)
        tres3 = ((t2[:, 2] - t2[:, 4]) != 0)
        tres4 = ((t2[:, 3] - t2[:, 5]) != 0)
        for i in [tres1, tres2, tres3, tres4]:
            if not i.all():
                print(f"ERROR in walks\n\n {t2}")
            self.assertTrue(i.all())

            
    def test_parallel_n2v(self):
        """
        Numba is capricious with parallel node2vec, test that it works
        """
        n_nodes = 10
        n_epoch = 4
        walklen=30
        fully_connected = np.ones((n_nodes,n_nodes))
        np.fill_diagonal(fully_connected, 0)
        fully_connected = cg.csrgraph(fully_connected, threads=0).normalize()
        t1 = fully_connected.random_walks(
            walklen=walklen,
            epochs=n_epoch,
            return_weight=0.5,
            neighbor_weight=1.5)

    def test_n2v_bounds(self):
        """
        Bug on node2vec random walks being segfault/out-of-bounds
        Should be fixed forever
        """
        G = cg.read_edgelist("./data/wiki_edgelist.txt")
        rw = G.random_walks(return_weight=0.2)
        self.assertEqual(int(G.nodes().max()), rw.max())
        self.assertEqual(int(G.nodes().min()), rw.min())


if __name__ == '__main__':
    unittest.main()

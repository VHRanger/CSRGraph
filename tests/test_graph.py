import networkx as nx
import numpy as np
from scipy import sparse
import unittest
import warnings

import csrgraph 
from csrgraph.graph import CSRGraph

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

class TestGraph(unittest.TestCase):
    def test_api(self):
        wg = nx.generators.classic.wheel_graph(10)
        A = nx.adj_matrix(wg)
        nodes = list(map(str, list(wg)))
        G = csrgraph.CSRGraph(wg)
        G.normalize()
        G = csrgraph.CSRGraph(A, nodes)
        G.normalize()

    def test_row_normalizer(self):
        # Multiplying by a constant returns the same
        test1 = CSRGraph(disconnected_graph * 3)
        test2 = CSRGraph(absorbing_state_graph_2 * 6)
        test3 = CSRGraph(absorbing_state_graph * 99)
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
        testzeros = CSRGraph(np.array([
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
        walks1 = CSRGraph(disconnected_graph).random_walks(
            start_nodes=[0,1],
            walklen=10
        )
        walks2 = CSRGraph(disconnected_graph).random_walks(
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
        fully_connected = CSRGraph(np.ones((n_nodes,n_nodes))).normalize()
        t1 = fully_connected.random_walks(
            walklen=n_walklen,
            epochs=10)
        expected_val = (n_nodes - 1) / 2
        self.assertTrue(np.abs(np.mean(t1) - expected_val) < 0.3)


    def test_given_absorbing_graph_walks_absorb(self):
        # Walks should be long enough to avoid flaky tests
        walks1 = CSRGraph(absorbing_state_graph).random_walks(
            walklen=80
        )
        walks2 = CSRGraph(absorbing_state_graph).random_walks(
            walklen=80
        )
        walks3 = CSRGraph(absorbing_state_graph).random_walks(
            walklen=50, epochs=80
        )
        walks4 = CSRGraph(absorbing_state_graph).random_walks(
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
        walks = CSRGraph(absorbing_state_graph, threads=3).random_walks(
            walklen=50, 
            epochs=80
        )
        self.assertTrue(True, "Should get here without issues")
        walks = CSRGraph(absorbing_state_graph, threads=0).random_walks(
            walklen=50, 
            epochs=80
        )
        self.assertTrue(True, "Should get here without issues again")


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
        fully_connected = CSRGraph(fully_connected, threads=1).normalize()
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
        fully_connected = CSRGraph(fully_connected, threads=1).normalize()
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
        fully_connected = CSRGraph(fully_connected, threads=0).normalize()
        t1 = fully_connected.random_walks(
            walklen=walklen,
            epochs=n_epoch,
            return_weight=0.5,
            neighbor_weight=1.5)


if __name__ == '__main__':
    unittest.main()

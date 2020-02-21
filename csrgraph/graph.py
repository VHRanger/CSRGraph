import networkx as nx
import numba
from numba import jit, jitclass
import numpy as np
import os
import pandas as pd
from scipy import sparse
import time
import warnings


# def _plain_bfs(G, source):
#     """A fast BFS node generator"""
#     seen = set()
#     nextlevel = {source}
#     while nextlevel:
#         thislevel = nextlevel
#         nextlevel = set()
#         for v in thislevel:
#             if v not in seen:
#                 yield v
#                 seen.add(v)
#                 nextlevel.update(G_adj[v])

# return sum(1 for node in _plain_bfs(G, arbitrary_element(G))) == len(G)


class Graph():

    def __init__(self, data, indptr, indices, nodenames):
        """

        """
        # edge weights. Coindexed to dst
        # TODO: don't store data if it's all 1's
        self.weights = data
        # start index of the edges for a node
        # indexes into data and dst
        self.indptr = indptr
        # edge destinations
        self.dst = indices
        # node name -> node ID
        self.names = dict(zip(nodenames, range(data.size)))


    def __getitem__(self, node):
        """
        returns a node's edges
        """
        if type(node) is not str:
            return self.dst[self.indptr[node] : self.indptr[node+1]]
        else:
            idx = self.names[node]
            res = self.dst[idx : idx+1]
            return [self.names[i] for i in res]


#
#
# TODO: Organize Graph method here
# Layout nodes by their 1d embedding's position
#
#


    # TODO: move to other file
    @staticmethod
    @jit(nopython=True, parallel=True, nogil=True, fastmath=True)
    def _random_walk(Tdata, Tindptr, Tindices,
                        sampling_nodes, walklen):
        """
        Create random walks from the transition matrix of a graph 
            in CSR sparse format

        NOTE: scales linearly with threads but hyperthreads don't seem to 
                accelerate this linearly
        
        Parameters
        ----------
        Tdata : 1d np.array
            CSR data vector from a sparse matrix. Can be accessed by M.data
        Tindptr : 1d np.array
            CSR index pointer vector from a sparse matrix. 
            Can be accessed by M.indptr
        Tindices : 1d np.array
            CSR column vector from a sparse matrix. 
            Can be accessed by M.indices
        sampling_nodes : 1d np.array of int
            List of node IDs to start random walks from.
            Is generally equal to np.arrange(n_nodes) repeated for each epoch
        walklen : int
            length of the random walks

        Returns
        -------
        out : 2d np.array (n_walks, walklen)
            A matrix where each row is a random walk, 
            and each entry is the ID of the node
        """
        n_walks = len(sampling_nodes)
        res = np.empty((n_walks, walklen), dtype=np.int64)
        for i in numba.prange(n_walks):
            # Current node (each element is one walk's state)
            state = sampling_nodes[i]
            for k in range(walklen-1):
                # Write state
                res[i, k] = state
                # Find row in csr indptr
                start = Tindptr[state]
                end = Tindptr[state+1]
                # transition probabilities
                p = Tdata[start:end]
                # cumulative distribution of transition probabilities
                cdf = np.cumsum(p)
                # Random draw in [0, 1] for each row
                # Choice is where random draw falls in cumulative distribution
                draw = np.random.rand()
                # Find where draw is in cdf
                # Then use its index to update state
                next_idx = np.searchsorted(cdf, draw)
                # Winner points to the column index of the next node
                state = Tindices[start + next_idx]
            # Write final states
            res[i, -1] = state
        return res


    # TODO: move to other file
    @staticmethod
    def _csr_normalize(mat):
        """
        Normalize a sparse CSR matrix row-wise (each row sums to 1)

        If a row is all 0's, it remains all 0's
        
        Parameters
        ----------
        mat : scipy.sparse.csr matrix
            Matrix in CSR sparse format

        Returns
        -------
        out : scipy.sparse.csr matrix
            Normalized matrix in CSR sparse format
        """
        n_nodes = mat.shape[0]
        # Normalize Adjacency matrix to transition matrix
        # Diagonal of the degree matrix is the sum of nonzero elements
        degrees_div = np.array(np.sum(mat, axis=1)).flatten()
        # This is equivalent to inverting the diag mat
        # weights are 1 / degree
        degrees = np.divide(
            1,
            degrees_div,
            out=np.zeros_like(degrees_div, dtype=float),
            where=(degrees_div != 0)
        )
        # construct sparse diag mat 
        # to broadcast weights to adj mat by dot product
        D = sparse.dia_matrix((n_nodes,n_nodes), dtype=np.float64)
        D.setdiag(degrees)   
        # premultiplying by diag mat is row-wise mul
        return sparse.csr_matrix(D.dot(mat))


    def normalize_weights(self):
        """
        Normalizes edge weights per node

        For any node in the Graph, the new edges' weights will sum to 1
        """
        csr = sparse.csr_matrix((self.weights, self.dst, self.indptr))
        csr = self._csr_normalize(csr)
        self.weights = csr.data
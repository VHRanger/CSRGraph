import gc
import networkx as nx
import numba
from numba import jit, jitclass
import numpy as np
import os
import pandas as pd
from scipy import sparse
import time
import warnings

from csrgraph.methods import (_row_norm, _random_walk, _node2vec_walks)

class CSRGraph():
    """
    This top level python class either calls external JIT'ed methods
        or methods from the JIT'ed internal graph
    """
    def __init__(self, data, nodenames=None, mmap=False, threads=0):
        """
        A class for large, possibly on-disk graphs.

        input : The graph data. Can be one of:

            **NetworkX Graph**

            **CSR Matrix**
            
            **(data, indices, indptr)**

        nodenames (array of str or int) : Node names
            The position in this array should correspond with the node ID
            So if passing a CSR Matrix or raw data, it should be co-indexed
            with the matrix/raw data arrays
        
        TODO: add numpy mmap support for very large on-disk graphs
            This also requires routines to read/write
            edgelists, etc. from disk
        """
        # If input is a CSRGraph, copy
        if isinstance(data, CSRGraph):
            self.weights = data.weights
            self.indptr = data.indptr
            self.dst = data.dst
            try:
                self.mat = data.mat
            except AttributeError:
                pass
        # NetworkX Graph input
        elif isinstance(data, nx.Graph):
            self.mat = nx.adj_matrix(data)
            nodenames = list(data)
            # TODO: test
            if np.array_equal(range(len(data)), nodenames):
                nodenames = None
        # CSR Matrix Input
        elif isinstance(data, sparse.csr_matrix):
            self.mat = data
        # Numpy Array Input
        elif isinstance(data, np.ndarray):
            self.mat = sparse.csr_matrix(data)
        else:
            raise ValueError('Incorrect input type')
        if hasattr(self, 'mat'):
            # edge weights. Coindexed to dst
            # TODO: don't store data if it's all 1's
            self.weights = self.mat.data
            # start index of the edges for a node
            # indexes into data and dst
            self.indptr = self.mat.indptr
            # edge destinations
            self.dst = self.mat.indices

        # node name -> node ID
        if nodenames is not None:
            self.names = dict(zip(nodenames, np.arange(self.dst.size)))

        # indptr has one more element than nnodes
        self.nnodes = self.indptr.size - 1

        # Manage threading through Numba hack
        if type(threads) is not int:
            raise ValueError("Threads argument must be an int!")
        if threads == 0:
            threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        threads = str(threads)
        # If we change the number of threads, recompile
        try:
            prev_numba_value = os.environ['NUMBA_NUM_THREADS']
        except KeyError:
            prev_numba_value = threads
        if threads != prev_numba_value:
            os.environ['NUMBA_NUM_THREADS'] = threads
            _random_walk.recompile()
            _row_norm.recompile()
            _node2vec_walks.recompile()

    def __getitem__(self, node):
        """
        returns a node's edges

        TODO: test
        """
        if type(node) is not str:
            return self.dst[self.indptr[node] : self.indptr[node+1]]
        else:
            idx = self.names[node]
            res = self.dst[self.indptr[idx] : self.indptr[idx+1]]
            # TODO: Change to mapping method to str
            return [self.names[i] for i in res]

    def matrix(self):
        """
        Return's the graph's scipy sparse CSR matrix
        """
        if hasattr(self, 'mat'):
            return self.mat
        else:
            return sparse.csr_matrix((weights, dst, indptr))

    def nodes(self):
        """
        Returns the graph's nodes, in order
        """
        if hasattr(self, 'names'):
            return self.names.keys()
        else:
            return np.arange(self.nnodes)

    def normalize(self, return_self=True):
        """
        Normalizes edge weights per node

        For any node in the Graph, the new edges' weights will sum to 1
        """
        new_weights = _row_norm(self.weights, self.indptr)
        if return_self:
            self.weights = new_weights
            if hasattr(self, 'mat'):
                self.mat = sparse.csr_matrix((self.weights, self.dst, 
                                                self.indptr))
            return self
        else:
            return CSRGraph(sparse.csr_matrix(
                (new_weights, self.dst, self.indptr)))

    def random_walks(self,
                walklen=10,
                epochs=1,
                start_nodes=None,
                normalize_self=False,
                return_weight=1.,
                neighbor_weight=1.):
        """
        Create random walks from the transition matrix of a graph 
            in CSR sparse format

        Parameters
        ----------
        T : scipy.sparse.csr matrix
            Graph transition matrix in CSR sparse format
        walklen : int
            length of the random walks
        epochs : int
            number of times to start a walk from each nodes
        return_weight : float in (0, inf]
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be 
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        neighbor_weight : float in (0, inf]
            Weight on the probability of visitng a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be 
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        threads : int
            number of threads to use.  0 is full use

        Returns
        -------
        out : 2d np.array (n_walks, walklen)
            A matrix where each row is a random walk, 
            and each entry is the ID of the node
        """
        # Make csr graph
        if normalize_self:
            self.normalize(return_self=True)
            T = self
        else:
            T = self.normalize(return_self=False)
        n_rows = T.nnodes
        if start_nodes is None:
            start_nodes = np.arange(n_rows)
        sampling_nodes = np.tile(start_nodes, epochs)
        # Node2Vec Biased walks if parameters specified
        if (return_weight > 1. or return_weight < 1. 
                or neighbor_weight < 1. or neighbor_weight > 1.):
            walks = _node2vec_walks(T.weights, T.indptr, T.dst, 
                                    sampling_nodes=sampling_nodes, 
                                    walklen=walklen, 
                                    return_weight=return_weight, 
                                    neighbor_weight=neighbor_weight)
        # much faster implementation for regular walks
        else:
            walks = _random_walk(T.weights, T.indptr, T.dst, 
                                 sampling_nodes, walklen)
        return walks

    
    #
    #
    # TODO: Map node names method in random_walks here
    #       Add tests for string node names
    #
    #


    #
    #
    # TODO: Organize Graph method here
    # Layout nodes by their 1d embedding's position
    #
    #

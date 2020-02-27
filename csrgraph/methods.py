"""
External repo for JIT'ed graph methods.
These are outside the main class so methods can call them
"""
import numba
from numba import jit, jitclass
import numpy as np
import os
from scipy import sparse

from csrgraph import graph as csrg

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _row_norm(weights, indptr):
    """
    Returns the weights for normalized rows in a CSR Matrix
    """
    n_nodes = indptr.size - 1
    res = np.empty(weights.size, dtype=np.float32)
    for i in numba.prange(n_nodes):
        s1 = indptr[i]
        s2 = indptr[i+1]
        rowsum = np.sum(weights[s1:s2])
        res[s1:s2] = weights[s1:s2] / rowsum
    return res


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _random_walk(weights, indptr, dst,
                 sampling_nodes, walklen):
    """
    Create random walks from the transition matrix of a graph 
        in CSR sparse format

    NOTE: scales linearly with threads but hyperthreads don't seem to 
            accelerate this linearly
    
    Parameters
    ----------
    weights : 1d np.array
        CSR data vector from a sparse matrix. Can be accessed by M.data
    indptr : 1d np.array
        CSR index pointer vector from a sparse matrix. 
        Can be accessed by M.indptr
    dst : 1d np.array
        CSR column vector from a sparse matrix. 
        Can be accessed by M.indices
    sampling_nodes : 1d np.array of int
        List of node IDs to start random walks from.
        If doing sampling from walks, is equal to np.arrange(n_nodes) 
            repeated for each epoch
    walklen : int
        length of the random walks

    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a random walk, 
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    res = np.zeros((n_walks, walklen), dtype=np.int64)
    n_nodes = indptr.size
    n_edges = weights.size
    for i in numba.prange(n_walks):
        # Current node (each element is one walk's state)
        state = sampling_nodes[i]
        for k in range(walklen-1):
            # Write state
            res[i, k] = state
            # Find row in csr indptr
            start = indptr[state]
            end = indptr[state+1]
            # transition probabilities
            p = weights[start:end]
            # cumulative distribution of transition probabilities
            cdf = np.cumsum(p)
            # Random draw in [0, 1] for each row
            # Choice is where random draw falls in cumulative distribution
            draw = np.random.rand()
            # Find where draw is in cdf
            # Then use its index to update state
            next_idx = np.searchsorted(cdf, draw)
            # Winner points to the column index of the next node
            state = dst[start + next_idx]
        # Write final states
        res[i, -1] = state
    return res

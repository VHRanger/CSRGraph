"""
Fast JIT'ed graph methods.
These are outside the CSRGraph class so methods can call them
"""
import gc
import numba
from numba import jit
import numpy as np
import pandas as pd
import os
from scipy import sparse

import csrgraph as cg

@jit(nopython=True, fastmath=True)
def _update_src_array(target, src, cnt):
    """
    Submethod for _edgelist_to_graph

    Fast way to build the src array from groupby result

    Params:
    ---------
    target : np.array
        The resulting csr src array we're building
        Modified in place
    src : np.array
        The result of groupby operation
        Counting each edge under that node
    cnt : np.array
        co-indexed to src above.
        The edge count for each node
    """
    for i in range(src.shape[0]):
        # Offset by one to add the number of nodes
        # in the current node to the src total
        # TODO: Would be faster backwards...?
        #       Start with the max value and reduce on each
        #       Can be done with a cumsum as well
        target[src[i]+1:] += cnt[i]

def _edgelist_to_graph(src, dst, weights, nnodes, nodenames=None):
    """
    Assumptions:
        1) edgelist is sorted by source nodes
        2) nodes are all ints in [0, num_nodes]
    Params:
    ---------
    elist : pd.Dataframe[src, dst, (weight)]
        df of edge pairs. Assumed to be sorted.
        If w weight column is present, named 'weight'
    Return:
    ----------
    csrgraph object 
    """
    new_src = np.zeros(nnodes + 1)
    # Fill indptr array
    new_src[0] = 0 # each idx points to node start idx
    grp = np.unique(src, return_counts=True)
    _update_src_array(new_src, grp[0], grp[1])
    return cg.csrgraph(
        sparse.csr_matrix((weights, dst, new_src)),
        nodenames=nodenames
    )


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _row_norm(weights, src):
    """
    Returns the weights for normalized rows in a CSR Matrix.
    
    Parameters
    ----------
    weights : array[float]
        The data array from a CSR Matrix. 
        For a scipy.csr_matrix, is accessed by M.data
  
    src : array[int]
        The index pointer array from a CSR Matrix. 
        For a scipy.csr_matrix, is accessed by M.indptr
        
    ----------------
    returns : array[float32]
        The normalized data array for the CSR Matrix
    """
    n_nodes = src.size - 1
    res = np.empty(weights.size, dtype=np.float32)
    for i in numba.prange(n_nodes):
        s1 = src[i]
        s2 = src[i+1]
        rowsum = np.sum(weights[s1:s2])
        res[s1:s2] = weights[s1:s2] / rowsum
    return res


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _node_degrees(src, dst):
    """Gets the degree for each node in graph"""
    nnodes = src.size - 1
    res = np.zeros(nnodes, dtype=np.int32)
    for i in numba.prange(nnodes):
        res[i] = src[i+1] - src[i]
    return res


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _src_multiply(node_weights, src, weights):
    """
    Modifies graph weights so each edge source is multiplied by node weight
    
    Returns nothing (in place change)
    
    Parameters
    ----------
    weights : array[float]
        The data array from a CSR Matrix. 
        For a scipy.csr_matrix, is accessed by M.data
  
    src : array[int]
        The index pointer array from a CSR Matrix. 
        For a scipy.csr_matrix, is accessed by M.indptr
    """
    n_nodes = src.size - 1
    assert n_nodes == node_weights.size
    for i in numba.prange(n_nodes):
        s1 = src[i]
        s2 = src[i+1]
        weights[s1:s2] = weights[s1:s2] * node_weights[i]


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _dst_multiply(node_weights, src, dst, weights):
    """
    Modifies graph weights so edge destination are multiplied by node_weight
    
    Returns nothing (in place change)
    
    Parameters
    ----------
    weights : array[float]
        The data array from a CSR Matrix. 
        For a scipy.csr_matrix, is accessed by M.data
  
    src : array[int]
        The index pointer array from a CSR Matrix. 
        For a scipy.csr_matrix, is accessed by M.indptr
    """
    n_nodes = src.size - 1
    assert n_nodes == node_weights.size
    assert weights.size == dst.size
    for i in numba.prange(n_nodes):
        weights[dst == i] *= node_weights[i]

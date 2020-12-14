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
    new_src[1:] = np.cumsum(np.bincount(src, minlength=nnodes))
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

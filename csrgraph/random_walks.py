"""
Fast JIT'ed random walk methods.
These are outside the CSRGraph class so methods can call them
"""
import gc
import numba
from numba import jit
import numpy as np
import pandas as pd

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _random_walk(weights, indptr, dst,
                 sampling_nodes, walklen):
    """
    Create random walks from the transition matrix of a graph 
        in CSR sparse format

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
    res = np.empty((n_walks, walklen), dtype=np.uint32)
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
            # If there are edges in the node, find next step
            if start != end:
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
            # If there are no edges, this is the end of the walk
            else:
              res[i, k:] = state
              break
        # Write final states
        res[i, -1] = state
    return res


@jit(nopython=True, nogil=True, fastmath=True)
def _node2vec_first_step(state, Tdata, Tindices, Tindptr):
    """
    Inner code for node2vec walks
    Normal random walk step
    Comments for this logic are in _random_walk
    """
    start = Tindptr[state]
    end = Tindptr[state+1]
    p = Tdata[start:end]
    cdf = np.cumsum(p)
    draw = np.random.rand()
    next_idx = np.searchsorted(cdf, draw)
    state = Tindices[start + next_idx]
    return state


@jit(nopython=True, nogil=True, fastmath=True)
def _node2vec_inner(
    res, i, k, state,
    Tdata, Tindices, Tindptr, 
    return_weight, neighbor_weight):
    """
    Inner loop core for node2vec walks
    Does the biased walk updating (pure function)
    
    All arguments are directly from the node2vec walks method
    """
    # Find rows in csr indptr
    prev = res[i, k-1]
    start = Tindptr[state]
    end = Tindptr[state+1]
    start_prev = Tindptr[prev]
    end_prev = Tindptr[prev+1]
    # Find overlaps and fix weights
    this_edges = Tindices[start:end]
    prev_edges = Tindices[start_prev:end_prev]
    p = np.copy(Tdata[start:end])
    ret_idx = np.where(this_edges == prev)
    p[ret_idx] = np.multiply(p[ret_idx], return_weight)
    for pe in prev_edges:
        n_idx_v = np.where(this_edges == pe)
        n_idx = n_idx_v[0]
        p[n_idx] = np.multiply(p[n_idx], neighbor_weight)
    # Get next state
    cdf = np.cumsum(np.divide(p, np.sum(p)))
    draw = np.random.rand()
    next_idx = np.searchsorted(cdf, draw)
    new_state = this_edges[next_idx]
    return new_state


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def _node2vec_walks(Tdata, Tindptr, Tindices,
                    sampling_nodes,
                    walklen,
                    return_weight,
                    neighbor_weight):
    """
    Create biased random walks from the transition matrix of a graph 
        in CSR sparse format. Bias method comes from Node2Vec paper.
    
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
    return_weight : float in (0, inf]
        Weight on the probability of returning to node coming from
        Having this higher tends the walks to be 
        more like a Breadth-First Search.
        Having this very high  (> 2) makes search very local.
        Equal to the inverse of p in the Node2Vec paper.
    explore_weight : float in (0, inf]
        Weight on the probability of visitng a neighbor node
        to the one we're coming from in the random walk
        Having this higher tends the walks to be 
        more like a Depth-First Search.
        Having this very high makes search more outward.
        Having this very low makes search very local.
        Equal to the inverse of q in the Node2Vec paper.
    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a biased random walk, 
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    res = np.empty((n_walks, walklen), dtype=np.uint32)
    for i in numba.prange(n_walks):
        # Current node (each element is one walk's state)
        state = sampling_nodes[i]
        res[i, 0] = state
        # Do one normal step first
        state = _node2vec_first_step(state, Tdata, Tindices, Tindptr)
        for k in range(1, walklen-1):
            # Write state
            res[i, k] = state
            state = _node2vec_inner(
                res, i, k, state,
                Tdata, Tindices, Tindptr, 
                return_weight, neighbor_weight
            )
        # Write final states
        res[i, -1] = state
    return res


@jit(nopython=True)
def graph_from_walks_inner(walks):
    """
    Inner loop creating edge pairs from random walk 
    context windows
    
    TODO: optimize? avoid redundant pairs (huge!)
        Context window only adds neighbors to middle of window
        window of 2 --> [_, _, word, _, _]
    """
    res = []
    for i in range(walks.shape[0]):   
        for a in range(0, walks.shape[1]):
            for b in range(1 + a, walks.shape[1]):
                res.append(np.array([walks[i, a], walks[i, b]]))
    return res

def walks_to_edgelist(walks):
    """
    Create a graph from a matrix of random walks
    
    Params:
    ------------
    
    walks : np.array[n_walks, walk_length]
        Random walks. Expected to be the output from 
        csrgraph.random_walks() method.
        
    window : int >= 1
        Skip-Gram context window.
        If two nodes are this number or less apart in a walk,
        an edge will be formed between them
        
    n_splits : int >= 1
        Splits
    """
    wg = pd.DataFrame(graph_from_walks_inner(walks))
    wg.columns = ['src', 'dst']
    wg['weight'] = 1
    wg = (wg.groupby(['src', 'dst'])
            .sum()
            .reset_index(drop=False)
        )
    wg.sort_values(by='src', inplace=True, ignore_index=True)
    gc.collect()
    return wg

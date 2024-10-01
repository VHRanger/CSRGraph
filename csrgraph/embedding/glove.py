import gc
import numba
from numba import jit
import numpy as np
import sklearn
import tqdm
import warnings

@jit(nopython=True, nogil=True, fastmath=True)
def _update_bgrad(b, bgrad, node, loss, learning_rate):
    """
    update bias gradients
    """
    lrate = learning_rate / np.sqrt(bgrad[node])
    b[node] -= lrate * loss
    bgrad[node] += loss ** 2

@jit(nopython=True, nogil=True, fastmath=True)
def _update_wgrad(learning_rate, loss, w1, w2, wgrad):
    """
    Update step: apply gradients and reproject on unit sphere.
    """
    for k in range(w1.size):
        lrate = learning_rate / np.sqrt(wgrad[k])
        grad = loss * w2[k]
        w1[k] = w1[k] - lrate * grad
        wgrad[k] += grad ** 2

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _glove_edges_update(data, dst, src, 
                        w, b, 
                        wgrad, bgrad,
                        shuffle_idx,
                        learning_rate=0.01,
                        max_count=50,
                        exponent=0.5,
                        max_loss=10.):
    """
    This implementation is UNSAFE. 
    We concurrently write to weights and gradients in separate threads
    This is only saved by the fact that edges >>> threads 
    so pr(race condition) is very low

    Couple of issues:
        - Only one weight matrix
        - unvectorized
        - unsafe
        - Assumes symmetric edges (would need two w matrix for directed graphs)
    Implementation inspired from https://github.com/maciejkula/glove-python/blob/master/glove/glove_cython.pyx
    """
    dim = w.shape[1]
    n_edges = dst.size
    largest_grad = np.zeros(n_edges)
    for i in numba.prange(n_edges):
        edge = shuffle_idx[i]
        # edge = i
        node1 = dst[edge]
        # Find node in src array
        node2 = np.searchsorted(src, edge)
        if edge < src[node2]:
            node2 = node2 - 1
        # Loss is really just the dot product 
        #   b/w two connected nodes
        pred = np.dot(w[node1], w[node2])
        # Assume no weights here
        # This zeros out the entry weight and the log(x)
        loss = pred + b[node1] + b[node2]
        entry_w = min(1., data[edge] / max_count) ** exponent
        loss = entry_w * (loss - np.log(data[edge]))
        # Clip the loss for numerical stability.
        if loss < -max_loss:
            loss = -max_loss
        elif loss > max_loss:
            loss = max_loss
        # Update step: apply gradients and reproject on unit sphere.
        _update_wgrad(learning_rate, loss, w[node1], w[node2], wgrad[node1])
        _update_wgrad(learning_rate, loss, w[node2], w[node1], wgrad[node2])
        # Update biases
        _update_bgrad(b, bgrad, node1, loss, learning_rate)
        _update_bgrad(b, bgrad, node2, loss, learning_rate)
        # track largest grad for early stopping
        largest_grad[edge] = np.max(np.abs(loss * w[node2]))
    return np.max(largest_grad)


##########################
#                        #
#       Main method      #
#                        #
##########################

def glove_main(src, dst, weights, n_components=1, 
               tol=0.0001, max_epoch=10_000, 
               learning_rate=0.1, max_loss=10.,
               max_count=50, exponent=0.5, verbose=True):
    """
    Global first order embedding for positive, count-valued sparse matrices.

    This algorithm is normally used in NLP on word co-occurence matrices.
    The algorithm fails if any value in the sparse matrix < 1.
    It is also a poor choice for matrices with homogeneous edge weights.

    Parameters:
    -------------
    n_components (int): 
        Number of individual embedding dimensions.
    tol : float in [0, 1]
        Optimization early stopping criterion.
        Stops when largest gradient is < tol
    max_epoch : int
        Stopping criterion.
    max_count : int
        Ceiling value on edge weights for numerical stability
    exponent : float
        Weighing exponent in loss function. 
        Having this lower reduces effect of large edge weights.
    learning_rate : float in [0, 1]
        Optimization learning rate.
    max_loss : float
        Loss value ceiling for numerical stability.

    References:
    -------------
    Paper: https://nlp.stanford.edu/pubs/glove.pdf
    Original implementation: https://github.com/stanfordnlp/GloVe/blob/master/src/glove.c
    """
    nnodes = src.size - 1
    w = np.random.rand(nnodes, n_components) - 0.5
    b = np.zeros(nnodes, dtype=np.float64)
    wgrad = np.ones_like(w)
    bgrad = np.ones_like(b)
    shuffle_idx = np.arange(dst.size)
    if verbose:
        epoch_range = tqdm.trange(0, max_epoch)
    else:
        epoch_range = range(0, max_epoch)
    for epoch in epoch_range:
        np.random.shuffle(shuffle_idx)
        lgrad = _glove_edges_update(
            weights, dst, src, w, b, 
            wgrad, bgrad, shuffle_idx,
            learning_rate=learning_rate,
            max_count=max_count, exponent=exponent,
            max_loss=max_loss)
        if np.abs(lgrad) < tol:
            return w
    warnings.warn(f"GloVe has not converged. Largest gradient : {lgrad}")
    return w

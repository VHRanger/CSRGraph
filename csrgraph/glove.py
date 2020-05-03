import numba
from numba import jit
import numpy as np
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
def _glove_edges_update(data, dst, indptr, 
                        w, b, 
                        wgrad, bgrad,
                        shuffle_idx,
                        learning_rate=0.01,
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
        - Assumes symmetric edges
        - Assumes unweighed edges
    Implementation inspired from https://github.com/maciejkula/glove-python/blob/master/glove/glove_cython.pyx
    """
    dim = w.shape[1]
    n_edges = dst.size
    largest_grad = np.zeros(n_edges)
    for i in numba.prange(n_edges):
        edge = shuffle_idx[i]
        # edge = i
        node1 = dst[edge]
        # Find node in indptr array
        node2 = np.searchsorted(indptr, edge)
        if edge < indptr[node2]:
            node2 = node2 - 1
        # Loss is really just the dot product 
        #   b/w two connected nodes
        pred = np.dot(w[node1], w[node2])
        # Assume no weights here
        # This zeros out the entry weight and the log(x)
        loss = pred + b[node1] + b[node2]
        # TODO: missing entry weight here and log(x)
        entry_w = min(1., data[edge] / 10) ** 0.5
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
#    /\ Original impl    #
#    ||                  #
#    \/ New Stuff        #
#                        #
##########################


##########################
#                        #
#          Methods       #
#                        #
##########################


def glove_by_edges(data, dst, indptr, nnodes, n_components=1, 
                   tol=0.0001, max_epoch=10_000, 
                   learning_rate=0.1, max_loss=10.):
    """
    Create embeddings using pseudo-GLoVe

    TODO: smart init values
         Based on the indptr array (# of edges corr ~ W value)??
    """
    w = np.random.rand(nnodes, n_components) - 0.5
    b = np.zeros(nnodes, dtype=np.float64)
    wgrad = np.ones_like(w)
    bgrad = np.ones_like(b)
    shuffle_idx = np.arange(dst.size)
    for epoch in range(max_epoch):
        np.random.shuffle(shuffle_idx)
        lgrad = _glove_edges_update(
            data, dst, indptr, w, b, 
            wgrad, bgrad, shuffle_idx,
            learning_rate=learning_rate,
            max_loss=max_loss)
        if np.abs(lgrad) < tol:
            return w
    warnings.warn(f"GloVe has not converged. Largest gradient : {lgrad}")
    return w


@jit(nopython=True)
def glove_by_nodes(dst, indptr, learning_rate=0.01):
    """
    similar to _by_edges, but by nodes

    TODO: impl
    """
    pass

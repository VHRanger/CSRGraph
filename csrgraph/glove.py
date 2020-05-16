import gc
import numba
from numba import jit
import numpy as np
import tqdm
import warnings

@jit(nopython=True, nogil=True, fastmath=True)
def _update_wgrad(learning_rate, loss, w1, w2):
    """Update w1: apply gradients and reproject"""
    for k in range(w1.size):
        grad = loss * w2[k]
        w1[k] = w1[k] - learning_rate * grad

@jit(nopython=True, nogil=True, fastmath=True)
def _update_wgrad_clipped(learning_rate, loss, w1, w2):
    """same as above, clamped in unit sphere"""
    for k in range(w1.size):
        grad = loss * w2[k]
        w1[k] = w1[k] - learning_rate * grad
        if w1[k] < -1.: w1[k] = -1.
        elif w1[k] > 1.: w1[k] = 1.

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _glove_edges_update(data, dst, indptr, w, b,
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
    n_edges = dst.size
    # TODO: compute this mean by parallel reduction
    loss_arr = np.zeros(n_edges)
    for i in numba.prange(n_edges):
        edge = i
        node1 = dst[edge]
        # Find node in indptr array
        node2 = np.searchsorted(indptr, edge)
        if edge < indptr[node2]:
            node2 = node2 - 1
        # Loss is dot product b/w two connected nodes
        pred = np.dot(w[node1], w[node2]) + b[node1] + b[node2]
        # Use arcsinh(weight) instead of log(weight)
        #    handles all real valued weights well
        loss = (pred - np.arcsinh(data[edge]))
        # Clip the loss for numerical stability.
        if loss < -max_loss: loss = -max_loss
        elif loss > max_loss: loss = max_loss
        # Update weights
        _update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        _update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        # Update biases
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss
        # track losses for early stopping
        loss_arr[edge] = loss
    return loss_arr

###########################
#                         #
#    /\ Contraction pass  #
#    ||                   #
#    \/ Relaxation pass   #
#                         #
###########################

def negative_edges(rng, src, dst, nedges):
    """
    TODO: remove
    Generate a list of negative edges
    NOTE: lossy -- may hit real edges. We assume sparsity
    """
    res = np.zeros((nedges, 2), dtype=np.uint32)
    nnodes = src.size - 1
    res[:, 0] = np.tile(np.arange(nnodes, dtype=np.uint32), 
                        int(np.ceil(nedges/nnodes)))[:nedges]
    res[:, 1] = rng.integers(low=0, high=nnodes, size=nedges)
    return res


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _glove_reverse_old(negative_edges, w, b,
                   learning_rate=0.01,
                   max_loss=10.):
    """
    TODO: Remove
    Negative sampling GLoVe pass

    negative_edges : array shaped [n_samples, 2]
        We pass in here to hoist out the complexity of 
        handling multithreaded RNG (which plays poorly with numba)
    """
    # TODO: find a way to generate random nodes on the fly
    #       profile against this pre-gen nodes
    n_edges = negative_edges.shape[0]
    for i in numba.prange(n_edges):
        node1 = negative_edges[i, 0]
        node2 = negative_edges[i, 1]
        # We assume no edge between nodes on negative sampling pass
        # Eg. data[node1, node2] = 0
        loss = np.dot(w[node1], w[node2]) + b[node1] + b[node2]
        # Clip the loss for numerical stability.
        if loss < -max_loss: loss = -max_loss
        elif loss > max_loss: loss = max_loss
        # Update weights but not gradients (negative pass)
        _update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        _update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        # Update biases in negative pass (not gradients)
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss

@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _glove_reverse(n_edges, w, b,
                   learning_rate=0.01,
                   max_loss=10.):
    """
    Negative sampling GLoVe pass

    negative_edges : array shaped [n_samples, 2]
        We pass in here to hoist out the complexity of 
        handling multithreaded RNG (which plays poorly with numba)
    """
    nnodes = w.shape[0]
    for i in numba.prange(n_edges):
    	# TODO: this thrashes the cache. Find a clever soln
        node1 = np.random.randint(0, nnodes)
        node2 = np.random.randint(0, nnodes)
        # We assume no edge between nodes on negative sampling pass
        # Eg. data[node1, node2] = 0
        loss = np.dot(w[node1], w[node2]) + b[node1] + b[node2]
        # Clip the loss for numerical stability.
        if loss < -max_loss: loss = -max_loss
        elif loss > max_loss: loss = max_loss
        # Update weights but not gradients (negative pass)
        _update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        _update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        # Update biases in negative pass (not gradients)
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss


##########################
#                        #
#       Main method      #
#                        #
##########################

def glove_by_edges(data, dst, src, n_components=2,
                   learning_rate=0.05, max_loss=10.,
                   tol=0.03, tol_samples=10,
                   max_epoch=500, verbose=True):
    """
    Create embeddings using pseudo-GLoVe

    TODO: smart init values
         Based on the indptr array (# of edges corr ~ W value)??
    """
    nnodes = src.size - 1
    w = (np.random.rand(nnodes, n_components) - 0.5)
    b = np.zeros(nnodes, dtype=np.float32)
    latest_loss = [np.inf] * tol_samples
    # rng = np.random.default_rng()
    if verbose:
        epoch_range = tqdm.trange(0, max_epoch)
    else:
        epoch_range = range(0, max_epoch)
    for epoch in epoch_range:
        # Relaxation pass
        _glove_reverse(
            dst.size, w, b,
            learning_rate=learning_rate,
            max_loss=max_loss)
        # Positive "contraction" pass
        # TODO: return only loss max
        loss = _glove_edges_update(
            data, dst, src, w, b,
            learning_rate=learning_rate,
            max_loss=max_loss)
        loss_ = np.mean(np.abs(loss))
        # Pct Change in loss
        max_latest = np.max(latest_loss)
        min_latest = np.min(latest_loss)
        if ((epoch > tol_samples)
            and (np.abs((max_latest - min_latest) / max_latest) < tol)
        ):
            if loss_ < max_loss:
                if verbose: 
                    print(f"Converged! Losses : {latest_loss}")
                return w
            else:
                err_str = (f"Could not learn: loss {loss_} = max loss {max_loss}\n"
                        + "This is often due to too large learning rates.")
                if verbose:
                    print(err_str)
                warnings.warn(err_str)
                break
        elif not np.isfinite(loss).all():
            raise ValueError(
                f"non finite loss: {latest_loss} on epoch {epoch}\n"
            + f"Losses: {loss}\n"
            + f"Previous losses: {[x for x in latest_loss if np.isfinite(x)]}"
            + f"Try Reducing max_loss parameter")
        else:
            latest_loss.append(loss_)
            latest_loss = latest_loss[1:]
            if verbose:
                epoch_range.set_description(f"Loss: {loss_:.4f}\t")
    warnings.warn(f"GVec has not converged. Losses : {latest_loss}")
    return w
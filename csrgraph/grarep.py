import math
from numba import jit
import numpy as np
import pandas as pd
import tqdm
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from csrgraph.methods import _row_norm

def _create_target_matrix(A, tranmat):
    """
    Creating a log transformed target matrix.

    Parameters : 
    ----------------
    A : scipy.sparse.csr_matrix
        Current higher-order matrix
    tranmat : scipy.sparse.csr_matrix
        Transition matrix of the original graph
        It's the per-row normalized adjacency matrix of the graph
    Return :
    ---------------
    scipy.sparse.csr_matrix
        The PMI matrix.
    """
    A = A.dot(tranmat)
    # TODO: add lambda parameter to math.log
    scores = np.log(A.data) - math.log(A.shape[0])
    A.data[scores >= 0] = 0.
    A.eliminate_zeros()
    return A

def grarep_main(src, dst, weights, 
                n_components=2,
                embedder=TruncatedSVD(
                    n_iter=10,
                    random_state=42),
                order=5,
                verbose=True):
    """An implementation of `"GraRep" <https://dl.acm.org/citation.cfm?id=2806512>`
    from the CIKM '15 paper "GraRep: Learning Graph Representations with Global
    Structural Information". The procedure uses sparse truncated SVD to learn
    embeddings for the powers of the PMI matrix computed from powers of the
    normalized adjacency matrix.
    Parameters : 
    ----------------
    n_components (int): 
        Number of individual embedding dimensions.
    order (int): 
        Number of PMI matrix powers.
    embedder : (instance of sklearn API compatible model)
        Should implement the `fit_transform` method: 
            https://scikit-learn.org/stable/glossary.html#term-fit-transform
        The model should also have `n_components` as a parameter
        for number of resulting embedding dimensions. See:
            https://scikit-learn.org/stable/modules/manifold.html#manifold
        If not compatible, set resulting dimensions in the model instance directly

    TODO: add lambda parameter in [0, 1]
            Which is negative sampling ratio

    Returns :
    ---------------
    list[np.array]
    Containing one matrix size (n_nodes, n_components) per order
    """
    embedder.n_components = n_components
    # create transition matrix
    norm_weights = _row_norm(weights, src)
    tranmat = csr_matrix((norm_weights, dst, src))
    target_matrix = tranmat.copy()
    res = []
    if verbose:
        order_range = tqdm.trange(0, order)
    else:
        order_range = range(0, order)
    for _ in order_range:
        target_matrix = _create_target_matrix(target_matrix, tranmat)
        res.append(embedder.fit_transform(target_matrix))
    return res
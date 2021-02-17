from collections.abc import Iterable
from copy import deepcopy
import gc
import networkx as nx
import numba
from numba import jit
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import time
import warnings

from csrgraph.methods import (
    _row_norm, _node_degrees, _src_multiply, _dst_multiply
)
from csrgraph.random_walks import (
    _random_walk, _node2vec_walks
)
from csrgraph import methods, random_walks
from csrgraph import ggvec, glove, grarep

UINT32_MAX = (2**32) - 1

class csrgraph():
    """
    This top level python class either calls external JIT'ed methods
        or methods from the JIT'ed internal graph
    """
    def __init__(self, data, nodenames=None, copy=True, threads=0):
        """
        A class for larger graphs.

        NOTE: this class tends to "steal data" by default.
            If you pass a numpy array/scipy matrix/csr graph
            Chances are this object will point to the same instance of data

        Parameters:
        -------------------

        data : The graph data. Can be one of:
            **NetworkX Graph**
            **Numpy dense matrix**
            **CSR Matrix**
            **(data, indices, indptr)**
            **CSRGraph object**

        nodenames (array of str or int) : Node names
            The position in this array should correspond with the node ID
            So if passing a CSR Matrix or raw data, it should be co-indexed
            with the matrix/raw data arrays

        copy : bool
            Wether to copy passed data to create new object
            Default behavior is to point to underlying ctor passed data
            For networkX graphs and numpy dense matrices we create a new object anyway
        threads : int
            number of threads to leverage for methods in this graph.
            WARNING: changes the numba environment variable to do it.
            Recompiles methods and changes it when changed.
            0 is numba default (usually all threads)

        TODO: add numpy mmap support for very large on-disk graphs
            Should be in a different class
            This also requires routines to read/write edgelists, etc. from disk
        
        TODO: subclass scipy.csr_matrix???
        """
        # If input is a CSRGraph, same object
        if isinstance(data, csrgraph):
            if copy:
                self.mat = data.mat.copy()
                self.names = deepcopy(data.names)
            else:
                self.mat = data.mat
                self.names = data.names
            if not nodenames:
                nodenames = self.names
            else:
                self.names = nodenames
        # NetworkX Graph input
        elif isinstance(data, (nx.Graph, nx.DiGraph)):
            mat = nx.adj_matrix(data)
            mat.data = mat.data.astype(np.float32)
            self.mat = mat
            nodenames = list(data)
        # CSR Matrix Input
        elif isinstance(data, sparse.csr_matrix):
            if copy: self.mat = data.copy()
            else: self.mat = data
        # Numpy Array Input
        elif isinstance(data, np.ndarray):
            self.mat = sparse.csr_matrix(data)
        else:
            raise ValueError(f"Incorrect input data type: {type(data).__name__}")
        # Now that we have the core csr_matrix, alias underlying arrays
        assert hasattr(self, 'mat')
        self.weights = self.mat.data
        self.src = self.mat.indptr
        self.dst = self.mat.indices
        # indptr has one more element than nnodes
        self.nnodes = self.src.size - 1
        # node name -> node ID
        if nodenames is not None:
            self.names = pd.Series(nodenames)
        else:
            self.names = pd.Series(np.arange(self.nnodes))
        # Bounds check once here otherwise there be dragons later
        max_idx = np.max(self.dst)
        if self.nnodes < max_idx:
            raise ValueError(f"""
                Out of bounds node: {max_idx}, nnodes: {self.nnodes}
            """)
        self.set_threads(threads)


    def set_threads(self, threads):
        self.threads = threads
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
            _node_degrees.recompile()
            _src_multiply.recompile()
            _dst_multiply.recompile()

    def __getitem__(self, node):
        """
        [] operator
        like networkX, gets names of neighbor nodes
        """
        # Get node ID from names array
        # This is O(n) by design -- we more often get names from IDs
        #    than we get IDs from names and we don't want to hold 2 maps
        # TODO : replace names with a pd.Index and use get_loc
        node_id = self.names[self.names == node].index[0]
        edges = self.dst[
            self.src[node_id] : self.src[node_id+1]
        ]
        return self.names.iloc[edges].values

    def nodes(self):
        """
        Returns the graph's nodes, in order
        """
        if self.names is not None:
            return self.names
        else:
            return np.arange(self.nnodes)

    def normalize(self, return_self=True):
        """
        Normalizes edge weights per node

        For any node in the Graph, the new edges' weights will sum to 1

        return_self : bool
            whether to change the graph's values and return itself
            this lets us call `G.normalize()` directly
        """
        new_weights = _row_norm(self.weights, self.src)
        if return_self:
            self.mat = sparse.csr_matrix((new_weights, self.dst, self.src))
            # Point objects to the correct places
            self.weights = self.mat.data
            self.src = self.mat.indptr
            self.dst = self.mat.indices
            gc.collect()
            return self
        else:
            return csrgraph(sparse.csr_matrix(
                (new_weights, self.dst, self.src)), 
                nodenames=self.names)

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
        explore_weight : float in (0, inf]
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
            walks = _node2vec_walks(T.weights, T.src, T.dst,
                                    sampling_nodes=sampling_nodes,
                                    walklen=walklen,
                                    return_weight=return_weight,
                                    neighbor_weight=neighbor_weight)
        # much faster implementation for regular walks
        else:
            walks = _random_walk(T.weights, T.src, T.dst,
                                 sampling_nodes, walklen)
        return walks

    def ggvec(self, n_components=2,
              learning_rate=0.05,
              tol="auto",
              max_epoch=500,
              negative_ratio=1.0,
              order=1,
              negative_decay=0.,
              exponent=0.5,
              max_loss=30.,
              tol_samples=100,
              verbose=True):
        """
        GGVec: Fast global first (and higher) order local embeddings.

        This algorithm directly minimizes related nodes' distances.
        It uses a relaxation pass (negative sample) + contraction pass (loss minimization)
        To find stable embeddings based on the minimal dot product of edge weights.

        Parameters:
        -------------
        n_components (int): 
            Number of individual embedding dimensions.
        order : int >= 1
            Meta-level of the embeddings. Improves link prediction performance.
            Setting this higher than 1 ~quadratically slows down algorithm
                Order = 1 directly optimizes the graph.
                Order = 2 optimizes graph plus 2nd order edges
                    (eg. neighbours of neighbours)
                Order = 3 optimizes up to 3rd order edges
            Higher order edges are automatically weighed using GraRep-style graph formation
            Eg. the higher-order graph is from stable high-order random walk distribution.
        negative_ratio : float in [0, 1]
            Negative sampling ratio.
            Setting this higher will do more negative sampling.
            This is slower, but can lead to higher quality embeddings.
        exponent : float
            Weighing exponent in loss function. 
            Having this lower reduces effect of large edge weights.
        tol : float in [0, 1] or "auto"
            Optimization early stopping criterion.
            Stops average loss < tol for tol_samples epochs.
            "auto" sets tol as a function of learning_rate
        tol_samples : int
            Optimization early stopping criterion.
            This is the number of epochs to sample for loss stability.
            Once loss is stable over this number of epochs we stop early.
        negative_decay : float in [0, 1]
            Decay on negative ratio.
            If >0 then negative ratio will decay by (1-negative_decay) ** epoch
            You should usually leave this to 0.
        max_epoch : int
            Stopping criterion.
        max_count : int
            Ceiling value on edge weights for numerical stability
        learning_rate : float in [0, 1]
            Optimization learning rate.
        max_loss : float
            Loss value ceiling for numerical stability.
        """
        if tol == 'auto':
            tol = max(learning_rate / 2, 0.05)
        # Higher order graph embeddings
        # Method inspired by GraRep (powers of transition matrix)
        if order > 1:
            norm_weights = _row_norm(self.weights, self.src)
            tranmat = sparse.csr_matrix((norm_weights, self.dst, self.src))
            target_matrix = tranmat.copy()
            res = np.zeros((self.nnodes, n_components))
            for _ in range(order - 1):
                target_matrix = target_matrix.dot(tranmat)
                w = ggvec.ggvec_main(
                    data=target_matrix.data, 
                    src=target_matrix.indptr, 
                    dst=target_matrix.indices,
                    n_components=n_components, tol=tol,
                    tol_samples=tol_samples,
                    max_epoch=max_epoch, learning_rate=learning_rate, 
                    negative_ratio=negative_ratio,
                    negative_decay=negative_decay,
                    exponent=exponent,
                    max_loss=max_loss, verbose=verbose)
                res = np.sum([res, w], axis=0)
            return res
        else:
            return ggvec.ggvec_main(
                data=self.weights, src=self.src, dst=self.dst,
                n_components=n_components, tol=tol,
                tol_samples=tol_samples,
                max_epoch=max_epoch, learning_rate=learning_rate, 
                negative_ratio=negative_ratio,
                negative_decay=negative_decay,
                exponent=exponent,
                max_loss=max_loss, verbose=verbose)

    def glove(self, n_components=2,
              tol=0.0001, max_epoch=10_000, 
              max_count=50, exponent=0.5,
              learning_rate=0.1, max_loss=10.,
              verbose=True):
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
        w = glove.glove_main(weights=self.weights, dst=self.dst, src=self.src,
                           n_components=n_components, tol=tol,
                           max_epoch=max_epoch, learning_rate=learning_rate,
                           max_count=max_count, exponent=exponent,
                           max_loss=max_loss, verbose=verbose)
        return w

    def grarep(self, n_components=2,
               order=5,
               embedder=TruncatedSVD(
                    n_iter=10,
                    random_state=42),
                verbose=True):
        """Implementation of GraRep: Embeddings of powers of the PMI matrix 
            of the graph adj matrix.

        NOTE: Unlike GGVec and GLoVe, this method returns a LIST OF EMBEDDINGS
              (one per order). You can sum them, or take the last one for embedding.
            Default pooling is to sum : 
                `lambda x : np.sum(x, axis=0)`
            You can also take only the highest order embedding:
                `lambda x : x[-1]`
            Etc.

        Original paper: https://dl.acm.org/citation.cfm?id=2806512

        Parameters : 
        ----------------
        n_components (int): 
            Number of individual embedding dimensions.
        order (int): 
            Number of PMI matrix powers.
            The performance degrades close to quadratically as a factor of this parameter.
            Generally should be kept under 5.
        embedder : (instance of sklearn API compatible model)
            Should implement the `fit_transform` method: 
                https://scikit-learn.org/stable/glossary.html#term-fit-transform
            The model should also have `n_components` as a parameter
            for number of resulting embedding dimensions. See:
                https://scikit-learn.org/stable/modules/manifold.html#manifold
            If not compatible, set resulting dimensions in the model instance directly
        merger : function[list[array]] -> array
            GraRep returns one embedding matrix per order.
            This function reduces it to a single matrix.
        """
        w_array = grarep.grarep_main(weights=self.weights, dst=self.dst, src=self.src,
                           n_components=n_components, order=order,
                           embedder=embedder,verbose=verbose)
        return w_array

    def random_walk_resample(self, walklen=4, epochs=30):
        """
        Create a new graph from random walk co-occurences.

        First, we generate random walks on the graph
        Then, any nodes appearing together in a walk get an edge
        Edge weights are co-occurence counts.

        Recommendation: many short walks > fewer long walks

        TODO: add node2vec walk parameters
        """
        walks = self.random_walks(walklen=walklen, epochs=epochs)
        elist = random_walks.walks_to_edgelist(walks)
        if 'weight' in elist.columns:
            weights = elist.weight.to_numpy()
        else:
            weights = np.ones(dst.shape[0])
        return methods._edgelist_to_graph(
            elist.src.to_numpy(), elist.dst.to_numpy(), 
            weights, nnodes=self.nnodes, nodenames=self.names)

    #
    #
    # TODO: Organize Graph method here
    # Layout nodes by their 1d embedding's position
    # Also layout by hilbert space filling curve
    #
    #

def read_edgelist(f, directed=True, sep=r"\s+", header=None, **readcsvkwargs):
    """
    Creates a csrgraph from an edgelist.

    The edgelist should be in the form 
       [source  destination]
        or 
       [source  destination  edge_weight]

    The first column needs to be the source, the second the destination.
    If there is a third column it's assumed to be edge weights.

    Otherwise, all arguments from pandas.read_csv can be used to read the file.

    f : str
        Filename to read
    directed : bool
        Whether the graph is directed or undirected.
        All csrgraphs are directed, undirected graphs simply add "return edges"
    sep : str
        CSV-style separator. Eg. Use "," if comma separated
    header : int or None
        pandas read_csv parameter. Use if column names are present
    read_csv_kwargs : keyword arguments for pd.read_csv
        Pass these kwargs as you would normally to pd.read_csv.
    Returns : csrgraph
    """
    # Read in csv correctly to each column
    elist = pd.read_csv(f, sep=sep, header=header, **readcsvkwargs)
    if len(elist.columns) == 2:
        elist.columns = ['src', 'dst']
        elist['weight'] = np.ones(elist.shape[0])
    elif len(elist.columns) == 3:
        elist.columns = ['src', 'dst', 'weight']
    else: 
        raise ValueError(f"""
            Invalid columns: {elist.columns}
            Expected 2 (source, destination)
            or 3 (source, destination, weight)
            Read File: \n{elist.head(5)}
        """)
    # Create name mapping to normalize node IDs
    # Somehow this is 1.5x faster than np.union1d. Shame on numpy.
    allnodes = list(
        set(elist.src.unique())
        .union(set(elist.dst.unique())))
    # Factor all nodes to unique IDs
    names = (
        pd.Series(allnodes).astype('category')
        .cat.categories
    )
    nnodes = names.shape[0]
    # Get the input data type
    if nnodes > UINT32_MAX:
        dtype = np.uint64
    else:
        dtype = np.uint32
    name_dict = dict(zip(names,
                         np.arange(names.shape[0], dtype=dtype)))
    elist.src = elist.src.map(name_dict)
    elist.dst = elist.dst.map(name_dict)
    # clean up temp data
    allnodes = None
    name_dict = None
    gc.collect()
    # If undirected graph, append edgelist to reversed self
    if not directed:
        other_df = elist.copy()
        other_df.columns = ['dst', 'src', 'weight']
        elist = pd.concat([elist, other_df])
        other_df = None
        gc.collect()
    # Need to sort by src for _edgelist_to_graph
    elist = elist.sort_values(by='src')
    # extract numpy arrays and clear memory
    src = elist.src.to_numpy()
    dst = elist.dst.to_numpy()
    weight = elist.weight.to_numpy()
    elist = None
    gc.collect()
    G = methods._edgelist_to_graph(
        src, dst, weight,
        nnodes, nodenames=names
    )
    return G

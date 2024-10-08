from collections.abc import Iterable
from copy import deepcopy
import csv
import gc
import io
import networkx as nx
import numba
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv as pa_csv
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import time
import warnings

from csrgraph.numba_backend.methods import (
    _row_norm, _node_degrees, _src_multiply, _dst_multiply
)
from csrgraph.numba_backend.random_walks import (
    _random_walk, _node2vec_walks,_node2vec_walks_with_rejective_sampling, seed_numba
)
from csrgraph.numba_backend import methods, random_walks
from csrgraph.embedding import ggvec, glove, grarep

__all__ = [
    "csrgraph",
    "read_edgelist",
    "from_df",
    "from_tuples",
]

UINT32_MAX = (2**32) - 1
UINT16_MAX = (2**16) - 1

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
        self.read_input_data(data, copy)
        # Now that we have the core csr_matrix, alias underlying arrays
        assert hasattr(self, 'weights')
        assert hasattr(self, 'src')
        assert hasattr(self, 'dst')
        # Convert to Arrow data
        self.weights = pa.Array.from_pandas(self.weights)
        self.src     = pa.Array.from_pandas(self.src)
        self.dst     = pa.Array.from_pandas(self.dst)
        # indptr has one more element than nnodes
        self.nnodes = len(self.src) - 1
        # If no node names, just a range of ints
        if nodenames is not None:
            self.names = nodenames
        if not hasattr(self, 'names'):
            self.names = pd.Series(np.arange(self.nnodes))
        self.names = pa.Array.from_pandas(self.names)
        if self.nnodes != len(self.names):
            raise ValueError(
                "index pointer must have length equal to number of nodes\n"
                + f"nnodes: {self.nnodes}, indptr array: {len(self.src)}\n"
                + f"number of node names: {len(self.names)}"
            )
        # Bounds check once here otherwise there be dragons later
        max_idx = np.max(self.dst)
        if self.nnodes < max_idx:
            raise ValueError(f"""
                Out of bounds node: {max_idx}, nnodes: {self.nnodes}
            """)
        self.set_threads(threads)


    def read_input_data(self, data, copy=True):
        """
        Convert from whatever input data format to our format 
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
        """
        # If input is a CSRGraph, same object
        if isinstance(data, csrgraph):
            if copy:
                self.weights = data.weights.copy()
                self.src = data.src.copy()
                self.dst = data.dst.copy()
                self.names = pd.Series(deepcopy(data.names))
            else:
                self.names = pd.Series(data.names)
                self.weights = data.weights
                self.src = data.indptr
                self.dst = data.indices
        # NetworkX Graph input
        elif isinstance(data, (nx.Graph, nx.DiGraph)):
            mat = nx.adjacency_matrix(data)
            mat.data = mat.data.astype(np.float32)
            self.weights = mat.data
            self.src = mat.indptr
            self.dst = mat.indices
            self.names = pd.Series(list(data))
        # CSR Matrix Input
        elif isinstance(data, (sparse.csr_matrix, sparse.csr_array)):
            if copy: mat = data.copy()
            else: mat = data
            self.weights = mat.data
            self.src = mat.indptr
            self.dst = mat.indices
        # Numpy Array Input
        elif isinstance(data, np.ndarray):
            mat = sparse.csr_matrix(data)
            self.weights = mat.data
            self.src = mat.indptr
            self.dst = mat.indices
        else:
            raise ValueError(f"Incorrect input data type: {type(data).__name__}")



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
        # node_id = self.names.get_loc(node)
        # edges = self.dst[
        #     (self.src[node_id]).as_py()
        #     : (self.src[node_id+1]).as_py()
        # ]
        pc.cast(node, self.names.type)        
        node_idx = pc.index(self.names, node).as_py()
        src_idx_start = self.src[node_idx].as_py()
        src_idx_end = self.src[node_idx + 1].as_py()
        edge_idx = self.dst[src_idx_start: src_idx_end]
        return pc.take(self.names, edge_idx)
        

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
        new_weights = _row_norm(self.weights.to_numpy(), self.src.to_numpy())
        if return_self:
            G_new = csrgraph(sparse.csr_matrix((new_weights, self.dst, self.src)))
            # Point objects to the correct places
            self.weights = G_new.weights
            self.src = G_new.src
            self.dst = G_new.dst
            G_new = None
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
                neighbor_weight=1.,
                rejective_sampling=False, 
                seed=None):
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
        rejective_sampling: bool
            for deepwalk (p=1, q=1), this parameters is of no use
            for node2vec walks, it determines if we use rejective sampling or not
            Rejective sampling is faster, but less stable
            Credit to https://github.com/louisabraham/fastnode2vec for original idea

        Returns
        -------
        out : 2d np.array (n_walks, walklen)
            A matrix where each row is a random walk,
            and each entry is the ID of the node
        """
        if seed is not None:
            seed = int(seed)
            seed_numba(seed)
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
            if rejective_sampling:
                walks = _node2vec_walks_with_rejective_sampling(
                    T.weights.to_numpy(), T.src.to_numpy(), T.dst.to_numpy(),
                    sampling_nodes=sampling_nodes,
                    walklen=walklen,
                    return_weight=return_weight,
                    neighbor_weight=neighbor_weight)
            else:
                walks = _node2vec_walks(
                    T.weights.to_numpy(), T.src.to_numpy(), T.dst.to_numpy(),
                    sampling_nodes=sampling_nodes,
                    walklen=walklen,
                    return_weight=return_weight,
                    neighbor_weight=neighbor_weight)
        # much faster implementation for regular walks
        else:
            walks = _random_walk(
                 T.weights.to_numpy(), T.src.to_numpy(), T.dst.to_numpy(),
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
            norm_weights = _row_norm(self.weights.to_numpy(), self.src.to_numpy())
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
                data=self.weights.to_numpy(), 
                src=self.src.to_numpy(), 
                dst=self.dst.to_numpy(),
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
        w = glove.glove_main(
            weights=self.weights.to_numpy(),
            dst=self.dst.to_numpy(),
            src=self.src.to_numpy(),
            n_components=n_components, tol=tol,
            max_epoch=max_epoch, learning_rate=learning_rate,
            max_count=max_count, exponent=exponent,
            max_loss=max_loss, verbose=verbose
        )
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
        w_array = grarep.grarep_main(
            weights=self.weights.to_numpy(),
            dst=self.dst.to_numpy(),
            src=self.src.to_numpy(),
            n_components=n_components, order=order,
            embedder=embedder, verbose=verbose
        )
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


    def to_scipy(self):
        return sparse.csr_array((self.weights, self.dst, self.src))


    def to_numpy(self):
        return self.to_scipy().todense()
    #
    #
    # TODO: Organize Graph method here
    # Layout nodes by their 1d embedding's position
    # Also layout by hilbert space filling curve
    #
    #

def read_edgelist(f, 
        directed=True, 
        sep="infer", 
        header="infer", 
        keep_default_na=False, 
        **readcsvkwargs):
    """
    Creates a csrgraph from an edgelist file.

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
    keep_default_na: bool
        pandas read_csv argument to prevent casting any value to NaN
    read_csv_kwargs : keyword arguments for pd.read_csv
        Pass these kwargs as you would normally to pd.read_csv.
    Returns : csrgraph
    """
    with open(f, encoding='utf-8', errors='ignore') if isinstance(f, str) \
            else io.TextIOWrapper(f, encoding='utf-8') as fname:
        csv_sniff = csv.Sniffer()
        sample = fname.read(1024)
        dialect = csv_sniff.sniff(sample)
        # Parse header
        skip_rows_before = 0
        if header == "infer":
            has_header = False
            has_header = csv_sniff.has_header(sample)
        elif header is None:
            has_header = False
        if sep == 'infer':
            sep = dialect.delimiter
        else:
            if len(sep) != 1:
                raise ValueError(f"sep must be 1 char, got {sep}")
            sep = sep
        read_options = pa_csv.ReadOptions(
            use_threads=True, 
            block_size=None, 
            skip_rows=0, # Rows to ignore before column names
            skip_rows_after_names=0, # Rows to ignore after column names
            column_names=None,
            # Autogenerate columns if there's no header only
            autogenerate_column_names=not has_header, 
            encoding='utf8'
        )
        parse_options = pa_csv.ParseOptions(
            delimiter=sep,
            quote_char=dialect.quotechar,
            double_quote=dialect.doublequote,
            escape_char=False,
            newlines_in_values=False,
            ignore_empty_lines=True,
            invalid_row_handler=None
        )
        convert_options = pa_csv.ConvertOptions(
            check_utf8=True, 
            column_types=None,
            null_values=None,
            true_values=None,
            false_values=None,
            decimal_point=".",
            strings_can_be_null=False,
            quoted_strings_can_be_null=True,
            include_columns=None,
            include_missing_columns=False,
            auto_dict_encode=False,
            auto_dict_max_cardinality=None,
            timestamp_parsers=None
        )
        # Reset file cursor to start for full read
        fname.seek(0)
        elist = pa_csv.read_csv(f, 
            read_options=read_options, 
            parse_options=parse_options,
            convert_options=convert_options
        ).to_pandas()
    return from_df(elist, directed=directed)


def edgelist_to_ipc():
    """
    Translates edgelist  
    """


def from_df(elist: pd.DataFrame, directed: bool = True) -> csrgraph:
    """
    Creates a csrgraph from a DataFrame of either two or three columns.

    elist :
        Either a DataFrame with two columns for source and target or three
        columns for source, target, and weight.
    Returns : csrgraph
    """
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
    dtype = np.uint16
    if nnodes > UINT16_MAX:
        dtype = np.uint32
    if nnodes > UINT32_MAX:
        dtype = np.uint64
    name_dict = dict(zip(names,
                         np.arange(names.shape[0], dtype=dtype)))
    elist.src = elist.src.map(name_dict).astype(dtype)
    elist.dst = elist.dst.map(name_dict).astype(dtype)
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


def from_tuples(tuples, directed: bool = False) -> csrgraph:
    """
    Creates a csrgraph from an iterable of edge tuples.

    tuples : iterable[tuple[str, str]] or iterable[tuple[str, str, str]]]
        Either an iterable of source, target pairs or an iterable of
        source, target, weight triples
    Returns : csrgraph
    """
    elist = pd.DataFrame(tuples)
    return from_df(elist, directed=directed)

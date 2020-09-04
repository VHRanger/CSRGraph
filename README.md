[![Build Status](https://travis-ci.com/VHRanger/CSRGraph.svg?branch=master)](https://travis-ci.com/VHRanger/CSRGraph)


# CSRGraphs

This library aims to be a graph analogue to Pandas. It aims to provide fast and memory efficient operations for read-only graphs.

By exploiting [CSR Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)), we can efficiently pack and traverse graphs in memory (as long as we don't add/delete nodes often).

# Accessing the underlying matrix

```python
import csrgraph as cg

G = cg.csrgraph(data)
G.mat # This is the underlying scipy.sparse.csr_matrix
```

For instance, all the procedures in scipy `csgraph` module [here](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html) will function directly on the `G.mat` object. 

The rest of the `csrgraph` class is mainly around managing node names, ingesting, and optimized (numba compiled) methods.

# Installation

`pip install csrgraph`

# TODO: 

- Add support for directed graphs

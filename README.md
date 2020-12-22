[![Build Status](https://travis-ci.com/VHRanger/CSRGraph.svg?branch=master)](https://travis-ci.com/VHRanger/CSRGraph)


# CSRGraphs

This library aims to provide fast and memory efficient operations for large read-only graphs.

By exploiting [CSR Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)), we can efficiently pack and traverse graphs in memory (as long as we don't add/delete nodes often).

# Installation

`pip install csrgraph`

# Reading in Graphs

You can read a graph from an **edgelist**:

```python
import csrgraph as cg

G = cg.read_edgelist("path_to_file.csv", directed=False, sep=',')
# Node names are stored and accessible
G['cat']
```

Other objects that can be put in the csrgraph constructor:

- **NetworkX Graph**. CSRGraph will support edge weights natively.
- **Numpy dense matrix** Should be square, one row/column per node.
- **CSR Matrix**. Same as above. This also supports the **(data, indices, indptr)** format.

# Methods on graphs

Currently, we support very fast random walks with `G.random_walks()`. The GloVe, GraRep and GGVec graph embedding algorithms are also directly accessible from the graph object.

### CSGraph methods

You can access the underlying `scipy.sparse.csr_matrix` with the `.mat` accessor.

```python
import csrgraph as cg

G = cg.csrgraph(data)
G.mat # This is the underlying scipy.sparse.csr_matrix
```

All the procedures in scipy `csgraph` module [here](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html) will function directly on the `G.mat` object.

# Gotchas

**All graphs are directed**. We support undirected graphs by adding "return edges" on each edge. The only issue is that this doubles the number of edges. You'll generally find CSRGraph's efficient memory use and fast methods makes up for this design.

**4.2B Node limit**. You can't currently have more nodes than `UINT32_MAX` (around 4.2billion) -- you'll run out of node IDs. You can have as many edges as will fit in RAM however.

**Only float edge weights** Eventually we might support complex edge weight objects, but for now we only support 32bit floats.


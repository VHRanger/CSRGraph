[![Build Status](https://travis-ci.com/VHRanger/CSRGraph.svg?branch=master)](https://travis-ci.com/VHRanger/CSRGraph)


# CSRGraphs

This library aims to be a graph analogue to Pandas. It aims to provide fast and memory efficient operations for read-only graphs.

By exploiting [CSR Sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)), we can efficiently pack and traverse graphs in memory (as long as we don't add/delete nodes often).

# Installation

`pip install csrgraph`

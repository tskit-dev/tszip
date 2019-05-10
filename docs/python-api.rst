.. _sec_python_api:

==========
Python API
==========

This page provides detailed documentation for the ``tszip`` Python API.

*************
Usage example
*************

Tszip can be used directly in Python to provide seamless compression and
decompression of tree sequences files. Here, we run an msprime simulation
and write the output to a ``.trees.tsz`` file:

.. code-block:: python

    import msprime
    import tszip

    ts = msprime.simulate(10, random_seed=1)
    tszip.compress(ts, "simulation.trees.tsz")

    # Later, we load the same tree sequence from the compressed file.
    ts = tszip.decompress("simulation.trees.tsz")


.. note:: For very small simulations like this example, the tszip file may
    be larger than the original uncompressed file.

***
API
***

.. autofunction:: tszip.compress

.. autofunction:: tszip.decompress

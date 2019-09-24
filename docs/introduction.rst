.. _sec_introduction:

============
Introduction
============

This is the documentation for tszip, a command line interface and Python API
for compressing `tskit <https://tskit.readthedocs.io/>`_ tree sequence files
used by `msprime <https://msprime.readthedocs.io/en/stable/>`_,
`SLiM <https://messerlab.org/slim/>`_, `fwdpy11
<https://fwdpy11.readthedocs.io/en/stable/pages/tsoverview.html>`_
and `tsinfer <https://tsinfer.readthedocs.io/>`_. Tszip achieves much better
compression than is possible using generic compression utilities
by building on the `zarr <https://zarr.readthedocs.io/en/stable/>`_
and `numcodecs <https://numcodecs.readthedocs.io/en/stable/>`_ packages.

The command line interface follows the design of `gzip <https://en.wikipedia.org/wiki/Gzip>`_
closely, so should be immediately familiar. Here we compress a large tree sequence
representing 1000 Genomes chromosome 22 using ``tszip`` and decompress it using
``tsunzip``:

.. code-block:: bash

    $ ls -lh
    total 297M
    -rw-r--r-- 1 jk jk 297M May 10 14:49 1kg_chr20.trees
    $ tszip 1kg_chr20.trees
    $ ls -lh
    total 46M
    -rw-r--r-- 1 jk jk 46M May 10 14:51 1kg_chr20.trees.tsz
    $ tsunzip 1kg_chr20.trees.tsz
    $ ls -lh
    total 297M
    -rw-r--r-- 1 jk jk 297M May 10 14:52 1kg_chr20.trees






(sec_welcome)=

# Welcome to tszip

``tszip`` is a command line interface and Python API for compressing
[tskit](https://tskit.dev/tskit) tree sequence files produced
and read by software projects in the [tskit ecosystem](https://tskit.dev/software)
such as [msprime](https://tskit.dev/software/msprime.html),
[SLiM](https://messerlab.org/slim/), [fwdpy11](https://tskit.dev/software/fwdpy11.html)
and [tsinfer](https://tskit.dev/software/tsinfer.html). Tszip achieves much better
compression than is possible using generic compression utilities by building on
the [zarr](https://zarr.readthedocs.io/en/stable/) and
[numcodecs](https://numcodecs.readthedocs.io/en/stable/) packages.

The command line interface follows the design of
[gzip](https://en.wikipedia.org/wiki/Gzip) closely, so should be immediately familiar.
Here we compress a large tree sequence representing 1000 Genomes chromosome 22 using
`tszip` and decompress it using ``tsunzip``:

```{code-block} bash
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
```

```{tableofcontents}
```

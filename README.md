# tszip
[![License](https://img.shields.io/github/license/tskit-dev/tszip)](https://github.com/tskit-dev/tszip/blob/main/LICENSE) [![PyPI version](https://img.shields.io/pypi/v/tszip.svg)](https://pypi.org/project/tszip/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/tszip.svg)](https://pypi.org/project/tszip/) [![Docs Build](https://github.com/tskit-dev/tszip/actions/workflows/docs.yml/badge.svg)](https://github.com/tskit-dev/tszip/actions/workflows/docs.yml) [![Binary wheels](https://github.com/tskit-dev/tszip/actions/workflows/wheels.yml/badge.svg)](https://github.com/tskit-dev/tszip/actions/workflows/wheels.yml) [![Tests](https://github.com/tskit-dev/tszip/actions/workflows/tests.yml/badge.svg)](https://github.com/tskit-dev/tszip/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/tskit-dev/tszip/branch/main/graph/badge.svg)](https://codecov.io/gh/tskit-dev/tszip)

Gzip-like compression for [tskit](https://tskit.dev/software/tskit.html) tree sequences. Compression is lossless for supported tskit tree sequences.

Please see the [documentation](https://tskit.dev/tszip/docs/stable/) ([latest](https://tskit.dev/tszip/docs/latest/)) for more details
and [installation instructions](https://tskit.dev/tszip/docs/stable/installation.html).

## Installation

Install from PyPI or conda-forge:

```
python -m pip install tszip
# or
conda install -c conda-forge tszip
```

## Quickstart

CLI usage:

```bash
# Compress a .trees file to a .tsz archive
tszip data.trees

# Decompress back to .trees
tsunzip data.trees.tsz
```

Along with the CLI, tszip can be used directly from Python:

```python
import tskit
import tszip

ts = tskit.load("data.trees")
tszip.compress(ts, "data.trees.tsz")  # write compressed archive

# load handles .tsz archives and plain .trees files
restored = tszip.load("data.trees.tsz")
print(restored.equals(ts))  # True
```

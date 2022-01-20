# MIT License
#
# Copyright (c) 2019 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Compression utilities for tskit tree sequences.
"""
import contextlib
import json
import logging
import os.path
import tempfile
import warnings
import zipfile
from typing import Mapping

import humanize
import numcodecs
import numpy as np
import tskit
import zarr

from . import exceptions
from . import provenance

logger = logging.getLogger(__name__)

FORMAT_NAME = "tszip"
FORMAT_VERSION = [1, 0]


def minimal_dtype(array):
    """
    Returns the smallest dtype that can be used to represent the values in the
    specified array. If the array is not one of the integer types, the dtype of the
    array itself is returned directly.
    """
    dtype = array.dtype
    if array.shape[0] == 0:
        return dtype
    if dtype.kind == "u":
        maxval = np.max(array)
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
        for dtype in map(np.dtype, dtypes):
            if maxval <= np.iinfo(dtype).max:
                break
    elif dtype.kind == "i":
        minval = np.min(array)
        maxval = np.max(array)
        dtypes = [np.int8, np.int16, np.int32, np.int64]
        for dtype in map(np.dtype, dtypes):
            if np.iinfo(dtype).min <= minval and maxval <= np.iinfo(dtype).max:
                break
    return dtype


def compress(ts, path, variants_only=False):
    """
    Compresses the specified tree sequence and writes it to the specified path.
    By default, fully lossless compression is used so that tree sequences are
    identical before and after compression. By specifying the ``variants_only``
    option, a lossy compression can be used, which discards any information
    that is not needed to represent the variants (which are stored losslessly).

    :param tskit.TreeSequence ts: The input tree sequence.
    :param str destination: The string or :class:`pathlib.Path` instance describing
        the location of the compressed file.
    :param bool variants_only: If True, discard all information not necessary
        to represent the variants in the input file.
    """
    destination = str(path)
    # Write the file into a temporary directory on the same file system so that
    # we can write the output atomically.
    destdir = os.path.dirname(os.path.abspath(destination))
    with tempfile.TemporaryDirectory(dir=destdir, prefix=".tszip_work_") as tmpdir:
        filename = os.path.join(tmpdir, "tmp.trees.tgz")
        logging.debug(f"Writing to temporary file {filename}")
        with zarr.ZipStore(filename, mode="w") as store:
            root = zarr.group(store=store)
            compress_zarr(ts, root, variants_only=variants_only)
        os.replace(filename, destination)
    logging.info(f"Wrote {destination}")


def decompress(path):
    """
    Decompresses the tszip compressed file and returns a tskit tree sequence
    instance.

    :param str path: The location of the tszip compressed file to load.
    :rtype: tskit.TreeSequence
    :return: A :class:`tskit.TreeSequence` instance corresponding to the
        the specified file.
    """
    with load_zarr(path) as root:
        return decompress_zarr(root)


class Column:
    """
    A single column that is stored in the compressed output.
    """

    def __init__(self, name, array, delta_filter=False):
        self.name = name
        self.array = array
        self.delta_filter = delta_filter

    def compress(self, root, compressor):
        array_type = "nparray"
        if isinstance(self.array, str):
            self.array = np.frombuffer(self.array.encode("utf-8"), np.int8)
            array_type = "str"
        elif isinstance(self.array, bytes):
            self.array = np.frombuffer(self.array, np.int8)
            array_type = "bytes"

        shape = self.array.shape
        chunks = shape
        if shape[0] == 0:
            chunks = (1,)
        dtype = minimal_dtype(self.array)
        filters = None
        if self.delta_filter:
            filters = [numcodecs.Delta(dtype=dtype)]
        compressed_array = root.empty(
            self.name,
            chunks=chunks,
            shape=shape,
            dtype=dtype,
            filters=filters,
            compressor=compressor,
        )
        compressed_array.attrs["tszip_type"] = array_type
        compressed_array[:] = self.array
        ratio = 0
        if compressed_array.nbytes > 0:
            ratio = compressed_array.nbytes / compressed_array.nbytes_stored
        logger.debug(
            "{}: output={} compression={:.1f}".format(
                self.name,
                humanize.naturalsize(compressed_array.nbytes_stored, binary=True),
                ratio,
            )
        )


def compress_zarr(ts, root, variants_only=False):

    provenance_dict = provenance.get_provenance_dict({"variants_only": variants_only})

    if variants_only:
        logging.info("Using lossy variants-only compression")
        # Reduce to site topology. Note that we will remove
        # any sites, individuals and populations here that have no references.
        ts = ts.simplify(reduce_to_site_topology=True)

    tables = ts.tables

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # When using a zipfile in Zarr we get some harmless warnings. See
        # https://zarr.readthedocs.io/en/stable/api/storage.html#zarr.storage.ZipStore
        root.attrs["format_name"] = FORMAT_NAME
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["sequence_length"] = tables.sequence_length
        root.attrs["provenance"] = provenance_dict

    columns = {}
    for key, value in tables.asdict().items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                columns[f"{key}/{sub_key}"] = sub_value
        else:
            columns[key] = value

    if variants_only:
        time = np.unique(tables.nodes.time)
        columns["node/time"] = np.searchsorted(time, tables.nodes.time)

    # Encoding array is a tuple so must be converted
    columns["encoding_version"] = np.asarray(columns["encoding_version"])

    # Sequence length is stored as an attr for compatibility with older versions of tszip
    del columns["sequence_length"]

    # Some columns benefit from being quantised
    coordinates = np.unique(
        np.hstack(
            [
                [0, ts.sequence_length],
                tables.edges.left,
                tables.edges.right,
                tables.sites.position,
                tables.migrations.left,
                tables.migrations.right,
            ]
        )
    )
    columns["coordinates"] = coordinates
    for name in [
        "edges/left",
        "edges/right",
        "migrations/left",
        "migrations/right",
        "sites/position",
    ]:
        columns[name] = np.searchsorted(coordinates, columns[name])

    # Some columns benefit from additional options
    delta_filter_cols = ["edges/parent", "sites/position"]

    # Note: we're not providing any options to set this here because Blosc+Zstd seems to
    # have a clear advantage in compression performance and speed. There is very little
    # difference between compression level 6 and 9, and it's extremely fast in any case
    # so there's no point in adding complexity. The shuffle filter in particular makes
    # big difference.
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=9, shuffle=numcodecs.Blosc.SHUFFLE
    )
    for name, data in columns.items():
        Column(
            name, data, delta_filter="_offset" in name or name in delta_filter_cols
        ).compress(root, compressor)


def check_format(root):
    try:
        format_name = root.attrs["format_name"]
        format_version = root.attrs["format_version"]
    except KeyError as ke:
        raise exceptions.FileFormatError("Incorrect file format") from ke
    if format_name != FORMAT_NAME:
        raise exceptions.FileFormatError(
            "Incorrect file format: expected '{}' got '{}'".format(
                FORMAT_NAME, format_name
            )
        )
    if format_version[0] < FORMAT_VERSION[0]:
        raise exceptions.FileFormatError(
            "Format version {} too old. Current version = {}".format(
                format_version, FORMAT_VERSION
            )
        )
    if format_version[0] > FORMAT_VERSION[0]:
        raise exceptions.FileFormatError(
            "Format version {} too new. Current version = {}".format(
                format_version, FORMAT_VERSION
            )
        )


@contextlib.contextmanager
def load_zarr(path):
    path = str(path)
    try:
        store = zarr.ZipStore(path, mode="r")
    except zipfile.BadZipFile as bzf:
        raise exceptions.FileFormatError("File is not in tgzip format") from bzf
    root = zarr.group(store=store)
    try:
        check_format(root)
        yield root
    finally:
        store.close()


def decompress_zarr(root):
    coordinates = root["coordinates"][:]
    dict_repr = {"sequence_length": root.attrs["sequence_length"]}

    quantised_arrays = [
        "edges/left",
        "edges/right",
        "migrations/left",
        "migrations/right",
        "sites/position",
    ]
    for key, value in root.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if f"{key}/{sub_key}" in quantised_arrays:
                    dict_repr.setdefault(key, {})[sub_key] = coordinates[sub_value]
                elif sub_key.endswith("metadata_schema"):
                    dict_repr.setdefault(key, {})[sub_key] = bytes(sub_value).decode(
                        "utf-8"
                    )
                else:
                    dict_repr.setdefault(key, {})[sub_key] = sub_value
        elif value.attrs.get("tszip_type") == "str":
            dict_repr[key] = bytes(value).decode("utf-8")
        elif value.attrs.get("tszip_type") == "bytes":
            dict_repr[key] = bytes(value)
        # We manually check these for legacy tszips that didn't store type info.
        elif key.endswith("metadata_schema"):
            dict_repr[key] = bytes(value).decode("utf-8")
        elif key.endswith("metadata"):
            dict_repr[key] = bytes(value)
        else:
            dict_repr[key] = value

    return tskit.TableCollection.fromdict(dict_repr).tree_sequence()


def print_summary(path, verbosity=0):
    arrays = []

    def visitor(array):
        if isinstance(array, zarr.core.Array):
            arrays.append(array)

    with load_zarr(path) as root:
        root.visitvalues(visitor)

    arrays.sort(key=lambda x: x.nbytes_stored)
    max_name_len = max(len(array.name) for array in arrays)
    stored = [
        humanize.naturalsize(array.nbytes_stored, binary=True) for array in arrays
    ]
    max_stored_len = max(len(size) for size in stored)
    actual = [humanize.naturalsize(array.nbytes, binary=True) for array in arrays]
    max_actual_len = max(len(size) for size in actual)

    line = "File: {}\t{}".format(
        path, humanize.naturalsize(os.path.getsize(path), binary=True)
    )
    print(line)
    if verbosity > 0:
        print("format_version:", root.attrs["format_version"])
        prov = root.attrs["provenance"]
        print("provenance: ", end="")
        print(json.dumps(prov, indent=4, sort_keys=True))
    fmt = "{:<{}} {:<{}}\t{:<{}}\t{}"
    line = fmt.format(
        "name",
        max_name_len,
        "stored",
        max_stored_len,
        "actual",
        max_actual_len,
        "ratio",
    )
    print(line)
    for array, stored, actual in zip(arrays, stored, actual):
        ratio = 0
        if array.nbytes > 0:
            ratio = array.nbytes_stored / array.nbytes
        line = fmt.format(
            array.name,
            max_name_len,
            stored,
            max_stored_len,
            actual,
            max_actual_len,
            f"{ratio:.2f}",
        )
        print(line)
        if verbosity > 0:
            for line in str(array.info).splitlines():
                print("\t", line)

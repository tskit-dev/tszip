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
import logging
import os
import warnings
import contextlib
import zipfile
import tempfile
import os.path
import json

import numcodecs
import zarr
import numpy as np
import tskit
import humanize

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
    if dtype.kind == 'u':
        maxval = np.max(array)
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
        for dtype in map(np.dtype, dtypes):
            if maxval <= np.iinfo(dtype).max:
                break
    elif dtype.kind == 'i':
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
        logging.debug("Writing to temporary file {}".format(filename))
        with zarr.ZipStore(filename, mode='w') as store:
            root = zarr.group(store=store)
            compress_zarr(ts, root, variants_only=variants_only)
        os.replace(filename, destination)
    logging.info("Wrote {}".format(destination))


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


class Column(object):
    """
    A single column that is stored in the compressed output.
    """
    def __init__(self, name, array, delta_filter=False):
        self.name = name
        self.array = array
        self.delta_filter = delta_filter

    def compress(self, root, compressor):
        shape = self.array.shape
        chunks = shape
        if shape[0] == 0:
            chunks = 1,
        dtype = minimal_dtype(self.array)
        filters = None
        if self.delta_filter:
            filters = [numcodecs.Delta(dtype=dtype)]
        compressed_array = root.empty(
            self.name, chunks=chunks, shape=shape, dtype=dtype,
            filters=filters, compressor=compressor)
        compressed_array[:] = self.array
        ratio = 0
        if compressed_array.nbytes > 0:
            ratio = compressed_array.nbytes / compressed_array.nbytes_stored
        logger.debug("{}: output={} compression={:.1f}".format(
            self.name,
            humanize.naturalsize(compressed_array.nbytes_stored, binary=True),
            ratio))


def compress_zarr(ts, root, variants_only=False):

    provenance_dict = provenance.get_provenance_dict({"variants_only": variants_only})

    if variants_only:
        logging.info("Using lossy variants-only compression")
        # Reduce to site topology and quantise node times. Note that we will remove
        # any sites, individuals and populations here that have no references.
        ts = ts.simplify(reduce_to_site_topology=True)
        tables = ts.tables
        time = np.unique(tables.nodes.time)
        node_time = np.searchsorted(time, tables.nodes.time)
    else:
        tables = ts.tables
        node_time = tables.nodes.time

    coordinates = np.unique(np.hstack([
        [0, ts.sequence_length], tables.edges.left, tables.edges.right,
        tables.sites.position, tables.migrations.left, tables.migrations.right]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # When using a zipfile in Zarr we get some harmless warnings. See
        # https://zarr.readthedocs.io/en/stable/api/storage.html#zarr.storage.ZipStore
        root.attrs["format_name"] = FORMAT_NAME
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["sequence_length"] = tables.sequence_length
        root.attrs["provenance"] = provenance_dict
        if tables.metadata_schema.schema is not None:
            root.attrs["metadata_schema"] = tables.metadata_schema.schema
            root.attrs["metadata"] = tables.metadata

    columns = [
        Column("coordinates", coordinates),
        Column("individuals/flags", tables.individuals.flags),
        Column("individuals/location", tables.individuals.location),
        Column(
            "individuals/location_offset", tables.individuals.location_offset,
            delta_filter=True),
        Column("individuals/metadata", tables.individuals.metadata),
        Column(
            "individuals/metadata_offset", tables.individuals.metadata_offset,
            delta_filter=True),

        Column("nodes/time", node_time),
        Column("nodes/flags", tables.nodes.flags),
        Column("nodes/population", tables.nodes.population),
        Column("nodes/individual", tables.nodes.individual),
        Column("nodes/metadata", tables.nodes.metadata),
        Column(
            "nodes/metadata_offset", tables.nodes.metadata_offset, delta_filter=True),

        # Delta filtering makes storage slightly worse for everything except parent.
        Column("edges/left", np.searchsorted(coordinates, tables.edges.left)),
        Column("edges/right", np.searchsorted(coordinates, tables.edges.right)),
        Column("edges/parent", tables.edges.parent, delta_filter=True),
        Column("edges/child", tables.edges.child),

        Column("migrations/left", np.searchsorted(coordinates, tables.migrations.left)),
        Column(
            "migrations/right", np.searchsorted(coordinates, tables.migrations.right)),
        Column("migrations/node", tables.migrations.node),
        Column("migrations/source", tables.migrations.source),
        Column("migrations/dest", tables.migrations.dest),
        Column("migrations/time", tables.migrations.time),

        Column(
            "sites/position", np.searchsorted(coordinates, tables.sites.position),
            delta_filter=True),
        Column("sites/ancestral_state", tables.sites.ancestral_state),
        Column("sites/ancestral_state_offset", tables.sites.ancestral_state_offset),
        Column("sites/metadata", tables.sites.metadata),
        Column("sites/metadata_offset", tables.sites.metadata_offset),

        Column("mutations/site", tables.mutations.site),
        Column("mutations/node", tables.mutations.node),
        Column("mutations/parent", tables.mutations.parent),
        Column("mutations/derived_state", tables.mutations.derived_state),
        Column("mutations/derived_state_offset", tables.mutations.derived_state_offset),
        Column("mutations/metadata", tables.mutations.metadata),
        Column("mutations/metadata_offset", tables.mutations.metadata_offset),

        Column("populations/metadata", tables.populations.metadata),
        Column("populations/metadata_offset", tables.populations.metadata_offset),

        Column("provenances/timestamp", tables.provenances.timestamp),
        Column("provenances/timestamp_offset", tables.provenances.timestamp_offset),
        Column("provenances/record", tables.provenances.record),
        Column("provenances/record_offset", tables.provenances.record_offset),
    ]

    # Note: we're not providing any options to set this here because Blosc+Zstd seems to
    # have a clear advantage in compression performance and speed. There is very little
    # difference between compression level 6 and 9, and it's extremely fast in any case
    # so there's no point in adding complexity. The shuffle filter in particular makes
    # big difference.
    compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
    for column in columns:
        column.compress(root, compressor)


def check_format(root):
    try:
        format_name = root.attrs["format_name"]
        format_version = root.attrs["format_version"]
    except KeyError as ke:
        raise exceptions.FileFormatError("Incorrect file format") from ke
    if format_name != FORMAT_NAME:
        raise exceptions.FileFormatError(
            "Incorrect file format: expected '{}' got '{}'".format(
                FORMAT_NAME, format_name))
    if format_version[0] < FORMAT_VERSION[0]:
        raise exceptions.FileFormatError(
            "Format version {} too old. Current version = {}".format(
                format_version, FORMAT_VERSION))
    if format_version[0] > FORMAT_VERSION[0]:
        raise exceptions.FileFormatError(
            "Format version {} too new. Current version = {}".format(
                format_version, FORMAT_VERSION))


@contextlib.contextmanager
def load_zarr(path):
    path = str(path)
    try:
        store = zarr.ZipStore(path, mode='r')
    except zipfile.BadZipFile as bzf:
        raise exceptions.FileFormatError("File is not in tgzip format") from bzf
    root = zarr.group(store=store)
    try:
        check_format(root)
        yield root
    finally:
        store.close()


def decompress_zarr(root):
    tables = tskit.TableCollection(root.attrs["sequence_length"])
    coordinates = root["coordinates"][:]
    if "metadata_schema" in root.attrs:
        tables.metadata_schema = tskit.MetadataSchema(root.attrs["metadata_schema"])
    if "metadata" in root.attrs:
        tables.metadata = root.attrs["metadata"]

    tables.individuals.set_columns(
        flags=root["individuals/flags"],
        location=root["individuals/location"],
        location_offset=root["individuals/location_offset"],
        metadata=root["individuals/metadata"],
        metadata_offset=root["individuals/metadata_offset"])

    tables.nodes.set_columns(
        flags=root["nodes/flags"],
        time=root["nodes/time"],
        population=root["nodes/population"],
        individual=root["nodes/individual"],
        metadata=root["nodes/metadata"],
        metadata_offset=root["nodes/metadata_offset"])

    tables.edges.set_columns(
        left=coordinates[root["edges/left"]],
        right=coordinates[root["edges/right"]],
        parent=root["edges/parent"],
        child=root["edges/child"])

    tables.migrations.set_columns(
        left=coordinates[root["migrations/left"]],
        right=coordinates[root["migrations/right"]],
        node=root["migrations/node"],
        source=root["migrations/source"],
        dest=root["migrations/dest"],
        time=root["migrations/time"])

    tables.sites.set_columns(
        position=coordinates[root["sites/position"]],
        ancestral_state=root["sites/ancestral_state"],
        ancestral_state_offset=root["sites/ancestral_state_offset"],
        metadata=root["sites/metadata"],
        metadata_offset=root["sites/metadata_offset"])

    tables.mutations.set_columns(
        site=root["mutations/site"],
        node=root["mutations/node"],
        parent=root["mutations/parent"],
        derived_state=root["mutations/derived_state"],
        derived_state_offset=root["mutations/derived_state_offset"],
        metadata=root["mutations/metadata"],
        metadata_offset=root["mutations/metadata_offset"])

    tables.populations.set_columns(
        metadata=root["populations/metadata"],
        metadata_offset=root["populations/metadata_offset"])

    tables.provenances.set_columns(
        timestamp=root["provenances/timestamp"],
        timestamp_offset=root["provenances/timestamp_offset"],
        record=root["provenances/record"],
        record_offset=root["provenances/record_offset"])

    return tables.tree_sequence()


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
        humanize.naturalsize(array.nbytes_stored, binary=True) for array in arrays]
    max_stored_len = max(len(size) for size in stored)
    actual = [
        humanize.naturalsize(array.nbytes, binary=True) for array in arrays]
    max_actual_len = max(len(size) for size in actual)

    line = "File: {}\t{}".format(
        path,
        humanize.naturalsize(os.path.getsize(path), binary=True))
    print(line)
    if verbosity > 0:
        print("format_version:", root.attrs["format_version"])
        prov = root.attrs["provenance"]
        print("provenance: ", end="")
        print(json.dumps(prov, indent=4, sort_keys=True))
    fmt = "{:<{}} {:<{}}\t{:<{}}\t{}"
    line = fmt.format(
        "name", max_name_len, "stored", max_stored_len, "actual", max_actual_len,
        "ratio")
    print(line)
    for array, stored, actual in zip(arrays, stored, actual):
        ratio = 0
        if array.nbytes > 0:
            ratio = array.nbytes_stored / array.nbytes
        line = fmt.format(
            array.name, max_name_len, stored, max_stored_len, actual, max_actual_len,
            "{:.2f}".format(ratio))
        print(line)
        if verbosity > 0:
            for line in str(array.info).splitlines():
                print("\t", line)

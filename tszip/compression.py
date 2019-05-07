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

import numcodecs
import zarr
import numpy as np
import tskit
import humanize

logger = logging.getLogger(__name__)

FORMAT_NAME = "tszip"
FORMAT_VERSION = (1, 0)


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


def compress(ts, path, compressor=None, variants_only=False):
    """
    Compresses the specified tree sequence and writes it to the specified
    path.
    """
    logging.info("Compressing to {}".format(path))
    # TODO use a temporary file here and mv once finished. Otherwise can't
    # be sure that the file is correctly written.
    try:
        with zarr.ZipStore(path, mode='w') as store:
            root = zarr.group(store=store)
            compress_zarr(ts, root, compressor=compressor, variants_only=variants_only)
    except Exception as e:
        os.unlink(path)
        raise e


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


def compress_zarr(ts, root, compressor=None, variants_only=False):
    tables = ts.dump_tables()

    coordinates = np.unique(np.hstack([
        [0, ts.sequence_length], tables.edges.left, tables.edges.right,
        tables.sites.position, tables.migrations.left, tables.migrations.right]))

    if variants_only:
        logging.info("Using lossy variants-only compression")
        # Reduce to site topology and quantise node times.
        tables.simplify(reduce_to_site_topology=True)
        time = np.unique(tables.nodes.time)
        node_time = np.searchsorted(time, tables.nodes.time)
    else:
        node_time = tables.nodes.time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # When using a zipfile in Zarr we get some harmless warnings. See
        # https://zarr.readthedocs.io/en/stable/api/storage.html#zarr.storage.ZipStore
        root.attrs["format_name"] = FORMAT_NAME
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["sequence_length"] = tables.sequence_length

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
    if compressor is None:
        compressor = numcodecs.Blosc(
            cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
    for column in columns:
        column.compress(root, compressor)

def check_format(root):
    try:
        format_name = root.attrs["format_name"]
        format_version = root.attrs["format_version"]
    except KeyError:
        raise exceptions.FileFormatError("Incorrect file format")
    if format_name != FORMAT_NAME:
        raise exceptions.FileFormatError(
            "Incorrect file format: expected '{}' got '{}'".format(
                self.FORMAT_NAME, format_name))
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
    store = zarr.ZipStore(path, mode='r')
    root = zarr.group(store=store)
    try:
        check_format(root)
        yield root
    finally:
        store.close()


def decompress(path):
    """
    Returns a decompressed tskit tree sequence read from the specified path.
    """
    with load_zarr(path) as root:
        tables = tskit.TableCollection(root.attrs["sequence_length"])
        coordinates = root["coordinates"][:]

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


def print_summary(path):

    def visitor(array):
        if isinstance(array, zarr.core.Array):
            # ratio = 0
            # if array.nbytes > 0:
            #     ratio = array.nbytes_stored / array.nbytes
            print(
                array.nbytes_stored, humanize.naturalsize(array.nbytes_stored),
                array.name, sep="\t")
            # print(array.info)

    with load_zarr(path) as root:
        root.visitvalues(visitor)

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

import numcodecs
import zarr
import numpy as np
import tskit

logger = logging.getLogger(__name__)


def compress(ts, path):
    """
    Compresses the specified tree sequence and writes it to the specified
    path.
    """
    logging.info("Compressing to {}".format(path))
    try:
        store = zarr.ZipStore(path, mode='w')
        root = zarr.group(store=store)
        compress_zarr(ts, root)
        store.close()
    except Exception as e:
        os.unlink(path)
        raise e


def compress_zarr(ts, root):
    tables = ts.dump_tables()
    root.attrs["sequence_length"] = tables.sequence_length

    compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)

    arrays = {
        "individuals/flags": tables.individuals.flags,
        "individuals/location": tables.individuals.location,
        "individuals/location_offset": tables.individuals.location_offset,
        "individuals/metadata": tables.individuals.metadata,
        "individuals/metadata_offset": tables.individuals.metadata_offset,

        "nodes/flags": tables.nodes.flags,
        "nodes/time": tables.nodes.time,
        "nodes/population": tables.nodes.population,
        "nodes/individual": tables.nodes.individual,
        "nodes/metadata": tables.nodes.metadata,
        "nodes/metadata_offset": tables.nodes.metadata_offset,

        "edges/left": tables.edges.left,
        "edges/right": tables.edges.right,
        "edges/parent": tables.edges.parent,
        "edges/child": tables.edges.child,

        "migrations/left": tables.migrations.left,
        "migrations/right": tables.migrations.right,
        "migrations/node": tables.migrations.node,
        "migrations/source": tables.migrations.source,
        "migrations/dest": tables.migrations.dest,
        "migrations/time": tables.migrations.time,

        "sites/position": tables.sites.position,
        "sites/ancestral_state": tables.sites.ancestral_state,
        "sites/ancestral_state_offset": tables.sites.ancestral_state_offset,
        "sites/metadata": tables.sites.metadata,
        "sites/metadata_offset": tables.sites.metadata_offset,

        "mutations/site": tables.mutations.site,
        "mutations/node": tables.mutations.node,
        "mutations/parent": tables.mutations.parent,
        "mutations/derived_state": tables.mutations.derived_state,
        "mutations/derived_state_offset": tables.mutations.derived_state_offset,
        "mutations/metadata": tables.mutations.metadata,
        "mutations/metadata_offset": tables.mutations.metadata_offset,

        "populations/metadata": tables.populations.metadata,
        "populations/metadata_offset": tables.populations.metadata_offset,

        "provenances/timestamp": tables.provenances.timestamp,
        "provenances/timestamp_offset": tables.provenances.timestamp_offset,
        "provenances/record": tables.provenances.record,
        "provenances/record_offset": tables.provenances.record_offset,
    }
    filters = None

    for column_name, array in arrays.items():
        compressed_col = root.empty(
            column_name, shape=array.shape, dtype=array.dtype, filters=filters,
            compressor=compressor)
        compressed_col[:] = array
        # print(compressed_col.info)


def decompress(path):
    """
    Returns a decompressed tskit tree sequence read from the specified path.
    """
    store = zarr.ZipStore(path, mode='r')
    root = zarr.group(store=store)
    return decompress_zarr(root)


def decompress_zarr(root):
    tables = tskit.TableCollection(root.attrs["sequence_length"])

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
        left=root["edges/left"],
        right=root["edges/right"],
        parent=root["edges/parent"],
        child=root["edges/child"])

    tables.migrations.set_columns(
        left=root["migrations/left"],
        right=root["migrations/right"],
        node=root["migrations/node"],
        source=root["migrations/source"],
        dest=root["migrations/dest"],
        time=root["migrations/time"])

    tables.sites.set_columns(
        position=root["sites/position"],
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

    # print(tables)
    return tables.tree_sequence()


def compress_zarr_agressive(ts, root):
    # TODO this current version is the most extreme option where we throw away
    # all the non-site information.

    # First reduce to site topology
    tables = ts.dump_tables()
    tables.simplify(reduce_to_site_topology=True)

    nodes = root.create_group("nodes")
    flags = nodes.empty("flags", shape=len(tables.nodes), dtype=np.uint8)
    flags[:] = tables.nodes.flags
    logger.debug(flags.info)

    # Get the indexes into the position array.
    pos_map = np.hstack([tables.sites.position, [tables.sequence_length]])
    pos_map[0] = 0
    left_mapped = np.searchsorted(pos_map, tables.edges.left)
    if np.any(pos_map[left_mapped] != tables.edges.left):
        raise ValueError("Invalid left coordinates")
    right_mapped = np.searchsorted(pos_map, tables.edges.right)
    if np.any(pos_map[right_mapped] != tables.edges.right):
        raise ValueError("Invalid right coordinates")

    filters = [numcodecs.Delta(dtype=np.int32, astype=np.int32)]
    compressor = numcodecs.Blosc(cname='zstd', clevel=9, shuffle=numcodecs.Blosc.SHUFFLE)
    edges = root.create_group("edges")
    parent = edges.empty(
        "parent", shape=len(tables.edges), dtype=np.int32, filters=filters,
        compressor=compressor)
    child = edges.empty(
        "child", shape=len(tables.edges), dtype=np.int32, filters=filters,
        compressor=compressor)
    left = edges.empty(
        "left", shape=len(tables.edges), dtype=np.uint32, filters=filters,
        compressor=compressor)
    right = edges.empty(
        "right", shape=len(tables.edges), dtype=np.uint32, filters=filters,
        compressor=compressor)
    parent[:] = tables.edges.parent
    child[:] = tables.edges.child
    left[:] = left_mapped
    right[:] = right_mapped

    mutations = root.create_group("mutations")
    site = mutations.empty(
        "site", shape=len(tables.mutations), dtype=np.int32, compressor=compressor)
    node = mutations.empty(
        "node", shape=len(tables.mutations), dtype=np.int32, compressor=compressor)
    site[:] = tables.mutations.site
    node[:] = tables.mutations.node


def decompress_zarr_aggressive(root):
    site = root["mutations/site"][:]
    num_sites = site[-1] + 1
    n = site.shape[0]
    tables = tskit.TableCollection(num_sites)
    tables.mutations.set_columns(
        node=root["mutations/node"],
        site=site,
        derived_state=np.zeros(n, dtype=np.int8) + ord("1"),
        derived_state_offset=np.arange(n + 1, dtype=np.uint32))
    tables.sites.set_columns(
        position=np.arange(num_sites),
        ancestral_state=np.zeros(num_sites, dtype=np.int8) + ord("0"),
        ancestral_state_offset=np.arange(num_sites + 1, dtype=np.uint32))
    flags = root["nodes/flags"][:]
    n = flags.shape[0]
    tables.nodes.set_columns(
        flags=flags.astype(np.uint32),
        time=np.arange(n))
    tables.edges.set_columns(
        left=root["edges/left"],
        right=root["edges/right"],
        parent=root["edges/parent"],
        child=root["edges/child"])
    return tables.tree_sequence()

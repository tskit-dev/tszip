#!/usr/bin/env python3
"""
Script to test zarr cross-version compatibility.
Usage: python test_zarr_cross_version.py [write|read] <filename>
"""
import pathlib
import sys

import msprime
import tskit

# Add parent directory to path so we can import tszip
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import tszip  # noqa: E402


def all_fields_ts(edge_metadata=True, migrations=True):
    """
    A tree sequence with data in all fields (except edge metadata is not set if
    edge_metadata is False and migrations are not defined if migrations is False
    (this is needed to test simplify, which doesn't allow either)

    """
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=5_000)
    demography.add_population(name="C", initial_size=1_000)
    demography.add_population(name="D", initial_size=500)
    demography.add_population(name="E", initial_size=100)
    demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(
        samples={"A": 10, "B": 10},
        demography=demography,
        sequence_length=5,
        random_seed=42,
        recombination_rate=1,
        record_migrations=migrations,
        record_provenance=True,
    )
    ts = msprime.sim_mutations(ts, rate=0.001, random_seed=42)
    tables = ts.dump_tables()
    # Add locations to individuals
    individuals_copy = tables.individuals.copy()
    tables.individuals.clear()
    for i, individual in enumerate(individuals_copy):
        tables.individuals.append(
            individual.replace(flags=i, location=[i, i + 1], parents=[i - 1, i - 1])
        )
    # Ensure all columns have unique values
    nodes_copy = tables.nodes.copy()
    tables.nodes.clear()
    for i, node in enumerate(nodes_copy):
        tables.nodes.append(
            node.replace(
                flags=i,
                time=node.time + 0.00001 * i,
                individual=i % len(tables.individuals),
                population=i % len(tables.populations),
            )
        )
    if migrations:
        tables.migrations.add_row(left=0, right=1, node=21, source=1, dest=3, time=1001)

    # Add metadata
    for name, table in tables.table_name_map.items():
        if name == "provenances":
            continue
        if name == "migrations" and not migrations:
            continue
        if name == "edges" and not edge_metadata:
            continue
        table.metadata_schema = tskit.MetadataSchema.permissive_json()
        metadatas = [f'{{"foo":"n_{name}_{u}"}}' for u in range(len(table))]
        metadata, metadata_offset = tskit.pack_strings(metadatas)
        table.set_columns(
            **{
                **table.asdict(),
                "metadata": metadata,
                "metadata_offset": metadata_offset,
            }
        )
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = "Test metadata"
    tables.time_units = "Test time units"

    tables.reference_sequence.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.reference_sequence.metadata = "Test reference metadata"
    tables.reference_sequence.data = "A" * int(ts.sequence_length)
    tables.reference_sequence.url = "http://example.com/a_reference"

    # Add some more rows to provenance to have enough for testing.
    for i in range(3):
        tables.provenances.add_row(record="A", timestamp=str(i))

    return tables.tree_sequence()


def write_test_file(filename):
    """Write a test file with current zarr version"""
    ts = all_fields_ts()
    tszip.compress(ts, filename)
    ts2 = tszip.decompress(filename)
    ts.tables.assert_equals(ts2.tables)


def read_test_file(filename):
    """Read and verify a test file with current zarr version"""
    try:
        tszip.decompress(filename)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    action = sys.argv[1]
    filename = sys.argv[2]
    if action == "write":
        write_test_file(filename)
    elif action == "read":
        read_test_file(filename)

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
Tests for the basic compression funcionality.
"""
import unittest
import tempfile
import pathlib

import msprime
import numpy as np
import zarr

import tszip
import tszip.compression as compression
import tszip.exceptions as exceptions


class TestMinimalDtype(unittest.TestCase):
    """
    Test that we compute the minimal dtype for data correctly.
    """
    def verify(self, dtype, values):
        for v in values:
            min_dtype = compression.minimal_dtype(np.array([v]))
            self.assertEqual(min_dtype, np.dtype(dtype))

    def test_int8(self):
        self.verify(np.int8, np.array([0, -1, 1, 127, -127], dtype=np.int64))

    def test_uint8(self):
        self.verify(np.uint8, np.array([0, 1, 127, 255], dtype=np.uint64))

    def test_int16(self):
        self.verify(np.int16, np.array(
            [2**15 - 1, -2**15 + 1, 2**7 + 1, -2**7 - 1], dtype=np.int64))

    def test_uint16(self):
        self.verify(np.uint16, np.array([256, 2**16 - 1], dtype=np.uint64))

    def test_int32(self):
        self.verify(np.int32, np.array(
            [2**31 - 1, -2**31 + 1, 2**15 + 1, -2**15 - 1], dtype=np.int64))

    def test_uint32(self):
        self.verify(np.uint32, np.array([2**16 + 1, 2**32 - 1], dtype=np.uint64))

    def test_int64(self):
        self.verify(np.int64, np.array(
            [2**63 - 1, -2**63 + 1, 2**31 + 1, -2**31 - 1], dtype=np.int64))

    def test_uint64(self):
        self.verify(np.uint64, np.array([2**32 + 1, 2**64 - 1], dtype=np.uint64))

    def test_float32(self):
        self.verify(np.float32, np.array([0.1, 1e-3], dtype=np.float32))

    def test_float64(self):
        self.verify(np.float64, np.array([0.1, 1e-3], dtype=np.float64))

    def test_empty(self):
        for dtype in map(np.dtype, [np.float64, np.int32, np.uint8]):
            min_dtype = compression.minimal_dtype(np.array([], dtype=dtype))
            self.assertEqual(min_dtype, dtype)


class RoundTripMixin(object):
    """
    Set of example tree sequences that we should be able to round trip.
    """
    def test_small_msprime_no_recomb(self):
        ts = msprime.simulate(10, mutation_rate=2, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        self.verify(ts)

    def test_small_msprime_recomb(self):
        ts = msprime.simulate(10, recombination_rate=2, mutation_rate=2, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)

    def test_small_msprime_migration(self):
        ts = msprime.simulate(
            population_configurations=[
                msprime.PopulationConfiguration(10),
                msprime.PopulationConfiguration(10)],
            migration_matrix=[[0, 1], [1, 0]],
            record_migrations=True,
            recombination_rate=2, mutation_rate=2, random_seed=2)
        self.assertGreater(ts.num_sites, 2)
        self.assertGreater(ts.num_migrations, 1)
        self.assertGreater(ts.num_trees, 2)
        self.verify(ts)


class TestGenotypeRoundTrip(unittest.TestCase, RoundTripMixin):
    """
    Tests that we can correctly roundtrip genotype data losslessly.
    """
    def verify(self, ts):
        if ts.num_migrations > 0:
            raise unittest.SkipTest("Migrations not supported")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "treeseq.tsz"
            tszip.compress(ts, path, variants_only=True)
            other_ts = tszip.decompress(path)
        self.assertEqual(ts.num_sites, other_ts.num_sites)
        for var1, var2 in zip(ts.variants(), other_ts.variants()):
            self.assertTrue(np.array_equal(var1.genotypes, var2.genotypes))
            self.assertEqual(var1.site.position, var2.site.position)
            self.assertEqual(var1.alleles, var2.alleles)


class TestExactRoundTrip(unittest.TestCase, RoundTripMixin):
    def verify(self, ts):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "treeseq.tsz"
            tszip.compress(ts, path)
            other_ts = tszip.decompress(path)
        self.assertEqual(ts.tables, other_ts.tables)


class TestFormat(unittest.TestCase):
    """
    Tests that we correctly write the format information to the file and
    that we read it also.
    """
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmpdir.name) / "treeseq.tsz"

    def tearDown(self):
        del self.tmpdir

    def test_format_written(self):
        ts = msprime.simulate(10, random_seed=1)
        tszip.compress(ts, self.path)
        store = zarr.ZipStore(self.path, mode='r')
        root = zarr.group(store=store)
        self.assertEqual(root.attrs["format_name"], compression.FORMAT_NAME)
        self.assertEqual(root.attrs["format_version"], compression.FORMAT_VERSION)

    def write_file(self, attrs, path):
        with zarr.ZipStore(path, mode='w') as store:
            root = zarr.group(store=store)
            root.attrs.update(attrs)

    def test_missing_format_keys(self):
        values = [{}, {"format_name": ""}, {"format_version": ""}]
        for attrs in values:
            self.write_file(attrs, self.path)
            with self.assertRaises(exceptions.FileFormatError):
                tszip.decompress(self.path)
            with self.assertRaises(exceptions.FileFormatError):
                tszip.print_summary(self.path)

    def test_bad_format_name(self):
        for bad_name in ["", "xyz", [1234]]:
            self.write_file({"format_name": bad_name, "format_version": [1, 0]}, self.path)
            with self.assertRaises(exceptions.FileFormatError):
                tszip.decompress(self.path)
            with self.assertRaises(exceptions.FileFormatError):
                tszip.print_summary(self.path)

    def test_format_too_old(self):
        self.write_file({"format_name": "tszip", "format_version": [0, 0]}, self.path)
        with self.assertRaises(exceptions.FileFormatError):
            tszip.decompress(self.path)
        with self.assertRaises(exceptions.FileFormatError):
            tszip.print_summary(self.path)

    def test_format_too_new(self):
        self.write_file({"format_name": "tszip", "format_version": [2, 0]}, self.path)
        with self.assertRaises(exceptions.FileFormatError):
            tszip.decompress(self.path)
        with self.assertRaises(exceptions.FileFormatError):
            tszip.print_summary(self.path)

    def test_wrong_format(self):
        for contents in ["", "1234", "X" * 1024]:
            with open(self.path, "w") as f:
                f.write(contents)
            with self.assertRaises(exceptions.FileFormatError):
                tszip.decompress(self.path)


class TestFileErrors(unittest.TestCase):
    """
    Tests that we correctly write the format information to the file and
    that we read it also.
    """
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmpdir.name) / "treeseq.tsz"

    def tearDown(self):
        del self.tmpdir

    def test_missing_file(self):
        path = "/no/such/file"
        with self.assertRaises(FileNotFoundError):
            tszip.decompress(path)

    def test_load_dir(self):
        with self.assertRaises(IsADirectoryError):
            tszip.decompress(self.path.parent)

    def test_save_dir(self):
        ts = msprime.simulate(10, random_seed=1)
        with self.assertRaises(IsADirectoryError):
            tszip.compress(ts, self.path.parent)

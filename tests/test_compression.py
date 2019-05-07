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

import tszip


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
        # G1 = ts.genotype_matrix()
        # G2 = other_ts.genotype_matrix()
        # self.assertTrue(np.array_equal(G1, G2))
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

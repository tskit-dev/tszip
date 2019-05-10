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
Tests for the basic provenance recording.
"""
import unittest

import tskit
import zarr
import numcodecs

import tszip
from tszip import provenance


class TestProvenance(unittest.TestCase):
    """
    Test basic provenance properties.
    """
    def test_schema_validates(self):
        for params in [{}, {"a": "a"}, {"a": {"a": 1}}]:
            d = provenance.get_provenance_dict(params)
            tskit.validate_provenance(d)

    def test_environment(self):
        # Basic environment should be the same as tskit.
        d_tszip = provenance.get_provenance_dict({})
        d_tskit = tskit.provenance.get_provenance_dict()
        self.assertEqual(d_tszip["environment"]["os"], d_tskit["environment"]["os"])
        self.assertEqual(
            d_tszip["environment"]["python"], d_tskit["environment"]["python"])

    def test_libraries(self):
        libs = provenance.get_provenance_dict({})["environment"]["libraries"]
        self.assertEqual(libs["tskit"]["version"], tskit.__version__)
        self.assertEqual(libs["zarr"]["version"], zarr.__version__)
        self.assertEqual(libs["numcodecs"]["version"], numcodecs.__version__)

    def test_sofware(self):
        software = provenance.get_provenance_dict({})["software"]
        self.assertEqual(software, {"name": "tszip", "version": tszip.__version__})

    def test_parameters(self):
        for params in [{}, {"a": "a"}, {"a": {"a": 1}}]:
            d = provenance.get_provenance_dict(params)["parameters"]
            self.assertEqual(d, params)

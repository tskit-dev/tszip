# MIT License
#
# Copyright (c) 2025 Tskit Developers
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
Compatibility layer for zarr v2/v3 API differences
"""
import zarr

ZARR_V3 = zarr.__version__.startswith("3.")


if ZARR_V3:
    from zarr.storage import ZipStore

    def create_zip_store(path, mode="r"):
        return ZipStore(path, mode=mode)

    def create_zarr_group(store=None):
        if store is None:
            return zarr.create_group(zarr_format=2)
        else:
            mode = "r" if getattr(store, "read_only", False) else "a"
            return zarr.open_group(store=store, zarr_format=2, mode=mode)

    def create_empty_array(
        group, name, shape, dtype, chunks=None, filters=None, compressor=None
    ):
        return group.empty(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            zarr_format=2,
            filters=filters,
            compressor=compressor,
        )

    def get_nbytes_stored(array):
        return array.nbytes_stored()

    def group_items(group):
        return group.members()

    def visit_arrays(group, visitor):
        for array in group.array_values():
            visitor(array)

else:

    def create_zip_store(path, mode="r"):
        return zarr.ZipStore(path, mode=mode)

    def create_zarr_group(store=None):
        return zarr.group(store=store)

    def create_empty_array(
        group, name, shape, dtype, chunks=None, filters=None, compressor=None
    ):
        return group.empty(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            filters=filters,
            compressor=compressor,
        )

    def get_nbytes_stored(array):
        return array.nbytes_stored

    def group_items(group):
        return group.items()

    def visit_arrays(group, visitor):
        group.visitvalues(visitor)

import zarr

_ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3


def open_zip_store(path, mode):
    """Open a ZipStore compatible with zarr v2 and v3."""
    return zarr.storage.ZipStore(str(path), mode=mode)


def open_group_for_read(store):
    """Open a zarr group for reading in zarr v2 format."""
    if _ZARR_V3:
        return zarr.open_group(store=store, zarr_format=2, mode="r")
    else:
        return zarr.open_group(store=store, mode="r")


def open_group_for_write(store):
    """Open a zarr group for writing in zarr v2 format."""
    if _ZARR_V3:
        return zarr.open_group(store=store, zarr_format=2, mode="a")
    else:
        return zarr.open_group(store=store, mode="a")


def empty_array(root, name, shape, dtype, chunks, filters, compressor):
    """Create an empty zarr array in zarr v2 format."""
    if _ZARR_V3:
        return root.empty(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            zarr_format=2,
            filters=filters,
            compressor=compressor,
        )
    else:
        return root.empty(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            filters=filters,
            compressor=compressor,
        )

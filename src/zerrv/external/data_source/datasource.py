from abc import ABC, abstractmethod
from typing import List, Iterator, Tuple, Union

import pathlib

import h5py
import z5py
from PIL import Image as PilImage
import numpy as np

import enum
from enum import IntEnum

from zerrv.external.array5d import Array5D, Point5D, Shape5D, Slice5D, Chunk5D, LazyArray5D
from zerrv.util import handle_path, h5n5_file

import logging

logger = logging.getLogger(__name__)


@enum.unique
class AddressMode(IntEnum):
    BLACK = 0
    MIRROR = enum.auto()
    WRAP = enum.auto()


class DataSource:
    """An entity able to retrieve raw Array5D's, usually from disk or network"""

    def __init__(self, url: str):
        self.url = url

    def __hash__(self):  # TODO: maybe include shape/tile_shape?
        return hash(self.url)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.url == other.url and self.tile_shape == other.tile_shape

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.shape} from {self.url}>"

    def cut_with(
        self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)
    ) -> "DataSourceSlice":
        return self.cut(Slice5D(t=t, c=c, x=x, y=y, z=z))

    def cut(self, slc: Slice5D) -> "DataSourceSlice":
        return DataSourceSlice.from_slice(self, slc.defined_with(self.shape))

    def all(self) -> "DataSourceSlice":
        return self.cut(self.shape.to_slice_5d())

    @property
    def tile_shape(self):
        """A sensible tile shape. Override this with your data chunk size"""
        return Shape5D(x=2048, y=2048, c=self.shape.c)

    @property
    @abstractmethod
    def shape(self) -> Shape5D:
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    def contains(self, slc: Slice5D) -> bool:
        return self.shape.to_slice_5d().contains(slc.defined_with(self.shape))

    def retrieve(
        self, roi: Slice5D, halo: Point5D = Point5D.zero(), address_mode: AddressMode = AddressMode.BLACK
    ) -> Array5D:
        # FIXME: Remove address_mode or implement all variations and make feature extractors
        # use te correct one
        roi = roi.defined_with(self.shape)
        assert self.shape.to_slice_5d().contains(roi)
        haloed_roi = roi.enlarged(halo)
        out = Array5D.allocate(haloed_roi.shape, dtype=self.dtype, value=0)

        data_roi = haloed_roi.clamped_with_slice(self.shape.to_slice_5d())
        data = self.do_retrieve(data_roi)

        offset = data_roi.start - haloed_roi.start
        out.set_slice(data, slc=data.shape.to_slice_5d().offset(offset))
        return out  # TODO: make slice read-only

    @abstractmethod
    def do_retrieve(self, roi: Slice5D) -> Array5D:
        pass

    def __getstate__(self):
        return {"url": self.url, "shape": self.shape}

    def __setstate__(self, data):
        self.__init__(data["url"])

    @property
    def json_data(self):
        return self.__getstate__()


class DataSourceSlice(Slice5D):
    def __init__(
        self, data_source: DataSource, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)
    ):
        super().__init__(t=t, c=c, x=x, y=y, z=z)
        self.data_source = data_source

    def rebuild(self, *, t=slice(None), c=slice(None), x=slice(None), y=slice(None), z=slice(None)):
        return self.__class__(self.data_source, t=t, c=c, x=x, y=y, z=z)

    def __hash__(self):
        return hash((super().__hash__(), self.data_source))

    def __eq__(self, other):
        if not isinstance(other, DataSourceSlice):
            return False
        return super().__eq__(other) and other.data_source == self.data_source

    @classmethod
    def from_slice(cls, data_source: DataSource, slc: Slice5D):
        return cls(data_source, **slc.to_dict())

    def retrieve(self, halo: Point5D = Point5D.zero()):
        return self.data_source.retrieve(self, halo)

    def get_tiles(self, tile_shape: Shape5D = None):
        for tile in super().get_tiles(tile_shape or self.data_source.tile_shape):
            yield tile.clamped_with_slice(self.data_source.shape.to_slice_5d())

    def mod_tile(self, tile_shape: Shape5D = None) -> "DataSourceSlice":
        return super().mod_tile(tile_shape or self.data_source.tile_shape)


class H5N5DataSource(DataSource):
    def __init__(self, url: str):
        super().__init__(url)
        external_path, internal_path, is_h5n5 = handle_path(pathlib.Path(url))
        assert is_h5n5, "This class only handles n5/h5 files"
        self.external_path = external_path
        self.internal_path = internal_path
        self._filep = h5n5_file(external_path)
        self._dataset = self._filep[internal_path]
        assert isinstance(self._dataset, (z5py.dataset.Dataset, h5py.Dataset))
        axiskeys = self.determine_axistags(self._dataset.shape)
        self._data = LazyArray5D(self._dataset, axiskeys=axiskeys[::-1])

    @staticmethod
    def determine_axistags(shape: List[int]) -> str:
        """
        # HACK: for now just assume some axistags for a given shape...
        """
        axisorders = {2: "yx", 3: "zyx", 4: "czyx", 5: "tczyx"}
        ndim = len(shape)

        if ndim in [0, 1]:
            raise ValueError(f"Got 'ndim' == {ndim}. {ndim}-D data not yet supported")
        elif ndim > 5:
            raise ValueError(f"Got 'ndim' == {ndim} dim. No Support for data with more than 5 " "dimensions")

        axisorder = axisorders[ndim]

        if ndim == 3 and shape[0] <= 4:
            # Special case: If the 3rd dim is small, assume it's 'c', not 'z'
            axisorder = "cyx"

        return axisorder

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def do_retrieve(self, roi: Slice5D):
        return self._data.cut(roi)

    def tile_shape(self) -> Chunk5D:
        chunks = self._dataset.chunks
        original_axiskeys = self._data._original_axiskeys
        # assert len(chunks) == len(original_axiskeys)
        logger.debug(f"chunks: {chunks}")
        logger.debug(f"original_axiskeys: {original_axiskeys}")

        return Chunk5D(**{k: v for k, v in zip(original_axiskeys, chunks)})

    @property
    def axiskeys(self):
        return self._data._axiskeys


class FlatDataSource(DataSource):
    """A naive implementation o DataSource that can read images using PIL"""

    def __init__(self, url: str):
        super().__init__(url)
        raw_data = np.asarray(PilImage.open(url))
        axiskeys = "yxc"[: len(raw_data.shape)]
        self._data = Array5D(raw_data, axiskeys=axiskeys)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def do_retrieve(self, roi: Slice5D):
        return self._data.cut(roi)

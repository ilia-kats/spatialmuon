import atexit
from typing import Optional

from rtree import index
import h5py
import numpy as np

from .backing import BackableObject
from ..utils import _read_hdf5_attribute


class HDF5Storage(index.CustomStorage):
    def __init__(self, grp: Optional[h5py.Group] = None):
        super().__init__()

        self._pages = []
        self._emptypages = []
        self._grp = None
        self._readonly = False
        self.backing = grp
        self._pageid = len(self._grp) if self.isbacked else 0

    @property
    def isbacked(self):
        return self._grp is not None

    @property
    def backing(self) -> Optional[h5py.Group]:
        return self._grp

    @backing.setter
    def backing(self, grp: Optional[h5py.Group]):
        if grp is not None:
            if self.hasData:
                self.to_hdf5(grp)
            self._pages = []
            self._emptypages = []
            self.clear = self.__clear_backed
            self.loadByteArray = self.__loadByteArray_backed
            self.storeByteArray = self.__storeByteArray_backed
            self.deleteByteArray = self.__deleteByteArray_backed
            self._readonly = grp.file.mode == "r"
        else:
            if self.isbacked:
                self.from_hdf5(self._grp)
            self.clear = self.__clear
            self.loadByteArray = self.__loadByteArray
            self.storeByteArray = self.__storeByteArray
            self.deleteByteArray = self.__deleteByteArray
            self._readonly = False
        self._grp = grp

    @property
    def hasData(self):
        return len(self._grp) > 0 if self.isbacked else len(self._pages) > 0

    def create(self, returnError):
        """Called when the storage is created on the C side"""
        pass

    def destroy(self, returnError):
        """Called when the storage is destroyed on the C side"""
        pass

    def __clear_backed(self):
        """Clear all our data"""
        del self._parent[self._name]
        self._grp = self._parent.create_group(self._name)

    def __clear(self):
        self._pages = []
        self._emptypages = []

    def __loadByteArray_backed(self, page, returnError):
        """Returns the data for page or returns an error"""
        try:
            return self._grp[str(page)][()].tobytes()
        except KeyError:
            returnError.contents.value = self.InvalidPageError

    def __loadByteArray(self, page, returnError):
        try:
            return self._pages[page]
        except IndexError:
            returnError.contents.value = self.InvalidPageError

    def __storeByteArray_backed(self, page, data, returnError):
        """Stores the data for page"""
        # this is called when deleting the index object due to a flush() in libspatialindex:~Buffer()
        # when having a read-only file, a bunch of stack traces are printed. Avoid that
        if self._readonly:
            return page
        data = np.frombuffer(data, dtype=np.uint8)
        if page == self.NewPage:
            self._pageid += 1
            self._grp[str(self._pageid)] = data
            page = self._pageid
            spage = str(page)
        else:
            spage = str(page)
            if spage not in self._grp:
                returnError.value = self.InvalidPageError
                return 0
            del self._grp[spage]
        dset = self._grp.create_dataset(
            spage,
            (len(data),),
            dtype=np.uint8,
            chunks=(len(data),),
            compression="gzip",
            compression_opts=9,
        )
        dset[()] = data
        return page

    def __storeByteArray(self, page, data, returnError):
        if page == self.NewPage:
            if len(self._emptypages) > 0:
                page = self._emptypages[-1]
                del self._emptypages[-1]
            else:
                self._pages.append(None)
                page = len(self._pages) - 1
        try:
            self._pages[page] = data
            return page
        except IndexError:
            returnError.contents.value = self.InvalidPageError
            return 0

    def __deleteByteArray_backed(self, page, returnError):
        """Deletes a page"""
        if not self._readonly:
            try:
                del self._grp[str(page)]
            except KeyError:
                returnError.contents.value = self.InvalidPageError

    def __deleteByteArray(self, page, returnError):
        try:
            self._pages[page] = None
            self._emptypages.append(page)
        except IndexError:
            returnError.contents.value = self.InvalidPageError

    def flush(self, returnError):
        pass

    def to_hdf5(self, grp):
        for i, page in enumerate(self._pages):
            if page is not None:
                dset = grp.create_dataset(
                    str(i),
                    (len(page),),
                    dtype=np.uint8,
                    chunks=(len(page),),
                    compression="gzip",
                    compression_opts=9,
                )
                dset[()] = np.frombuffer(page, dtype=np.uint8)

    def from_hdf5(self, grp):
        for pageidx, page in grp.items():
            # HDF5 files written with HDF5Storage are not necessarily ordered
            idx = int(pageidx)
            if idx >= len(self._pages):
                for _ in range(len(self._pages), idx + 1):
                    self._pages.append(None)
            self._pages[idx] = page[()].tobytes()


class SpatialIndex(BackableObject):
    def __init__(
        self,
        backing: Optional[h5py.Group] = None,
        coordinates: Optional[np.ndarray] = None,
        dimension: Optional[int] = 2,
        **kwargs,
    ):
        super().__init__(backing)

        if dimension is None and coordinates is None:
            raise RuntimeError("must provide either coordinates or dimension")
        elif coordinates is not None:
            dimension = coordinates.shape[1]
        self._prop = index.Property(
            type=index.RT_RTree, variant=index.RT_Star, dimension=dimension
        )
        self._storage = HDF5Storage(self.backing)
        self._index = index.Index(self._storage, interleaved=True, properties=self._prop)

        # This is needed to prevent a bunch of exceptions occurring at program exit. The reason is that Index
        # flushes its internal buffer when it's destroyed, leading to a bunch of calls to storage.storeByteArray.
        # The backed implementation of storeByteArray performes a check if the page is already saved (with `page in grp`).
        # The corresponding h5py code has an import inside the checking function, which raises the exception because
        # the import machinery is already not functional at this point.
        atexit.register(self._cleanup)

        if coordinates is not None:
            self.set_coordinates(coordinates, **kwargs)

    def _cleanup(self):
        del self._index

    def __del__(self):
        atexit.unregister(self._cleanup)

    @staticmethod
    def _encoding() -> str:
        return "rtree-index"

    @staticmethod
    def _encodingversion() -> str:
        return "0.1.0"

    def _writeable_object(self, parent, key):
        if key is None:
            return parent
        else:
            if key in parent:
                del parent[key]
            return parent.create_group(key, track_order=True)

    def _write_attributes_impl(self, obj):
        pass

    def _set_backing(self, value):
        super()._set_backing(value)
        if value is not None:
            self._write_attributes(value)
        self._storage.backing = value

    def _write(self, grp):
        self._storage.to_hdf5(grp)

    def set_coordinates(self, coordinates: np.ndarray, progressbar: bool = False, **kwargs):
        if coordinates.shape[1] != self._prop.dimension:
            raise RuntimeError("coordinate dimension is different from index dimension")
        if progressbar:
            from tqdm.auto import tqdm

            coordinates = tqdm(coordinates, **kwargs)
        for i, c in enumerate(coordinates):
            self._index.insert(i, np.hstack((c, c)))

from rtree import index
import h5py
import numpy as np

class HDF5Storage(index.CustomStorage):
    def __init__(self, parentobj: h5py.Group, name: str):
        super().__init__()

        self._parent = parentobj
        self._name = name

        if self._name in self._parent:
            self._grp = self._parent[self._name]
            self._hasData = len(self._grp) > 0
            self._pageid = len(self._grp)
        else:
            self._grp = self._parent.create_group(self._name)
            self._grp.attrs["encoding-type"] = "rtree-index"
            self._grp.attrs["encoding-version"] = "0.1.0"
            self._hasData = False
            self._pageid = 0

    @property
    def hasData(self):
        return self._hasData

    def create(self, returnError):
        """ Called when the storage is created on the C side """
        pass

    def destroy(self, returnError):
        """ Called when the storage is destroyed on the C side """
        pass

    def clear(self):
        """ Clear all our data """
        del self._parent[self._name]
        self._grp = self._parent.create_group(self._name)

    def loadByteArray(self, page, returnError):
        """ Returns the data for page or returns an error """
        try:
            return self._grp[str(page)][()].tobytes()
        except KeyError:
            returnError.contents.value = self.InvalidPageError

    def storeByteArray(self, page, data, returnError):
        """ Stores the data for page """
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
        dset = self._grp.create_dataset(spage, (len(data),), dtype=np.uint8, chunks=(len(data),), compression="gzip", compression_opts=9)
        dset[()] = data
        return page

    def deleteByteArray(self, page, returnError):
        """ Deletes a page """
        try:
            del self._grp[str(page)]
        except KeyError:
            returnError.contents.value = self.InvalidPageError

    def flush(self, returnError):
        pass

class SerializableStorage(index.CustomStorage):
    def __init__(self, grp=None):
        self._pages = []
        self._emptypages = []

        if grp is not None:
            self.from_hdf5(grp)

    @property
    def hasData(self):
        return len(self._pages) > 0

    def create(self, returnError):
        pass

    def destroy(self, returnError):
        pass

    def clear(self):
        self._pages = []
        self._emptypages = []

    def loadByteArray(self, page, returnError):
        try:
            return self._pages[page]
        except IndexError:
            returnError.contents.value = self.InvalidPageError

    def storeByteArray(self, page, data, returnError):
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

    def deleteByteArray(self, page, returnError):
        try:
            self._pages[page] = None
            self._emptypages.append(page)
        except IndexError:
            returnError.contents.value = self.InvalidPageError

    def flush(self, returnError):
        pass

    def to_hdf5(self, parent, name):
        if name in parent:
            del parent[name]
        grp = parent.create_group(name, track_order=True)
        for i, page in enumerate(self._pages):
            if page is not None:
                dset = grp.create_dataset(str(i), (len(page),), dtype=np.uint8, chunks=(len(page),), compression="gzip", compression_opts=9)
                dset[()] = np.frombuffer(page, dtype=np.uint8)
        grp.attrs["encoding-type"] = "rtree-index"
        grp.attrs["encoding-version"] = "0.1.0"

    def from_hdf5(self, parent, name):
        for pageidx, page in parent[name].items():
            # HDF5 files written with HDF5Storage are not necessarily ordered
            idx = int(pageidx)
            if idx >= len(self.pages):
                for _ in range(len(self.pages), idx + 1):
                    self._pages.append(None)
            self._pages[idx] = page[()].tobytes()

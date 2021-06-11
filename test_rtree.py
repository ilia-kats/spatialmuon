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
            return self._pageid
        else:
            spage = str(page)
            if spage not in self._grp:
                returnError.value = self.InvalidPageError
                return 0
            del self._grp[spage]
            self._grp[spage] = data
            return page

    def deleteByteArray(self, page, returnError):
        """ Deletes a page """
        try:
            del self._grp[str(page)]
        except KeyError:
            returnError.contents.value = self.InvalidPageError

f = h5py.File("../test.h5", "a", libver="latest")
storage = HDF5Storage(f, "index")
p = index.Property(type=index.RT_RTree, variant=index.RT_Star, dimension=2, pagesize=4096)

idx = index.Index(storage, interleaved=True, properties=p)

coords = np.random.random(size=(1000000, 2))
for i, c in enumerate(coords):
    idx.insert(i, np.hstack((c,c)))


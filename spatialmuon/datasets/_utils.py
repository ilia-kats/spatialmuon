import os
import urllib.request
import zipfile
import hashlib

from tqdm import tqdm


class TqdmDownload(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update({"unit": "B", "unit_scale": True, "unit_divisor": 1024})
        super().__init__(*args, **kwargs)

    def update_to(self, nblocks=1, blocksize=1, total=-1):
        self.total = total
        self.update(nblocks * blocksize - self.n)


def download(url, outfile, desc):
    with TqdmDownload(desc="downloading " + desc) as t:
        urllib.request.urlretrieve(url, outfile, t.update_to)


def unzip(file, outdir, files=None, rm=True):
    zfile = zipfile.ZipFile(file)
    os.makedirs(outdir, exist_ok=True)
    if files is not None:
        for f in files:
            zfile.extract(f, outdir)
    else:
        zfile.extractall(outdir)
    zfile.close()
    if rm:
        os.unlink(file)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

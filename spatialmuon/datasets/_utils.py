import os
import urllib.request
import zipfile

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

def unzip(file, outdir):
    zfile = zipfile.ZipFile(file)
    os.makedirs(outdir, exist_ok=True)
    zfile.extractall(outdir)
    zfile.close()

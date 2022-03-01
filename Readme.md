# Spatial Muon

`spatialmuon` is a framework to handle spatial data in a technology-agnostic manner.

[Documentation](https://spatialmuon.readthedocs.io/) | [Discord](https://discord.com/invite/MMsgDhnSwQ)


[![Documentation Status](https://readthedocs.org/projects/spatialmuon/badge/?version=latest)](http://spatialmuon.readthedocs.io/?badge=latest)
[![PyPi version](https://img.shields.io/pypi/v/spatialmuon)](https://pypi.org/project/spatialmuon)
[![codecov](https://codecov.io/gh/ilia-kats/spatialmuon/branch/main/graph/badge.svg?token=NE7FEDB388)](https://codecov.io/gh/ilia-kats/spatialmuon)

## Datasets
We provide ready-to-use SpatialMuon objects (`.h5smu` files) for several datasets, and for each for them we include the script that we used to convert the data into the new format.

### Where to find the processed data

You can find the following datasets converted into the `.h5smu` format inside the folder 
`/data/spatialmuon/datasets` of the `pc02` GPU machine. 
For each dataset there is the folder `raw` and the folder `smu`. The former is useful in case you want to rerun the conversion scripts without having to redownload the files (and for large files also without having to extract them), the second contains `.h5smu` files ready to go.

 Main technology/ies | Biology | Origin | Brief description | Script | Folder in the `pc02` machine |
|-------------|-------------|---------|-----------|---------|------|
| Visium | Mouse brain | [Kleshchevnikov et al. (2020)](https://doi.org/10.1101/2020.11.15.378125) | TODO: how many samples, 1? consecutive slides, 5? One H&E stained image for each Visium slide.| [visium_mousebrain.py](./spatialmuon/datasets/visium_mousebrain.py) | `visium_mousebrain` |
| SeqFISH+ | Mouse brain | [Eng et al. (2019)](https://doi.org/10.1038/s41586-019-1049-y) | 2 samples, 7 fields of view for each sample. DAPI stained images. Masks for segmented cells | [seqfishplus.py](./spatialmuon/datasets/seqfishplus.py) | `seqfishplus` |
| Imaging Mass Cytometry | Breast cancer | [Jackson et al. (2020)](https://doi.org/10.5281/zenodo.3518284) | ~300 patients and ~700 slides in total. ~50 proteins. Masks for segmentaed cells. | [imc_download.py](./spatialmuon/datasets/imc_download.py) | `imc` |



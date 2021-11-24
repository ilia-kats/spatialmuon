# Spatial Muon
## Datasets
We provide ready-to-use SpatialMuon objects (`.h5smu` files) for several datasets, and for each for them we include 
the script that we used to convert the data into the new format.

 Main technology/ies | Biology | Origin | Brief description | Script | Folder in the `pc02` machine |
|-------------|-------------|---------|-----------|---------|------|
| Visium | Mouse brain | [Kleshchevnikov et al. (2020)](https://doi.org/10.1101/2020.11.15.378125) | TODO: how many samples, 1? consecutive slides, 5? One H&E stained image for each Visium slide.| [visium_mousebrain.py](./spatialmuon/datasets/visium_mousebrain.py) | `visium_mousebrain` |
| SeqFISH+ | Mouse brain | [Eng et al. (2019)](https://doi.org/10.1038/s41586-019-1049-y) | 2 samples, 7 fields of view for each sample. DAPI stained images. Masks for segmented cells | [seqfishplus.py](./spatialmuon/datasets/seqfishplus.py) | `seqfishplus` |
| Imaging Mass Cytometry | Breast cancer | [Jackson et al. (2020)](https://doi.org/10.5281/zenodo.3518284) | ~300 patients and ~700 slides in total. ~50 proteins. Masks for segmentaed cells. | [imc_download.py](./spatialmuon/datasets/imc_download.py) | `imc` |



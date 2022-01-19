import sys
import os
from pathlib import Path
import matplotlib


def initialize_testing():
    debugging = False
    try:
        __file__
    except NameError as e:
        if str(e) == "name '__file__' is not defined":
            debugging = True
        else:
            raise e
    if sys.gettrace() is not None:
        debugging = True

    if not debugging:
        # Get current file and pre-generate paths and names
        test_data_dir = Path(__file__).parent / Path("data")
        matplotlib.use("Agg")
    else:
        test_data_dir = Path(os.path.expanduser("~/spatialmuon/tests/data/"))
    return test_data_dir.resolve(), debugging

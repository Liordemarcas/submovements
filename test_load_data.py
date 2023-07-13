import numpy as np
from loaddata import loaddata

def test_no_csv_files():
    dirname = "path/to/empty/directory"
    try:
        loaddata(dirname)
        assert False, "Expected ValueError"
    except ValueError:
        print ('Must specify a directory to load the csv files from')

def test_valid_input():
    dirname = "path/to/csv/files/directory"
    try:
        result = loaddata(dirname)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(item, list) for item in result)
        assert all(isinstance(sublist, np.ndarray) for sublist in result)
    except ValueError:
        assert False, "Unexpected ValueError"

def test_data_loading():
    dirname = "path/to/csv/files/directory"
    try:
        position_filtered, velocity, time = loaddata(dirname)
        assert len(position_filtered) > 0
        assert len(velocity) > 0
        assert len(time) > 0
        assert all(isinstance(arr, np.ndarray) for arr in position_filtered)
        assert all(isinstance(arr, np.ndarray) for arr in velocity)
        assert all(isinstance(arr, np.ndarray) for arr in time)
    except ValueError:
        assert False, "Unexpected ValueError"
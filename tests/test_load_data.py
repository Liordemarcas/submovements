import numpy as np
from loaddata import load_data

def test_no_csv_files():
    dirname ='data'
    try:
        load_data(dirname)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Must specify a directory to load the csv files from'

def test_valid_input():
    dirname = "data"
    try:
        result = load_data(dirname)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(item, list) for item in result)
        assert all(isinstance(sublist, np.ndarray) for sublist in result)
    except ValueError:
        assert False, "Unexpected ValueError"

def test_data_loading():
    dirname = "data"
    try:
        position_filtered, velocity, time = load_data(dirname)
        assert len(position_filtered) > 0
        assert len(velocity) > 0
        assert len(time) > 0
        assert all(isinstance(arr, np.ndarray) for arr in position_filtered)
        assert all(isinstance(arr, np.ndarray) for arr in velocity)
        assert all(isinstance(arr, np.ndarray) for arr in time)
    except ValueError:
        assert False, "Unexpected ValueError"
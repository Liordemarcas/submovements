import numpy as np
from decompose2D_file import decompose2D


def test_time_shape():
    time = np.array([0.1, 0.2, 0.3])
    vel = np.array([[1, 2, 3], [4, 5, 6]])
    try:
        decompose2D(time, vel)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'time must be a 1*N vector'

def test_vel_shape():
    time = np.array([0.1])
    vel = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    try:
        decompose2D(time, vel)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'vel must be an 2*N matrix'

def test_vel_time_match():
    time = np.array([0.1, 0.2, 0.3])
    vel = np.array([[1, 2, 3], [4, 5, 6]])
    try:
        decompose2D(time, vel)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'vel must match time'

def test_bounds():
    time = np.array([0.1])
    vel = np.array([[1, 2]])
    try:
        decompose2D(time, vel)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Lower bounds exceed upper bound - infeasible'

def test_valid_input():
    time = np.array([0.1])
    vel = np.array([[1, 2]])
    try:
        result = decompose2D(time, vel)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], np.ndarray)
    except ValueError:
        assert False, "Unexpected ValueError"
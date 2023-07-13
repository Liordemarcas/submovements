import os
import re
import numpy as np
from scipy.signal import filtfilt, butter

def loaddata(dirname):
    """
    Loads data from CSV files in the specified directory.

    Parameters:
        dirname (str): Directory path containing the CSV files.

    Returns:
        position_filtered (list): List of position data arrays after filtering.
        velocity (list): List of velocity data arrays.
        time (list): List of time data arrays.

    Raises:
        ValueError: If the directory does not contain any CSV files.

    """

    # Get a list of files in the directory
    files = os.listdir(dirname)
# Filter only the CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
# Raise an error if no CSV files are found
    if not csv_files:
        raise ValueError('Must specify a directory to load the csv files from')
# Extract block and trial information from file names
    blocks = []
    trials = []
    filenames = []
    for filename in csv_files:
        filenames.append(filename)
        match = re.search(r'tb_.*block(\d*)_trial(\d*).csv', filename) #checking for correct file name
        block = int(match.group(1))
        trial = int(match.group(2))
        blocks.append(block)
        trials.append(trial)
# We have lists of blocks and trials and looks for max to see how much blocks and trials we have in this folder
    max_block = max(blocks)
    max_trial = max(trials)
    position_filtered = []
    velocity = []
    time = []
 
# Process data for each block and trial
    for block in range(1, max_block + 1):
        for trial in range(1, max_trial + 1):
            trial_index = [i for i, (_block, _trial) in enumerate(zip(blocks, trials)) if _block == block and _trial == trial]
            if not trial_index:
                continue
            trial_num = (block - 1) * max_trial + trial
            data = np.loadtxt(os.path.join(dirname, csv_files[trial_index[0]]), delimiter=',')
            pressure = data[:, 3]
            position = data[pressure > 0, :2] / 1000
            _time = data[pressure > 0, 4] / 1000  # seconds
            _time = _time - _time[0]
            dt = np.median(np.diff(_time))
            b, a = butter(2, 5 / ((1 / dt) / 2))
            _position_filtered = filtfilt(b, a, position, axis=0)
            _velocity = np.vstack([[0, 0], np.diff(_position_filtered, axis=0) / dt])
#orgenaizing the data in correct variables for future functions/ 
            time.append(_time)
            position_filtered.append(_position_filtered)
            velocity.append(_velocity)

    return position_filtered, velocity, time
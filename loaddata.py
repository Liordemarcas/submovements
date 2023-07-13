""""
function [positionfiltered,velocity,time] = loaddata(dirname)

d = dir([dirname '/*.csv']);

if isempty(d) 
    error('Must specify a directory to load the csv files from');
end

clear block trial
for k=1:numel(d)
   filename{k} = d(k).name; 
   r = regexp(filename{k},'tb_.*block(\d*)_trial(\d*).csv','tokens');
   block(k) = str2double(r{1}{1}); 
   trial(k) = str2double(r{1}{2});
end

for b=1:max(block)
    for t=1:max(trial)
        trialindex = find(b==block & t==trial);
        if isempty(trialindex)
            continue;
        end
        trialnum = (b-1)*max(trial) + t;
        data = load([d(trialindex).folder '/' d(trialindex).name]);
        pressure = data(:,4);
        % Only take the part where the pressure > 0
        position{trialnum} = data(pressure>0,1:2)./1000;
        % Calculate velocity after filtering
        
        time{trialnum} = data(pressure>0,5) ./ 1000; % seconds
        time{trialnum} = time{trialnum} - time{trialnum}(1);
        dt = median(diff(time{trialnum}));
        [B,A] = butter(2,5/((1/dt)/2));
        positionfiltered{trialnum} = filtfilt(B,A,position{trialnum});
        velocity{trialnum} = [[0 0]; diff(positionfiltered{trialnum})./dt];
    end
end
"""


import os
import re
import numpy as np
from scipy.signal import filtfilt, butter


def loaddata(dirname):
    files = os.listdir(dirname)
    csv_files = [f for f in files if f.endswith('.csv')]

    if not csv_files:
        raise ValueError('Must specify a directory to load the csv files from')

    blocks = []
    trials = []
    filenames = []
    for filename in csv_files:
        filenames.append(filename)
        match = re.search(r'tb_.*block(\d*)_trial(\d*).csv', filename)
        block = int(match.group(1))
        trial = int(match.group(2))
        blocks.append(block)
        trials.append(trial)

    max_block = max(blocks)
    max_trial = max(trials)
# we're all good up to here
    position_filtered = []
    velocity = []
    time = []

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

            time.append(_time)
            position_filtered.append(_position_filtered)
            velocity.append(_velocity)

    return position_filtered, velocity, time
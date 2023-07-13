"""
% PLOTPOSITION - plot the position data
% 
% plotposition(position,time,plottype)
%
% plottype = 1 (default) -> x vs y
% plottype = 2           -> time vs x/y

function plotposition(position,time,plottype)

if nargin<3
    plottype=1;
end

figure;
cols = ceil(sqrt(numel(position)));
rows = ceil(numel(position) / cols);

for k=1:numel(position)
        subplot(rows,cols,k);
        if plottype==1
            plot(position{k}(:,1),position{k}(:,2));
            axis equal
        elseif plottype==2
            plot(time{k},position{k});
            if k==numel(position)
                legend('x','y');
            end
        else
            error('Unknown plot type');
        end
end
"""


import numpy as np
import matplotlib.pyplot as plt

def plot_position(position, time, plot_type=1):
    """This function can plot the time against different position type. 
    plot_Type = 1 is time against x & y position 
    plot_type = 2 is time against tanginal position """
    
    if plot_type not in [1, 2]:
        raise ValueError('Unknown plot type')

    num_positions = len(position)
    cols = int(np.ceil(np.sqrt(num_positions)))
    rows = int(np.ceil(num_positions / cols))

    fig, axs = plt.subplots(rows, cols)

    for k in range(num_positions):
        if isinstance(axs, np.ndarray):
            ax = axs[k // cols, k % cols]

        if plot_type == 1:
            x, y = np.split(position[k],[-1], axis=1)
            ax.plot(x, y)
            ax.set_ylim([0, max(y)])
            ax.set_xlim([0, max(x)])
            if k == num_positions - 1:
                ax.legend(['x', 'y'])
        elif plot_type == 2:
            ax.plot(time[k], position[k])
            if k == num_velocity - 1:
                ax.legend(['x', 'y'])
    plt.show()




"""
% PLOTVELOCITY - plot the velocity data
% 
% plotvelocity(velocity,time,plottype)
%
% plottype = 1 (default) -> time vs v_x/v_y
% plottype = 2           -> time vs tangential velocity

function plotvelocity(velocity,time,plottype)

if nargin<3
    plottype=1;
end

figure;
cols = ceil(sqrt(numel(velocity)));
rows = ceil(numel(velocity) / cols);

for k=1:numel(velocity)
        subplot(rows,cols,k);
        if plottype==1
            plot(time{k},velocity{k});
            if k==numel(velocity)
                legend('v_x','v_y');
            end
        elseif plottype==2
            tangvel = sqrt(sum(velocity{k}.^2,2));
            plot(time{k},tangvel);
        else
            error('Unknown plot type');
        end
end


"""

def plot_velocity(velocity, time, plot_type=1):
    """This function can plot the time against different position type. 
    plot_Type = 1 is time against x & y position 
    plot_type = 2 is time against tanginal position """
    
    if plot_type not in [1, 2]:
        raise ValueError('Unknown plot type')

    num_velocity = len(velocity)
    cols = int(np.ceil(np.sqrt(num_velocity)))
    rows = int(np.ceil(num_velocity / cols))

    fig, axs = plt.subplots(rows, cols)

    for k in range(num_velocity):
        if isinstance(axs, np.ndarray):
            ax = axs[k // cols, k % cols]

        if plot_type == 1:
            ax.plot(time[k], velocity[k])
            if k == num_velocity - 1:
                ax.legend(['v_x', 'v_y'])
        elif plot_type == 2:
            tangvel = np.sqrt(np.sum(np.square(velocity[k]), axis=1))
            ax.plot(time[k], tangvel)
    plt.show()
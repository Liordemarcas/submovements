


import numpy as np
import matplotlib.pyplot as plt

def plot_position(position, time, plot_type=1):
        """
    Plot the position data against time.

    Parameters:
        position (list): List of position data arrays.
        time (list): List of time data arrays.
        plot_type (int, optional): Type of plot to generate. Default is 1.
            - plot_type = 1: Time vs. x & y position.
            - plot_type = 2: Time vs. tangential position.

    Raises:
        ValueError: If the plot_type is unknown.

    """
    if plot_type not in [1, 2]:
        raise ValueError('Unknown plot type')

    num_positions = len(position)
    cols = int(np.ceil(np.sqrt(num_positions)))
    rows = int(np.ceil(num_positions / cols))
#create demo figure with num subplots correlated with num of trials in data
    fig, axs = plt.subplots(rows, cols)
#inserting each trial to a subplot
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





def plot_velocity(velocity, time, plot_type=1):
    """
    Plot the velocity data against time.

    Parameters:
        velocity (list): List of velocity data arrays.
        time (list): List of time data arrays.
        plot_type (int, optional): Type of plot to generate. Default is 1.
            - plot_type = 1: Time vs. v_x & v_y velocity.
            - plot_type = 2: Time vs. tangential velocity.

    Raises:
        ValueError: If the plot_type is unknown.

    """
    
    if plot_type not in [1, 2]:
        raise ValueError('Unknown plot type')

    num_velocity = len(velocity)
    cols = int(np.ceil(np.sqrt(num_velocity)))
    rows = int(np.ceil(num_velocity / cols))
#create demo figure with num subplots correlated with num of trials in data

    fig, axs = plt.subplots(rows, cols)
#inserting each trial to a subplot
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
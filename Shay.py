
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

    fig, axis = plt.subplots(rows, cols)

    for k in range(num_positions):
        if isinstance(axis, np.ndarray):
            axis = axis[k // cols, k % cols]

        if plot_type == 1:
            axis.plot(time[k], position[k])
            if k == num_positions - 1:
                axis.legend(['v_x', 'v_y'])
        elif plot_type == 2:
            tangvel = np.sqrt(np.sum(np.square(position[k]), axis=1))
            axis.plot(time[k], tangvel)

    plt.show()


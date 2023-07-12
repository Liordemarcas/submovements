import numpy as np
import matplotlib.pyplot as plt

def plot_velocity(velocity, time, plot_type=1):
    """This function can plot the time against different velocity type. 
    plot_Type = 1 is time against x & y velocity 
    plot_type = 2 is time against tanginal velocity """
    if plot_type not in [1, 2]:
        raise ValueError('Unknown plot type')

    num_velocities = len(velocity)
    cols = int(np.ceil(np.sqrt(num_velocities)))
    rows = int(np.ceil(num_velocities / cols))

    fig, axis = plt.subplots(rows, cols)

    for k in range(num_velocities):
        if isinstance(axis, np.ndarray):
            axis = axis[k // cols, k % cols]

        if plot_type == 1:
            axis.plot(time[k], velocity[k])
            if k == num_velocities - 1:
                axis.legend(['v_x', 'v_y'])
        elif plot_type == 2:
            tangvel = np.sqrt(np.sum(np.square(velocity[k]), axis=1))
            axis.plot(time[k], tangvel)

    plt.show()

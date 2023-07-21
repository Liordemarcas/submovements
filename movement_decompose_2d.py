""" module movement_decompose_2d
A module to decompose a 2d movement to multiple submovments
Contain the following functions:
    load_data            - read a folder full of csv files, 
                           collect movements position, velocities & recorded time
    plot_position        - plot movement position in time
    plot_velocity        - plot movement velocity in time
    decompose_2D         - estimate submovments parameters from movement
    plot_submovements_2D - plot the expected velocities from submovment group
Made By:
Omer Ophir:             https://github.com/omerophir
Omri FK:                https://github.com/OmriFK
Shay Eylon:             https://github.com/ShayEylon
Lior de Marcas (LdM):   https://github.com/Liordemarcas

See origin @ https://github.com/Liordemarcas/submovements
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter
from scipy.optimize import minimize

def load_data(dir_name):
    """
    Loads data from CSV files in the specified directory.

    Parameters:
        dir_name (str): Directory path containing the CSV files.
                        In each file expect:
                            Cols 0 & 1 to be X-Y position
                            Col 3 to be pen-pressure (only colecting data when it is >0)
                            Col 4 to be time
    Returns:
        position_filtered (list): List of position data arrays after filtering.
        velocity (list): List of velocity data arrays.
        time (list): List of time data arrays.

    Raises:
        ValueError: If the directory does not contain any CSV files.

    """

    # Get a list of files in the directory
    files = os.listdir(dir_name)
    # Filter only the CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    # Raise an error if no CSV files are found
    if not csv_files:
        raise ValueError('Must specify a directory to load the csv files from')
    # Extract block and trial information from file names
    blocks = []
    trials = []
    file_names = []
    for file_name in csv_files:
        file_names.append(file_name)
        match = re.search(r'tb_.*block(\d*)_trial(\d*).csv', file_name) #checking for correct file name
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
            trial_index = [i for i, (_block, _trial) in enumerate(zip(blocks, trials))
                           if _block == block and _trial == trial]
            if not trial_index:
                continue

            data = np.loadtxt(os.path.join(dir_name, csv_files[trial_index[0]]), delimiter=',')
            pressure = data[:, 3]
            position = data[pressure > 0, :2] / 1000
            _time = data[pressure > 0, 4] / 1000  # seconds
            _time = _time - _time[0]
            dt = np.median(np.diff(_time))
            b, a = butter(2, 5 / ((1 / dt) / 2))
            _position_filtered = filtfilt(b, a, position, axis=0)
            _velocity = np.vstack([[0, 0], np.diff(_position_filtered, axis=0) / dt])

            #organizing the data in correct variables for future functions
            time.append(_time)
            position_filtered.append(_position_filtered)
            velocity.append(_velocity)

    return position_filtered, velocity, time

def plot_position(position, time, plot_type=1):
    """
    Plot the position data against time.

    Parameters:
        position (list): List of position data arrays.
        time (list): List of time data arrays.
        plot_type (int, optional): Type of plot to generate. Default is 1.
            - plot_type = 1: x vs y position
            - plot_type = 2: Time vs. x & y position.

    Raises:
        ValueError: If the plot_type is unknown.

    """
    if plot_type not in [1, 2]:
        raise ValueError('Unknown plot type')

    num_positions = len(position)
    cols = int(np.ceil(np.sqrt(num_positions)))
    rows = int(np.ceil(num_positions / cols))
    #create demo figure with num subplots correlated with num of trials in data
    _ , axs = plt.subplots(rows, cols)
    #inserting each trial to a subplot
    for k in range(num_positions):
        if isinstance(axs, np.ndarray):
            ax = axs[k // cols, k % cols]

        if plot_type == 1:
            x, y = np.split(position[k],[-1], axis=1)
            ax.plot(x, y)
            ax.set_ylim([0, max(y)])
            ax.set_xlim([0, max(x)])
        elif plot_type == 2:
            ax.plot(time[k], position[k])
            if k == num_positions - 1:
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

    _ , axs = plt.subplots(rows, cols)
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

def decompose_2D(time: np.ndarray,vel: np.ndarray,
                 n_sub_movement: int = 4, x_rng: int = (-5., 5.),
                 y_rng: int = (0.1, 5)) -> tuple[float,np.ndarray,np.ndarray]:
    """
    decompose_2D - decompose two dimensional movement into submovements using the velocity profiles

    best_error, final_parms, best_velocity = decompose(time,vel,numsubmovements,xrng,yrng)

    vel should be a 2 x N matrix, with the x and y velocities

    t should be a 1 x N matrix with the corresponding time (in seconds)

    n_sub_movement is the number of submovements to look for, if it is
    empty or not specified, the function will try 1 to 4 submovements

    x_rng is the valid range for the amplitude of x values (default = (-5 5))

    y_rng is the valid range for the amplitude of y values (default = (0.1 5))

    min(t0) = 0.167 * submovement number


    best_error the best (lowest) value of the error function

    best_parameters contains the function parameters corresponding to the best values
    [start_t, movment_duration, displace_x, displace_y]. 
    If there are multiple submovements, each submovement is in different row.

    best_velocity is the velocity profile coresponding to the best values (UNIMPLANTED!!!)

    Jason Friedman, 2021
    www.curiousjason.com
    """
    # Input validation
    if time.ndim > 1:
        raise ValueError('time must be a 1D')

    if vel.shape[1] != 2:
        raise ValueError('vel must be an N*2 ndarray')

    if vel.shape[0] != time.size:
        raise ValueError('vel must match time')

    # calculate tangential velocity
    tang_vel = np.sqrt(vel[:,0]**2 + vel[:,1]**2)

    lower_bounds = np.array([0,                          0.167  , x_rng[0], y_rng[0]])
    upper_bounds = np.array([max(time[-1]-0.167,0.1),    1.     , x_rng[1], y_rng[1]])
    #submovment:             start,                     duration, Xpos,    Ypos

    if np.any(lower_bounds > upper_bounds):
        raise ValueError('Lower bounds exceed upper bound - infeasible')

    # initiate matrices for parameters and bounds for each submovment
    parm_per_sub = 4 # hard coded - can be change if different methods are used
    init_parm = np.empty(shape=(n_sub_movement,parm_per_sub),dtype=float) # submovement parameters
    all_lower_bounds = np.empty(shape=(n_sub_movement,parm_per_sub),dtype=float) # lower bound for each parameter
    all_upper_bounds = np.empty(shape=(n_sub_movement,parm_per_sub),dtype=float) # upper bound for each parameter

    # initiate best error found
    best_error = np.inf

    # try optimazation 20 times, select the time with least error
    for _ in range(20):
        # create initial parameters for each submovement
        for iSub in range(n_sub_movement):
            init_parm[iSub,:] = lower_bounds + (upper_bounds - lower_bounds)*np.random.rand(1,parm_per_sub)
            all_upper_bounds[iSub,:] = upper_bounds.copy()
            all_lower_bounds[iSub,:] = lower_bounds.copy()
            all_lower_bounds[iSub,0] = (iSub) * 0.167

        # function to minimize
        def error_fun(parms):
            epsilon = _calculate_error_MJ2D(parms, time, vel, tang_vel)
            return epsilon

        # run the optimizer
        res = minimize(error_fun,
                       x0=init_parm.flatten(),
                       method='trust-constr',
                       bounds=tuple(zip(all_lower_bounds.flatten(),all_upper_bounds.flatten())),
                       options = {'maxiter':5000})

        # calculate error for the result found
        epsilon = error_fun(res.x)

        # save result if error is smaller than best found
        if epsilon < best_error:
            best_error = epsilon
            best_parameters = res.x

    # organize parameters to so every submovment is a row
    final_parms = best_parameters.reshape((n_sub_movement,parm_per_sub))

    return best_error, final_parms

def plot_submovements_2D(parameters, t: np.ndarray = None, plot_type: int = 1) -> tuple[plt.axes, plt.figure]:
    """
    plot_submovements_2D - plot 2D submovements after decomposition

    plot_submovements_2D(parameters,t,plot_type,x0,y0)

    The parameters should in sets of 4 for each submovement:
    [start_t, movment_duration, displace_x, displace_y]


    plot_type:
    1 = time vs submovement velocity + sum velocity (default)
    2 = time vs submovement velocity 
    3 = time vs submovement position - extra parameters (x0,y0) specifies the start 
    position of the first submovement (the other submovements are assumed
    to start where the previous submovement ended) (NOT IMPLAMENTED!!)
    4 = same as 3, but without the sum (NOT IMPLAMENTED!!)
    5 = submovement position x vs y + sum (NOT IMPLAMENTED!!)
    """
    if int(len(parameters))%4 != 0:
        raise ValueError('The parameters vector must have a length that is a multiple of 4')

    # parse inputs
    numsubmovements = parameters.shape[0] # each submovment is in a different row
    start_t      = parameters[:, 0]
    movment_dur  = parameters[:, 1]
    displace_x   = parameters[:, 2]
    displace_y   = parameters[:, 3]

    # make sure parameters are ordered by movment start time
    order = np.argsort(start_t)
    start_t      = start_t[order]
    movment_dur  = movment_dur[order]
    displace_x   = displace_x[order]
    displace_y   = displace_y[order]

    # if no time was given, plot from start of first movement to end of last movment
    if t is None:
        movement_end = start_t + movment_dur # end time of each movement
        t = np.linspace(min(start_t),max(movement_end),num=100)

    # init velocities
    vel_x = np.zeros((numsubmovements,t.size))
    vel_y = np.zeros((numsubmovements,t.size))

    # using minimum jerk, find velocities curve for each submovment
    for isub in range(numsubmovements):
        vel_x[isub,:], vel_y[isub,:], _ = _minimum_jerk_velocity_2D(start_t[isub],movment_dur[isub], displace_x [isub], displace_y[isub],t)

    # get total velocity expected from submovments
    sum_vx = np.sum(vel_x,axis=0)
    sum_vy = np.sum(vel_y,axis=0)

    # create the figure
    fig, axs = plt.subplots(1, 1)
    vx_lines    = axs.plot(t,vel_x.transpose(), 'b',   label=r'$V_{x}$')
    vy_lines    = axs.plot(t,vel_y.transpose(), 'r',   label=r'$V_{y}$')
    if plot_type == 1:
        vx_sum_line = axs.plot(t,sum_vx        , 'b--', label=r'$Sum V_{x}$')
        vy_sum_line = axs.plot(t,sum_vy        , 'r--', label=r'$Sum V_{y}$')
        axs.legend(handles=[vx_lines[0], vx_sum_line[0], vy_lines[0], vy_sum_line[0]])
    else:
        axs.legend(handles=[vx_lines[0], vy_lines[0]])

    axs.set_xlabel('Time')
    axs.set_ylabel('Velocity')

    # display the figure
    #plt.show()

    # return axe & figure for later plotting
    return axs, fig

def _calculate_error_MJ2D(parameters: np.ndarray,time: np.ndarray,
                         vel: np.ndarray,tangvel: np.ndarray,
                         timedelta: float = 0.005) -> float:
    """
    Calculates the error between predicted and actual trajectories in a 
    2D space based on the given parameters.
    Parameters:
        parameters (array): List of parameters for each submovement. Each submovement requires 4 parameters: T0, D, Dx, Dy.
        time (array): Array of time values.
        vel (array): Array of velocity values.
        tangvel (array): Array of tangential velocity values.
        timedelta (float, optional): Time interval between consecutive points. Default is 0.005.

    Returns:
        epsilon (float): Error between predicted and actual trajectories.
    """

    # Calculate the number of submovements
    n_sub_movement = int(len(parameters)/4)
    # Find the last time point in the trajectory
    last_time = 0

    for i in range(n_sub_movement):

        start_t = parameters[i*4-4]
        movment_dur= parameters[i*4-3]
        last_time = max([last_time, start_t+movment_dur])
    # Adjust the last time point to align with the given time interval
    last_time = (last_time*(1/timedelta))/(1/timedelta)
    # If the last time is greater than the last time point in the given time array,
    # extend the time, velocity, and tangential velocity arrays with zeros

    if last_time > time[-1]:
        new_time = np.arange(time[-1], last_time + timedelta, timedelta)
        time = np.concatenate((time[:-1], new_time))
        vel = np.concatenate((vel, np.zeros((len(time) - len(vel), vel.shape[1]))))
        tangvel = np.concatenate((tangvel, np.zeros((len(time) - len(tangvel)))))

    # Initialize arrays for predicted trajectories, Jacobian matrices, and Hessians
    trajectory_x = vel[:,0]
    trajectory_y = vel[:,1]


    predicted_x = np.zeros([n_sub_movement, len(time)])
    predicted_y = np.zeros([n_sub_movement, len(time)])
    predicted = np.zeros([n_sub_movement, len(time)])

    #these variables were used in the Matlab code, we don't need them
    #but still left it for future change in code if needed

    # Jx = np.zeros([n_sub_movement,4*n_sub_movement, len(time)])
    # Jy = np.zeros([n_sub_movement,4*n_sub_movement, len(time)])
    # J = np.zeros([n_sub_movement,4*n_sub_movement, len(time)])

    # Hx = np.zeros([n_sub_movement,4*n_sub_movement, len(time)])
    # Hy = np.zeros([n_sub_movement,4*n_sub_movement, len(time)])
    # H = np.zeros([n_sub_movement,4*n_sub_movement, len(time)])

    # Calculate predicted trajectories for each submovement

    for i in range(n_sub_movement):
        start_t     = parameters[i*4-4]
        movment_dur = parameters[i*4-3]
        displace_x  = parameters[i*4-2]
        displace_y  = parameters[i*4-1]

        this_rgn = np.where((time > start_t) & (time < start_t+movment_dur))[0]

        predicted_x[i,this_rgn], predicted_y[i,this_rgn], predicted[i,this_rgn] = _minimum_jerk_velocity_2D(start_t, movment_dur, displace_x, displace_y, time[this_rgn])


    # Calculate the sum of predicted trajectories and actual trajectories squared
    sum_predicted_x = sum(predicted_x,1)
    sum_predicted_y = sum(predicted_y,1)
    sum_predicted  = sum(predicted,1)
    sum_traj_sq = sum(trajectory_x**2 + trajectory_y**2 + tangvel**2)

    # Calculate the error between predicted and actual trajectories
    epsilon = np.sum((sum_predicted_x - trajectory_x)**2 + (sum_predicted_y - trajectory_y)**2 + (sum_predicted - tangvel)**2) / np.sum(sum_traj_sq)

    return epsilon

def _minimum_jerk_velocity_2D(start_t: float,movment_dur: float,
                              displace_x: float,displace_y: float,
                              t: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    minimumJerkVelocity2D - evaluate a minimum jerk velocity curve with separate displacement for x / y

    see Flash and Hogan (1985) for details on the minimum jerk equation

        start_t = movement start time (scalar)
        movment_dur  = movement duration (scalar)
        displace_x   = displacement resulting from the movement (x) (scalar)
        displace_y   = displacement resulting from the movement (y) (scalar)

    The function is evaluated at times t (vector)

    The function also optionally returns the first-order and second-order
    partial derivatives, for use with optimization routines (NOT IMPLAMENTED!!)

    x_vel, y_vel and tan_vel are the x velocity, y velocity and tangential velocities
    Jx, Jy and J are the gradients (partial derivatives) of the same quantities (NOT IMPLAMENTED!!)
    Hx, Hy and H are the Hessian (second-order partial derivatives) (NOT IMPLAMENTED!!)
    """

    # normalise time to t0 and movement duration, take only the time of the movement
    normlized_time = (t - start_t)/movment_dur
    logical_movement = (normlized_time >= 0) & (normlized_time <= 1)

    # normalise displacement to movment duration
    norm_disp_x = displace_x/movment_dur
    norm_disp_y = displace_y/movment_dur
    tang_norm_disp = np.sqrt(norm_disp_x**2 + norm_disp_y**2)

    # make x_vel, y_vel & tan_vel that are zero outside of calculated area
    x_vel = np.zeros(t.size)
    y_vel = np.zeros(t.size)
    tan_vel  = np.zeros(t.size)

    # calculate velocities
    def min_jerk_2d_fun(base_val):
        # the polynomial function from Flash and Hogan (1985)
        return base_val * (-60*normlized_time[logical_movement]**3 + 30*normlized_time[logical_movement]**4 + 30*normlized_time[logical_movement]**2)

    x_vel[logical_movement]   = min_jerk_2d_fun(norm_disp_x)
    y_vel[logical_movement]   = min_jerk_2d_fun(norm_disp_y)
    tan_vel[logical_movement] = min_jerk_2d_fun(tang_norm_disp)

    return x_vel, y_vel, tan_vel

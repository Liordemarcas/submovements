import numpy as np
from minimumJerkVelocity2D__file import min_jerk_velocity_2D

def calculate_error_MJ2D(parameters,time,vel,tangvel,timedelta = 0.005):
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

        t0 = parameters[i*4-4]
        d= parameters[i*4-3]
        last_time = max([last_time, t0+d])
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
        t0 = parameters[i*4-4]
        d= parameters[i*4-3]
        dx = parameters[i*4-2]
        dy = parameters[i*4-1]

        this_rgn = np.where((time > t0) & (time < t0+d))[0]

        predicted_x[i,this_rgn], predicted_y[i,this_rgn], predicted[i,this_rgn] = min_jerk_velocity_2D(t0, d, dx, dy, time[this_rgn])

    
    # Calculate the sum of predicted trajectories and actual trajectories squared
    sum_predicted_x = sum(predicted_x,1)
    sum_predicted_y = sum(predicted_y,1)
    sum_predicted  = sum(predicted,1)
    sum_traj_sq = sum(trajectory_x**2 + trajectory_y**2 + tangvel**2)
   
    # Calculate the error between predicted and actual trajectories
    epsilon = np.sum((sum_predicted_x - trajectory_x)**2 + (sum_predicted_y - trajectory_y)**2 + (sum_predicted - tangvel)**2) / np.sum(sum_traj_sq)
 
    return(epsilon)
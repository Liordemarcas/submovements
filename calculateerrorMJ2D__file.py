import numpy as np
from minimumJerkVelocity2D import minimumJerkVelocity2D

def calculateerrorMJ2D(parameters,time,vel,tangvel,timedelta = 0.005):
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
    numsubmovements = int(len(parameters)/4)
    # Find the last time point in the trajectory
    last_time = 0
    
    for i in range(numsubmovements):

        T0 = parameters[i*4-4]
        D = parameters[i*4-3]
        last_time = max([last_time, T0+D])
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

    predicted_x = np.zeros([numsubmovements, len(time)])
    predicted_y = np.zeros([numsubmovements, len(time)])
    predicted = np.zeros([numsubmovements, len(time)])
#these variables were used in the Matlab code, we don't need them
#but still left it for future change in code if needed

    # Jx = np.zeros([numsubmovements,4*numsubmovements, len(time)])
    # Jy = np.zeros([numsubmovements,4*numsubmovements, len(time)])
    # J = np.zeros([numsubmovements,4*numsubmovements, len(time)])

    # Hx = np.zeros([numsubmovements,4*numsubmovements, len(time)])
    # Hy = np.zeros([numsubmovements,4*numsubmovements, len(time)])
    # H = np.zeros([numsubmovements,4*numsubmovements, len(time)])

# Calculate predicted trajectories for each submovement

    for i in range(numsubmovements):
        T0 = parameters[i*4-4]
        D = parameters[i*4-3]
        Dx = parameters[i*4-2]
        Dy = parameters[i*4-1]

        thisrgn = np.where((time > T0) & (time < T0+D))[0]

        predicted_x[i,thisrgn], predicted_y[i,thisrgn], predicted[i,thisrgn] = minimumJerkVelocity2D(T0, D, Dx, Dy, time[thisrgn])

    
    # Calculate the sum of predicted trajectories and actual trajectories squared
    sumpredictedx = sum(predicted_x,1)
    sumpredictedy = sum(predicted_y,1)
    sumpredicted  = sum(predicted,1)
    sumtrajsq = sum(trajectory_x**2 + trajectory_y**2 + tangvel**2)
   
    # Calculate the error between predicted and actual trajectories
    epsilon = np.sum((sumpredictedx - trajectory_x)**2 + (sumpredictedy - trajectory_y)**2 + (sumpredicted - tangvel)**2) / np.sum(sumtrajsq)
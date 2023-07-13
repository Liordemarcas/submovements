import numpy as np
#from submovment_file import submovment
from calculateerrorMJ2D__file import calculate_error_MJ2D
from scipy.optimize import minimize
from typing import Tuple

def decompose_2D(time: np.ndarray,vel: np.ndarray,n_sub_movement: int = 4,x_rng: int = (-5., 5.),y_rng: int = (0.1, 5)) \
    -> Tuple[float,np.ndarray,np.ndarray]:
    """
    DECOMPOSE - decompose two dimensional movement into submovements using the velocity profiles

    [best,bestParameters,bestVelocity] = decompose(time,vel,numsubmovements,xrng,yrng)

    vel should be a 2 x N matrix, with the x and y velocities

    t should be a 1 x N matrix with the corresponding time (in seconds)

    numsubmovements is the number of submovements to look for, if it is
    empty or not specified, the function will try 1 to 4 submovements

    xrng is the valid range for the amplitude of x values (default = [-5 5])

    yrng is the valid range for the amplitude of y values (default = [0.1 5])

    min(t0) = 0.167 * submovement number


    bestError the best (lowest) value of the error function

    bestParameters contains the function parameters corresponding to the best values
    [t0 D Ax Ay]. If there are multiple submovements, it will be have a
    length of 4*numsubmovements

    bestVelocity is the velocity profile coresponding to the best values (UNIMPLANTED!!!)

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
    parm_per_sub = 4 # hard coded - can be change if diffrent methods are used
    init_parm = np.empty(shape=(n_sub_movement,parm_per_sub),dtype=float) # submovement parameters
    all_lower_bounds = np.empty(shape=(n_sub_movement,parm_per_sub),dtype=float) # lower bound for each parameter
    all_upper_bounds = np.empty(shape=(n_sub_movement,parm_per_sub),dtype=float) # upper bound for each parameter

    # initate best error found
    best_error = np.inf

    # try optimazation 20 times, select the time with least error
    for iTry in range(20):
        # create inital parameters for each submovement
        for iSub in range(n_sub_movement):
            init_parm[iSub,:] = lower_bounds + (upper_bounds - lower_bounds)*np.random.rand(1,parm_per_sub)
            all_upper_bounds[iSub,:] = upper_bounds.copy()
            all_lower_bounds[iSub,:] = lower_bounds.copy()
            all_lower_bounds[iSub,0] = (iSub-1) * 0.167
            #all_subs[iSub].parms = lower_bounds + (upper_bounds - lower_bounds)*np.random.rand(1,parm_per_sub)
            #all_subs[iSub].lower_bounds = lower_bounds.copy()
            #all_subs[iSub].lower_bounds[0] = (iSub-1) * 0.167
            #all_subs[iSub].upper_bounds = upper_bounds

        # function to minimaize
        #error_fun = lambda parms: calculateerrorMJ2D(parms, time, vel, tangvel)
        def error_fun(parms):
            epsilon = calculate_error_MJ2D(parms, time, vel, tang_vel)
            return epsilon

        # run the optimazer
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

    # orginaze parameters to so every submovment is a row
    final_parms = best_parameters.reshape((n_sub_movement,parm_per_sub))

    return best_error, final_parms
        
import numpy as np
#from submovment_file import submovment
from calculateerrorMJ2D_file import calculateerrorMJ2D
from scipy.optimize import minimize


def decompose2D(time: np.ndarry,vel: np.ndarray,numsubmovements: int = 4,xrng: int = (-5., 5.),yrng: int = (0.1, 5)) \
    -> tuple(float,np.ndarray,np.narray):
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

    bestVelocity is the velocity profile coresponding to the best values

    Jason Friedman, 2021
    www.curiousjason.com
    """

    # Input validation
    if time.shape[0] > 1:
        raise ValueError('time must be a 1*N vector')

    if vel.shape[0] != 2:
        raise ValueError('vel must be an 2*N matrix')

    if vel.shape[1] != time.size:
        raise ValueError('vel must match time')
    
    lower_bounds = np.array([0,                          0.167  , xrng[0], yrng[0]])
    upper_bounds = np.array([max(time[-1]-0.167,0.1),    1.     , xrng[1], yrng[1]])
    #submovment:             start,                     duration, Xpos,    Ypos

    if np.any(lower_bounds > upper_bounds):
        raise ValueError('Lower bounds exceed upper bound - infeasible')
    
    # initiate matrices for parameters and bounds for each submovment
    parm_per_sub = 4 # hard coded - can be change if diffrent methods are used
    init_parm = np.empty(shape=(numsubmovements,parm_per_sub),dtype=float) # submovement parameters
    all_lower_bounds = np.empty(shape=(numsubmovements,parm_per_sub),dtype=float) # lower bound for each parameter
    all_upper_bounds = np.empty(shape=(numsubmovements,parm_per_sub),dtype=float) # upper bound for each parameter

    # try optimazation 20 times, select the time with least error
    for iTry in range(1,20,1):
        # create inital parameters for each submovement
        for iSub in range(numsubmovements):
            init_parm[iSub,:] = lower_bounds + (upper_bounds - lower_bounds)*np.random.rand(1,parm_per_sub)
            all_upper_bounds[iSub,:] = upper_bounds.copy()
            all_lower_bounds[iSub,:] = lower_bounds.copy()
            all_lower_bounds[iSub,0] = (iSub-1) * 0.167
            #all_subs[iSub].parms = lower_bounds + (upper_bounds - lower_bounds)*np.random.rand(1,parm_per_sub)
            #all_subs[iSub].lower_bounds = lower_bounds.copy()
            #all_subs[iSub].lower_bounds[0] = (iSub-1) * 0.167
            #all_subs[iSub].upper_bounds = upper_bounds
        
        # calculate tangential velocity
        tangvel = np.sqrt(vel[1,:]**2 + vel[2,:]**2)

        # function to minimaize
        error_fun = lambda parms: calculateerrorMJ2D(parms, time, vel, tangvel)

        # run the optimazer
        res = minimize(error_fun,
                       x0=init_parm.flatten(), 
                       method='trust-constr',
                       bounds=tuple(zip(all_lower_bounds.flatten(),all_upper_bounds.flatten())),
                       options = {'maxiter':5000, 'max_nfev': 10**13})
        
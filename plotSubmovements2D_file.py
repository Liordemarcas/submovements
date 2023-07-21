import numpy as np
from minimumJerkVelocity2D import minimumJerkVelocity2D
import matplotlib.pyplot as plt


def plot_submovements_2D(parameters,x0: float = 0, y0: float = 0, t: np.ndarray = None) \
    -> tuple[plt.axes, plt.figure]:
    """
    PLOTSUBMOVEMENTS2D - plot 2D submovements after decomposition

    plotSubmovements2D(parameters,t,plottype,x0,y0)

    The parameters should in sets of 4 for each submovement:
    [t0 D Ax Ay]


    plottype:
    1 = time vs submovement velocity + sum velocity (default)
    2 = time vs submovement velocity
    3 = time vs submovement position - extra parameters (x0,y0) specifies the start
    position of the first submovement (the other submovements are assumed
    to start where the previous submovement ended)
    4 = same as 3, but without the sum
    5 = submovement position x vs y + sum
    """
    if int(len(parameters))%4 != 0:
        raise ValueError('The parameters vector must have a length that is a multiple of 4')

    # parse inputs
    numsubmovements = parameters.shape[0]
    t0 = parameters[:, 0]
    D  = parameters[:, 1]
    Ax = parameters[:, 2]
    Ay = parameters[:, 3]

    # make sure parameters are ordered by movment start time
    order = np.argsort(t0)
    t0 = t0[order]
    D  = D[order]
    Ax = Ax[order]
    Ay = Ay[order]

    # get x and y movement from displacement
    x0 = np.concatenate((x0,x0 + np.cumsum(Ax[0:-1])))
    y0 = np.concatenate((y0,y0 + np.cumsum(Ay[0:-1])))
   
    # if no time was given, plot from start of first movement to end of last movment
    if t is None:
        tf = t0 + D # end time of each movement
        t = np.linspace(min(t0),max(tf),num=100)

    # init velocities
    vx = np.zeros((numsubmovements,t.size))
    vy = np.zeros((numsubmovements,t.size))

    # using minimum jerk, find velocities curve for each submovment
    for isub in range(numsubmovements):
        vx[isub,:], vy[isub,:], _ = minimumJerkVelocity2D(t0[isub],D[isub], Ax [isub], Ay[isub],t)
    
    # get total velocity expected from submovments
    sum_vx = np.sum(vx,axis=0)
    sum_vy = np.sum(vy,axis=0)

    # create the figure
    fig, axs = plt.subplots(1, 1)
    vx_lines    = axs.plot(t,vx.transpose(), 'b',   label=r'$V_{x}$')
    vx_sum_line = axs.plot(t,sum_vx        , 'b--', label=r'$Sum V_{x}$')
    vy_lines    = axs.plot(t,vy.transpose(), 'r',   label=r'$V_{y}$')
    vy_sum_line = axs.plot(t,sum_vy        , 'r--', label=r'$Sum V_{y}$')
    axs.legend(handles=[vx_lines[0], vx_sum_line[0], vy_lines[0], vy_sum_line[0]])
    axs.set_xlabel('Time')
    axs.set_ylabel('Velocity')

    # display the figure
    #plt.show()

    # return axe & figure for later ploting
    return axs, fig
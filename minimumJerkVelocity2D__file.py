import numpy as np
from typing import Tuple

def min_jerk_velocity_2D(t0: float,d: float,ax: float,ay: float,t: np.ndarray):
    """
    minimumJerkVelocity2D - evaluate a minimum jerk velocity curve with seperate displacement for x / y

    see Flash and Hogan (1985) for details on the minimum jerk equation

        t0 = movement start time
        D  = movement duration
        Ax = displacement resulting from the movement (x)
        Ay = displacement resulting from the movement (y)

    The function is evaluated at times t

    The function also optionally returns the first-order and second-order
    partial derivatives, for use with optimization routines (NOT IMPLAMENTED!!)

    Bx, y_vel and B are the x velocity, y velocity and tangential velocities
    Jx, Jy and J are the gradients (partial derivatives) of the same quantities (NOT IMPLAMENTED!!)
    Hx, Hy and H are the Hessian (second-order partial derivatives) (NOT IMPLAMENTED!!)

    [Bx,y_vel,B,Jx,Jy,J,Hx,Hy,H] = minimumJerkVelocity2D(t0,D,Ax,Ay,t)
    """

    # normalise time to t0 and movement duration, take only the time of the movement
    normlized_time= (t - t0)/d
    logical_movement = (normlized_time >= 0) & (normlized_time <= 1)

    # make Bx, y_vel & B that are zero outside of calculated area
    # % Bx, y_vel and B are the x velocity, y velocity and tangential velocities
    x_vel = y_vel = tan_vel = np.zeros(t.size)

    
    x_vel[logical_movement] = ax/d * (-60 * normlized_time[logical_movement]**3 + 30 * normlized_time[logical_movement]**4 + 30 * normlized_time[logical_movement]**2)
    y_vel[logical_movement] = ay/d * (-60 * normlized_time[logical_movement]**3 + 30 * normlized_time[logical_movement]**4 + 30 * normlized_time[logical_movement]**2)

    a_tan = np.sqrt((ax/d)**2 + (ay/d)**2)
    tan_vel[logical_movement] = a_tan * (-60 * normlized_time[logical_movement]**3 + 30 * normlized_time[logical_movement]**4 + 30 * normlized_time[logical_movement]**2)
    return x_vel, y_vel, tan_vel
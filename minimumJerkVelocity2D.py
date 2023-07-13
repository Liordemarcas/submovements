import numpy as np
from typing import Tuple

def minimumJerkVelocity2D(t0: float,D: float,Ax: float,Ay: float,t: np.ndarray):
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

    Bx, By and B are the x velocity, y velocity and tangential velocities
    Jx, Jy and J are the gradients (partial derivatives) of the same quantities (NOT IMPLAMENTED!!)
    Hx, Hy and H are the Hessian (second-order partial derivatives) (NOT IMPLAMENTED!!)

    [Bx,By,B,Jx,Jy,J,Hx,Hy,H] = minimumJerkVelocity2D(t0,D,Ax,Ay,t)
    """

    # normalise time to t0 and movement duration, take only the time of the movement
    nt = (t - t0)/D
    r = (nt >= 0) & (nt <= 1)

    # make Bx, By & B that are zero outside of calculated area
    Bx = By = B = np.zeros(t.size)

    # calcu


    Bx[r] = Ax/D * (-60 * nt[r]**3 + 30 * nt[r]**4 + 30 * nt[r]**2)
    By[r] = Ay/D * (-60 * nt[r]**3 + 30 * nt[r]**4 + 30 * nt[r]**2)

    A_tang = np.sqrt((Ax/D)**2 + (Ay/D)**2)
    B[r] = A_tang * (-60 * nt[r]**3 + 30 * nt[r]**4 + 30 * nt[r]**2)
    return Bx, By, B